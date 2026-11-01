"""Price-shape TCN backend with a bounded, restricted Torch state codec."""

import hashlib
import pickle
import platform
import time
import zipfile
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Literal, cast

import numpy as np
import torch
from pydantic import Field, JsonValue
from torch import Tensor, nn

from regimex.contracts import Seed, StrictModel
from regimex.directions import DIRECTION_NAMES, AnalysisConfig
from regimex.features import DoubleArray, compute_features
from regimex.model_bundles import BundleMember, BundleRole, ModelBundle, verify_model_bundle
from regimex.model_contracts import (
    BundleProvenance,
    FitResult,
    ObservationInputs,
    SelectionEvaluator,
    TrainingInputs,
    require_observation_inputs,
)
from regimex.model_registry import (
    InferenceRuntime,
    TCNParameters,
    TrainingPolicy,
    TrainingRuntime,
)
from regimex.prediction_inputs import PredictionInputs, PreparedPredictionInputs
from regimex.resource_limits import ResourceBudgetExceeded
from regimex.segmentation import DECODER_VERSION
from regimex.storage import digest, json_bytes, output_directory, write_json
from regimex.temporal import (
    CONTEXT_SIZES,
    SELECTION_BATCH_SIZE,
    TEMPORAL_NAMES,
    TEMPORAL_NUMERIC_NAMES,
    TCNClassifier,
    TemporalPreprocessor,
    fit_tcn,
    predict_tcn,
    temporal_features,
)

ARCHITECTURE_ID = "tcn"
IMPLEMENTATION_VERSION = "0.2.0"
FEATURE_VERSION = "temporal-price-shape/1.0"
PREPROCESSING_VERSION = "temporal-standardization/1.0"
CODEC_VERSION: Literal["temporal-tcn/2.0"] = "temporal-tcn/2.0"
_JSON_LIMIT = 16 * 1024**2
_MEMBERS: dict[BundleRole, str] = {
    "configuration": "configuration.json",
    "preprocessing": "preprocessing.npz",
    "weights": "weights.pt",
    "validation": "validation.json",
    "history": "history.json",
}


class TCNConfiguration(StrictModel):
    codec_version: Literal["temporal-tcn/2.0"] = CODEC_VERSION
    loss_version: Literal["direction-cross-entropy/2.0"]
    smoothing_scope: Literal["equal_adjacent_targets"]
    boundary_weight_radius: Literal[4]
    boundary_weight_multiplier: Literal[2]
    parameters: TCNParameters
    training_policy: TrainingPolicy
    training_runtime: TrainingRuntime
    analysis: AnalysisConfig
    feature_names: tuple[str, ...] = TEMPORAL_NAMES
    context_sizes: tuple[int, ...] = CONTEXT_SIZES
    kernel_size: Literal[3] = 3
    gradient_clip: Literal[1] = 1
    smoothing_change_clip: Literal[16] = 16
    selection_batch_size: Literal[16] = SELECTION_BATCH_SIZE
    environment: dict[str, JsonValue]


class EpochRecord(StrictModel):
    seed: Seed
    epoch: int = Field(ge=1, le=1000)
    training_loss: float = Field(ge=0)
    validation_segment_macro_f1: float | None = Field(default=None, ge=0, le=1)


class HistoryRecord(StrictModel):
    epochs: tuple[EpochRecord, ...] = Field(min_length=1, max_length=32_000)


class ValidationRecord(StrictModel):
    metric: Literal["decoded_segment_macro_f1/iou-0.50"] = "decoded_segment_macro_f1/iou-0.50"
    score: float = Field(ge=0, le=1)
    selected_seed: Seed
    selected_epoch: int = Field(ge=1, le=1000)


def _validate_policy(policy: TrainingPolicy) -> None:
    TrainingPolicy.model_validate(policy.model_dump())
    if policy.optimizer != "adamw":
        raise ValueError("TCN supports only the AdamW optimizer")
    if len(policy.seeds) > 32:
        raise ValueError("TCN supports at most 32 search seeds")


def _environment() -> dict[str, JsonValue]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: version(name) for name in ("regimex", "numpy", "torch")},
        "torch_cpu_capability": str(torch.backends.cpu.get_cpu_capability()),
    }


def _feature_inputs(observations: ObservationInputs) -> PredictionInputs:
    require_observation_inputs(observations)
    # This conversion is private to this backend. The first two float32 columns
    # preserve the original relative-log-price/adjacent-return representation.
    return PredictionInputs(
        np.concatenate([compute_features(rows) for rows in observations.observations]),
        np.r_[0, np.cumsum([item.step_count for item in observations.info])].astype(np.int64),
        observations.info,
    )


def _prepared(
    observations: ObservationInputs, preprocessing: TemporalPreprocessor
) -> PreparedPredictionInputs:
    raw = _feature_inputs(observations)
    return PreparedPredictionInputs(
        preprocessing.transform(temporal_features(raw)), raw.offsets, raw.info
    )


def _state_shapes(parameters: TCNParameters) -> dict[str, tuple[int, ...]]:
    hidden = parameters.channels
    shapes = {
        "input.weight": (hidden, len(TEMPORAL_NAMES), 1),
        "input.bias": (hidden,),
        "head.weight": (len(DIRECTION_NAMES), hidden, 1),
        "head.bias": (len(DIRECTION_NAMES),),
    }
    for index in range(len(parameters.dilations)):
        shapes.update(
            {
                f"blocks.{index}.conv.weight": (hidden, hidden, 3),
                f"blocks.{index}.conv.bias": (hidden,),
                f"blocks.{index}.mix.weight": (hidden, hidden, 1),
                f"blocks.{index}.mix.bias": (hidden,),
            }
        )
    return shapes


def _state_bytes(parameters: TCNParameters) -> int:
    return sum(int(np.prod(shape)) * 4 for shape in _state_shapes(parameters).values())


def _activation_bytes(parameters: TCNParameters, batch_size: int, step_count: int) -> int:
    """Bound padded activations and convolution workspaces for one phase."""
    return (
        batch_size
        * (step_count + 2 * max(parameters.dilations))
        * parameters.channels
        * (len(parameters.dilations) + 1)
        * 4
        * 8
    )


def _freeze(array: DoubleArray, *, positive: bool = False) -> DoubleArray:
    if (
        array.dtype != np.float64
        or array.shape != (len(TEMPORAL_NAMES),)
        or not np.isfinite(array).all()
        or (positive and (array <= 0).any())
    ):
        raise ValueError("TCN preprocessing has incompatible dtype, shape, finiteness or scale")
    result = array.copy()
    result.flags.writeable = False
    return result


def _validate_state(state: object, parameters: TCNParameters) -> dict[str, Tensor]:
    shapes = _state_shapes(parameters)
    if not isinstance(state, dict) or state.keys() != shapes.keys():
        raise ValueError("TCN state dictionary members differ from its architecture")
    result: dict[str, Tensor] = {}
    for name, tensor in state.items():
        if (
            type(tensor) is not Tensor
            or tensor.dtype != torch.float32
            or tensor.device.type != "cpu"
            or tensor.layout != torch.strided
            or tuple(tensor.shape) != shapes[name]
            or not tensor.is_contiguous()
            or not bool(torch.isfinite(tensor).all())
            or tensor.untyped_storage().nbytes() != tensor.numel() * tensor.element_size()
        ):
            raise ValueError(
                "TCN weight dtype, device, shape, storage or finiteness is incompatible"
            )
        result[name] = tensor
    return result


@dataclass(frozen=True, slots=True)
class TCNPredictor:
    preprocessor: TemporalPreprocessor
    model: TCNClassifier
    configuration: TCNConfiguration
    architecture_id: str = ARCHITECTURE_ID
    implementation_version: str = IMPLEMENTATION_VERSION
    class_names: tuple[str, ...] = DIRECTION_NAMES

    def __post_init__(self) -> None:
        TCNConfiguration.model_validate_json(self.configuration.model_dump_json())
        if (
            self.configuration.feature_names != TEMPORAL_NAMES
            or self.configuration.context_sizes != CONTEXT_SIZES
            or tuple(cast(nn.Conv1d, block.conv).dilation[0] for block in self.model.blocks)
            != self.configuration.parameters.dilations
            or any(
                cast(nn.Dropout, block.dropout).p != self.configuration.parameters.dropout
                for block in self.model.blocks
            )
        ):
            raise ValueError("TCN model and feature configuration disagree")
        preprocessing = TemporalPreprocessor(
            _freeze(self.preprocessor.fill),
            _freeze(self.preprocessor.mean),
            _freeze(self.preprocessor.scale, positive=True),
        )
        mask = len(TEMPORAL_NUMERIC_NAMES)
        if (
            preprocessing.fill[mask:].any()
            or preprocessing.mean[mask:].any()
            or not np.array_equal(preprocessing.scale[mask:], np.ones(mask))
        ):
            raise ValueError("TCN preprocessing must preserve explicit missing-value masks")
        _validate_state(self.model.state_dict(), self.configuration.parameters)
        self.model.requires_grad_(False)
        self.model.eval()
        object.__setattr__(self, "preprocessor", preprocessing)

    @property
    def analysis(self) -> AnalysisConfig:
        return self.configuration.analysis

    @property
    def numeric_bytes(self) -> int:
        return _state_bytes(self.configuration.parameters) + 3 * len(TEMPORAL_NAMES) * 8

    def predict(self, inputs: ObservationInputs, *, runtime: InferenceRuntime) -> DoubleArray:
        require_observation_inputs(inputs)
        InferenceRuntime.model_validate(runtime.model_dump())
        if self.numeric_bytes > runtime.model_cache_budget_bytes:
            raise ResourceBudgetExceeded("TCN model exceeds the inference model cache budget")
        # These process-global settings are applied only inside computation workers.
        torch.set_num_threads(runtime.threads)
        torch.use_deterministic_algorithms(True)
        result = predict_tcn(
            self.model, _prepared(inputs, self.preprocessor), runtime.max_batch_size
        )
        if (
            not np.isfinite(result).all()
            or (result < 0).any()
            or not np.allclose(result.sum(axis=1), 1, rtol=1e-12, atol=1e-12)
        ):
            raise ValueError("TCN returned invalid original-grid probabilities")
        return result


def _load_json[T: StrictModel](path: Path, model: type[T]) -> T:
    with path.open("rb") as stream:
        data = stream.read(_JSON_LIMIT + 1)
    if len(data) > _JSON_LIMIT:
        raise ValueError("TCN JSON member exceeds its codec size limit")
    return model.model_validate_json(data)


def _read_preprocessing(path: Path) -> TemporalPreprocessor:
    arrays: dict[str, DoubleArray] = {}
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            if len(members) != 3 or {item.filename for item in members} != {
                name + ".npy" for name in ("fill", "mean", "scale")
            }:
                raise ValueError("TCN preprocessing archive has unexpected members")
            for member in members:
                if (
                    member.compress_type != zipfile.ZIP_STORED
                    or member.file_size > len(TEMPORAL_NAMES) * 8 + 4096
                    or member.flag_bits & 1
                ):
                    raise ValueError("TCN preprocessing archive encoding or size is incompatible")
                with archive.open(member) as stream:
                    if np.lib.format.read_magic(stream) != (1, 0):
                        raise ValueError("TCN preprocessing requires NPY format 1.0")
                    shape, fortran, dtype = np.lib.format.read_array_header_1_0(
                        stream, max_header_size=4096
                    )
                    if shape != (len(TEMPORAL_NAMES),) or fortran or dtype != np.dtype(np.float64):
                        raise ValueError("TCN preprocessing dtype or shape is incompatible")
                    data = stream.read(len(TEMPORAL_NAMES) * 8 + 1)
                    if len(data) != len(TEMPORAL_NAMES) * 8:
                        raise ValueError("TCN preprocessing array size is incompatible")
                    arrays[member.filename[:-4]] = np.frombuffer(data, dtype=np.float64).copy()
    except (OSError, zipfile.BadZipFile, EOFError) as exc:
        raise ValueError("invalid TCN preprocessing archive") from exc
    return TemporalPreprocessor(arrays["fill"], arrays["mean"], arrays["scale"])


def _read_state(path: Path, parameters: TCNParameters) -> dict[str, Tensor]:
    expected_bytes = _state_bytes(parameters)
    tensor_count = len(_state_shapes(parameters))
    # Torch's restricted loader still needs resource limits. Only its plain,
    # uncompressed state-dictionary ZIP format is accepted by this codec.
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            names = [member.filename for member in members]
            roots = {name.split("/", 1)[0] for name in names}
            if (
                len(set(names)) != len(names)
                or len(roots) != 1
                or not names
                or any(
                    "\\" in name
                    or any(part in {"", ".", ".."} for part in name.split("/"))
                    or "/" not in name
                    for name in names
                )
                or len(members) > tensor_count + 16
                or sum(member.file_size for member in members) > expected_bytes + 64 * 1024
                or any(
                    member.compress_type != zipfile.ZIP_STORED or member.flag_bits & 1
                    for member in members
                )
            ):
                raise ValueError("TCN weight archive members, encoding or size are incompatible")
            root = next(iter(roots))
            relative = {name.removeprefix(root + "/") for name in names}
            expected_data = {f"data/{index}" for index in range(tensor_count)}
            metadata = {
                "data.pkl",
                "byteorder",
                "version",
                ".data/serialization_id",
                ".format_version",
                ".storage_alignment",
            }
            if not expected_data <= relative or not relative <= expected_data | metadata:
                raise ValueError(
                    "TCN weights must contain exactly the configured plain tensor state"
                )
        state = torch.load(path, weights_only=True, map_location="cpu")
        return _validate_state(state, parameters)
    except (OSError, pickle.UnpicklingError, EOFError, zipfile.BadZipFile, RuntimeError) as exc:
        raise ValueError("invalid restricted TCN weights") from exc


class TCNBackend:
    architecture_id = ARCHITECTURE_ID
    implementation_version = IMPLEMENTATION_VERSION

    def fit(
        self,
        training: TrainingInputs,
        validation: ObservationInputs,
        *,
        parameters: StrictModel,
        policy: TrainingPolicy,
        analysis: AnalysisConfig,
        runtime: TrainingRuntime,
        evaluator: SelectionEvaluator,
    ) -> FitResult:
        require_observation_inputs(validation)
        if type(training) is not TrainingInputs or type(parameters) is not TCNParameters:
            raise TypeError("TCN requires TrainingInputs and TCNParameters")
        TCNParameters.model_validate(parameters.model_dump())
        TrainingRuntime.model_validate(runtime.model_dump())
        AnalysisConfig.model_validate(analysis.model_dump())
        _validate_policy(policy)
        if {item.experiment_id for item in training.observations.info} & {
            item.experiment_id for item in validation.info
        }:
            raise ValueError("TCN training and validation experiments must be disjoint")
        rows = training.observations.row_count + validation.row_count
        training_activations = _activation_bytes(
            parameters,
            min(policy.batch_size, len(training.observations.info)),
            max(info.step_count for info in training.observations.info),
        )
        validation_activations = _activation_bytes(
            parameters,
            min(SELECTION_BATCH_SIZE, len(validation.info)),
            max(info.step_count for info in validation.info),
        )
        # Include feature temporaries, AdamW state and saved gradients/checkpoints,
        # Training and validation run sequentially but use different batch sizes
        # and sequence lengths. Preserve the fixed validation batch policy: silently
        # shrinking batches could change floating-point results and checkpoint ties.
        # The child process provides hard OS limits in addition to this estimate.
        required = (
            rows * len(TEMPORAL_NAMES) * 8 * 8
            + _state_bytes(parameters) * 10
            + max(training_activations, validation_activations)
        )
        if required > runtime.memory_budget_bytes:
            raise ResourceBudgetExceeded(
                "TCN feature and model allocations exceed the training memory budget"
            )
        started = time.monotonic()

        def check_budget() -> None:
            if time.monotonic() - started >= runtime.time_budget_seconds:
                raise ResourceBudgetExceeded("TCN exceeded the training time budget")

        def evaluate(probabilities: DoubleArray) -> float:
            check_budget()
            score = evaluator(probabilities)
            if not np.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("TCN validation score must be segment macro F1 in [0, 1]")
            return score

        raw = _feature_inputs(training.observations)
        values = temporal_features(raw)
        preprocessing = TemporalPreprocessor.fit(values)
        prepared_training = PreparedPredictionInputs(
            preprocessing.transform(values), raw.offsets, raw.info
        )
        prepared_validation = _prepared(validation, preprocessing)
        check_budget()
        model, history, selected_seed = fit_tcn(
            prepared_training,
            training.targets,
            prepared_validation,
            parameters=parameters,
            policy=policy,
            runtime=runtime,
            scorer=evaluate,
            check_budget=check_budget,
        )
        check_budget()
        selected = max(
            (entry for entry in history if "validation_segment_macro_f1" in entry),
            key=lambda entry: cast(float, entry["validation_segment_macro_f1"]),
        )
        environment = _environment()
        configuration = TCNConfiguration(
            loss_version="direction-cross-entropy/2.0",
            smoothing_scope="equal_adjacent_targets",
            boundary_weight_radius=4,
            boundary_weight_multiplier=2,
            parameters=parameters,
            training_policy=policy,
            training_runtime=runtime,
            analysis=analysis,
            environment=environment,
        )
        return FitResult(
            predictor=TCNPredictor(preprocessing, model, configuration),
            history=tuple(history),
            selected_seed=selected_seed,
            validation_score=cast(float, selected["validation_segment_macro_f1"]),
            selected_settings={"epoch": selected["epoch"]},
            environment=environment,
        )

    @staticmethod
    def _validate_selection(
        configuration: TCNConfiguration, validation: ValidationRecord, history: HistoryRecord
    ) -> None:
        _validate_policy(configuration.training_policy)
        policy = configuration.training_policy
        seeds: list[int] = []
        previous_epoch = 0
        seed_best = -1.0
        stale = 0
        for entry in history.epochs:
            if not seeds or entry.seed != seeds[-1]:
                if seeds and previous_epoch != policy.epochs and stale < policy.patience:
                    raise ValueError("TCN history ends a seed before its stopping condition")
                seeds.append(entry.seed)
                previous_epoch = 0
                seed_best = -1.0
                stale = 0
            elif stale >= policy.patience:
                raise ValueError("TCN history continues after its stopping condition")
            if entry.epoch != previous_epoch + 1 or entry.epoch > policy.epochs:
                raise ValueError("TCN history epochs differ from the configured search")
            scored = (
                entry.epoch == 1
                or entry.epoch % policy.validation_interval == 0
                or entry.epoch == policy.epochs
            )
            if scored != (entry.validation_segment_macro_f1 is not None):
                raise ValueError("TCN validation history differs from the configured interval")
            if entry.validation_segment_macro_f1 is not None:
                if entry.validation_segment_macro_f1 > seed_best:
                    seed_best = entry.validation_segment_macro_f1
                    stale = 0
                else:
                    stale += 1
            previous_epoch = entry.epoch
        if previous_epoch != policy.epochs and stale < policy.patience:
            raise ValueError("TCN history ends a seed before its stopping condition")
        if tuple(seeds) != policy.seeds:
            raise ValueError("TCN history seeds differ from the configured search")
        best = max(
            (entry for entry in history.epochs if entry.validation_segment_macro_f1 is not None),
            key=lambda entry: cast(float, entry.validation_segment_macro_f1),
        )
        if (validation.selected_seed, validation.selected_epoch, validation.score) != (
            best.seed,
            best.epoch,
            best.validation_segment_macro_f1,
        ):
            raise ValueError("TCN selection disagrees with the validation history")

    def write_bundle(
        self,
        result: FitResult,
        root: Path,
        *,
        provenance: BundleProvenance,
        analysis: AnalysisConfig,
    ) -> str:
        if not isinstance(result.predictor, TCNPredictor):
            raise TypeError("TCN bundles require a fitted TCN predictor")
        predictor = result.predictor
        if set(result.selected_settings) != {"epoch"}:
            raise ValueError("TCN selected settings must identify exactly one checkpoint epoch")
        analysis_sha256 = hashlib.sha256(json_bytes(analysis.model_dump(mode="json"))).hexdigest()
        environment_sha256 = hashlib.sha256(json_bytes(result.environment)).hexdigest()
        if (
            analysis != predictor.analysis
            or analysis_sha256 != provenance.analysis_profile_sha256
            or environment_sha256 != provenance.environment_sha256
            or result.environment != predictor.configuration.environment
            or result.selected_seed != provenance.training.selected_seed
            or result.validation_score != provenance.training.validation_segment_macro_f1
            or predictor.configuration.training_policy.seeds != provenance.training.seeds
        ):
            raise ValueError("TCN bundle provenance differs from its fitted configuration")
        validation = ValidationRecord(
            score=result.validation_score,
            selected_seed=result.selected_seed,
            selected_epoch=cast(int, result.selected_settings.get("epoch")),
        )
        history = HistoryRecord.model_validate({"epochs": result.history})
        self._validate_selection(predictor.configuration, validation, history)
        with output_directory(root) as output:
            write_json(
                output / _MEMBERS["configuration"], predictor.configuration.model_dump(mode="json")
            )
            np.savez(
                output / _MEMBERS["preprocessing"],
                fill=predictor.preprocessor.fill,
                mean=predictor.preprocessor.mean,
                scale=predictor.preprocessor.scale,
            )
            state = {
                name: tensor.detach().cpu().clone().contiguous()
                for name, tensor in predictor.model.state_dict().items()
            }
            _validate_state(state, predictor.configuration.parameters)
            torch.save(state, output / _MEMBERS["weights"])
            write_json(output / _MEMBERS["validation"], validation.model_dump(mode="json"))
            write_json(output / _MEMBERS["history"], history.model_dump(mode="json"))
            members = tuple(
                BundleMember(
                    path=name,
                    role=role,
                    size_bytes=(output / name).stat().st_size,
                    sha256=digest(output / name),
                )
                for role, name in _MEMBERS.items()
            )
            manifest = ModelBundle(
                bundle_version="1.0",
                model_version_id=provenance.model_version_id,
                architecture_id=self.architecture_id,
                implementation_version=self.implementation_version,
                interface_version="1.0",
                observation_version="2.1",
                feature_version=FEATURE_VERSION,
                preprocessing_version=PREPROCESSING_VERSION,
                decoder_version=DECODER_VERSION,
                analysis_profile=provenance.analysis_profile,
                analysis_profile_sha256=analysis_sha256,
                class_names=("sideways", "growth", "decline"),
                training=provenance.training,
                build_id=provenance.build_id,
                environment_sha256=environment_sha256,
                members=members,
            )
            write_json(output / "manifest.json", manifest.model_dump(mode="json"))
            manifest_sha256 = digest(output / "manifest.json")
        return manifest_sha256

    def load_bundle(
        self, root: Path, *, manifest_sha256: str, runtime: InferenceRuntime
    ) -> TCNPredictor:
        InferenceRuntime.model_validate(runtime.model_dump())
        bundle = verify_model_bundle(
            root,
            manifest_sha256=manifest_sha256,
            architecture_id=self.architecture_id,
            implementation_version=self.implementation_version,
        )
        if (
            {member.role: member.path for member in bundle.members} != _MEMBERS
            or bundle.feature_version != FEATURE_VERSION
            or bundle.preprocessing_version != PREPROCESSING_VERSION
            or bundle.decoder_version != DECODER_VERSION
        ):
            raise ValueError("TCN bundle uses an incompatible codec or scientific contract")
        configuration = _load_json(root / _MEMBERS["configuration"], TCNConfiguration)
        validation = _load_json(root / _MEMBERS["validation"], ValidationRecord)
        history = _load_json(root / _MEMBERS["history"], HistoryRecord)
        if (
            configuration.feature_names != TEMPORAL_NAMES
            or configuration.context_sizes != CONTEXT_SIZES
            or hashlib.sha256(
                json_bytes(configuration.analysis.model_dump(mode="json"))
            ).hexdigest()
            != bundle.analysis_profile_sha256
            or hashlib.sha256(json_bytes(configuration.environment)).hexdigest()
            != bundle.environment_sha256
            or bundle.training.seeds != configuration.training_policy.seeds
            or bundle.training.selected_seed != validation.selected_seed
            or bundle.training.validation_segment_macro_f1 != validation.score
        ):
            raise ValueError("TCN bundle configuration and manifest provenance disagree")
        self._validate_selection(configuration, validation, history)
        if (
            _state_bytes(configuration.parameters) + 3 * len(TEMPORAL_NAMES) * 8
            > runtime.model_cache_budget_bytes
        ):
            raise ResourceBudgetExceeded("TCN model exceeds the inference model cache budget")
        preprocessing = _read_preprocessing(root / _MEMBERS["preprocessing"])
        state = _read_state(root / _MEMBERS["weights"], configuration.parameters)
        torch.set_num_threads(runtime.threads)
        # Loading restores weights; it must not consume the training RNG stream.
        with torch.random.fork_rng(devices=[]):
            model = TCNClassifier(
                len(TEMPORAL_NAMES),
                configuration.parameters.channels,
                configuration.parameters.dropout,
                configuration.parameters.dilations,
            )
        model.load_state_dict(state, strict=True)
        return TCNPredictor(preprocessing, model, configuration)


def create_backend() -> TCNBackend:
    return TCNBackend()


def probe_runtime(device: str) -> None:
    """Exercise CPU Torch without fitting or changing process-global settings."""
    if device != "cpu":
        raise ValueError("TCN supports CPU execution only")
    value = torch.tensor([1.0, 2.0], dtype=torch.float32, device="cpu")
    if not torch.equal(value * value, torch.tensor([1.0, 4.0])):
        raise ValueError("Torch CPU tensor execution failed")
