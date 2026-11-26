"""Price-shape TCN backend with a bounded, restricted Torch state codec."""

import hashlib
import pickle
import platform
import time
import zipfile
from dataclasses import dataclass
from importlib.metadata import version
from itertools import batched
from pathlib import Path
from typing import Annotated, Literal, cast

import numpy as np
import torch
from pydantic import Field, JsonValue, RootModel
from torch import Tensor, nn

from regimex.backends.epoch_history import HistoryRecord, ValidationRecord, validate_epoch_selection
from regimex.backends.tcn_layout import state_shapes, trainable_parameter_count
from regimex.backends.tcn_update_records import (
    UpdateHistoryRecord,
    UpdateValidationRecord,
    validate_update_selection,
)
from regimex.contracts import StrictModel
from regimex.directions import DIRECTION_NAMES, AnalysisConfig
from regimex.features import DoubleArray, FloatArray, IntArray, log_price_channels
from regimex.model_bundles import (
    BundleMember,
    BundleRole,
    ModelBundle,
    read_bundle_json,
    verify_model_bundle,
)
from regimex.model_contracts import (
    BundleProvenance,
    FitResult,
    ObservationInputs,
    SelectionEvaluator,
    TrainingInputs,
    require_observation_inputs,
)
from regimex.model_registry import (
    FixedUpdatePolicy,
    InferenceRuntime,
    TCNParameters,
    TrainingPolicy,
    TrainingPolicySpec,
    TrainingRuntime,
    validate_tcn_policy,
)
from regimex.prediction_inputs import PreparedPredictionInputs
from regimex.resource_limits import ResourceBudgetExceeded
from regimex.segmentation import DECODER_VERSION
from regimex.storage import digest, json_bytes, output_directory, write_json
from regimex.temporal import (
    CONTEXT_SIZES,
    SELECTION_BATCH_SIZE,
    TEMPORAL_NAMES,
    TCNClassifier,
    TemporalPreprocessor,
    feature_names,
    fit_tcn,
    predict_tcn,
    temporal_price_features,
)
from regimex.temporal_updates import fit_tcn_updates

ARCHITECTURE_ID = "tcn"
IMPLEMENTATION_VERSION = "0.2.0"
FEATURE_VERSION = "temporal-price-shape/1.0"
SIDE_SLOPE_FEATURE_VERSION = "temporal-price-shape/2.0"
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


class _TCNConfigurationBase(StrictModel):
    loss_version: Literal["direction-cross-entropy/2.0"]
    smoothing_scope: Literal["equal_adjacent_targets"]
    boundary_weight_radius: Literal[4]
    boundary_weight_multiplier: Literal[2]
    parameters: TCNParameters
    training_runtime: TrainingRuntime
    analysis: AnalysisConfig
    feature_names: tuple[str, ...] = TEMPORAL_NAMES
    context_sizes: tuple[int, ...] = CONTEXT_SIZES
    kernel_size: Literal[3] = 3
    gradient_clip: Literal[1] = 1
    smoothing_change_clip: Literal[16] = 16
    selection_batch_size: Literal[16] = SELECTION_BATCH_SIZE
    environment: dict[str, JsonValue]


class TCNConfiguration(_TCNConfigurationBase):
    codec_version: Literal["temporal-tcn/2.0"] = CODEC_VERSION
    training_policy: TrainingPolicy


class FixedUpdateTCNConfiguration(_TCNConfigurationBase):
    codec_version: Literal["temporal-tcn/3.0"] = "temporal-tcn/3.0"
    training_policy: FixedUpdatePolicy
    training_sequence_count: int = Field(ge=1)


type NativeTCNConfiguration = Annotated[
    TCNConfiguration | FixedUpdateTCNConfiguration, Field(discriminator="codec_version")
]


def _environment() -> dict[str, JsonValue]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: version(name) for name in ("regimex", "numpy", "torch")},
        "torch_cpu_capability": str(torch.backends.cpu.get_cpu_capability()),
    }


def _feature_inputs(
    observations: ObservationInputs,
    *,
    side_slopes: bool = False,
    selection: slice = slice(None),
) -> tuple[FloatArray, IntArray]:
    require_observation_inputs(observations)
    sequences = [
        temporal_price_features(log_price_channels(rows), side_slopes=side_slopes)
        for rows in observations.observations[selection]
    ]
    offsets = np.r_[0, np.cumsum([len(sequence) for sequence in sequences], dtype=np.int64)]
    return np.concatenate(sequences), offsets


def fit_preprocessing(
    observations: ObservationInputs, *, side_slopes: bool = False
) -> tuple[TemporalPreprocessor, PreparedPredictionInputs]:
    """Fit on training instruments and transform them with the same statistics."""
    values, offsets = _feature_inputs(observations, side_slopes=side_slopes)
    preprocessing = TemporalPreprocessor.fit(values, side_slopes=side_slopes)
    prepared = PreparedPredictionInputs(preprocessing.transform(values), offsets, observations.info)
    return preprocessing, prepared


def prepare_observations(
    observations: ObservationInputs,
    preprocessing: TemporalPreprocessor,
    *,
    side_slopes: bool = False,
    selection: slice = slice(None),
) -> PreparedPredictionInputs:
    values, offsets = _feature_inputs(observations, side_slopes=side_slopes, selection=selection)
    return PreparedPredictionInputs(
        preprocessing.transform(values), offsets, observations.info[selection]
    )


def _feature_version(parameters: TCNParameters) -> str:
    return SIDE_SLOPE_FEATURE_VERSION if parameters.side_slopes else FEATURE_VERSION


def _state_bytes(parameters: TCNParameters) -> int:
    return 4 * trainable_parameter_count(parameters)


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


def _freeze(array: DoubleArray, width: int, *, positive: bool = False) -> DoubleArray:
    if (
        array.dtype != np.float64
        or array.shape != (width,)
        or not np.isfinite(array).all()
        or (positive and (array <= 0).any())
    ):
        raise ValueError("TCN preprocessing has incompatible dtype, shape, finiteness or scale")
    result = array.copy()
    result.flags.writeable = False
    return result


def _validate_state(state: object, parameters: TCNParameters) -> dict[str, Tensor]:
    shapes = state_shapes(parameters)
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
    configuration: NativeTCNConfiguration
    architecture_id: str = ARCHITECTURE_ID
    implementation_version: str = IMPLEMENTATION_VERSION
    class_names: tuple[str, ...] = DIRECTION_NAMES

    def __post_init__(self) -> None:
        RootModel[NativeTCNConfiguration].model_validate_json(self.configuration.model_dump_json())
        if (
            self.configuration.feature_names
            != feature_names(self.configuration.parameters.side_slopes)
            or self.configuration.context_sizes != CONTEXT_SIZES
            or tuple(cast(nn.Conv1d, block.conv).dilation[0] for block in self.model.blocks)
            != self.configuration.parameters.dilations
            or any(
                cast(nn.Dropout, block.dropout).p != self.configuration.parameters.dropout
                for block in self.model.blocks
            )
        ):
            raise ValueError("TCN model and feature configuration disagree")
        width = len(self.configuration.feature_names)
        preprocessing = TemporalPreprocessor(
            _freeze(self.preprocessor.fill, width),
            _freeze(self.preprocessor.mean, width),
            _freeze(self.preprocessor.scale, width, positive=True),
        )
        mask = width // 2
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
        return (
            _state_bytes(self.configuration.parameters)
            + 3 * len(self.configuration.feature_names) * 8
        )

    def predict(self, inputs: ObservationInputs, *, runtime: InferenceRuntime) -> DoubleArray:
        require_observation_inputs(inputs)
        InferenceRuntime.model_validate(runtime.model_dump())
        if self.numeric_bytes > runtime.model_cache_budget_bytes:
            raise ResourceBudgetExceeded("TCN model exceeds the inference model cache budget")
        # These process-global settings are applied only inside computation workers.
        torch.set_num_threads(runtime.threads)
        # Eager operators honor error mode without loading TorchInductor in each fresh worker.
        torch.set_deterministic_debug_mode("error")
        result = np.empty((inputs.row_count, len(DIRECTION_NAMES)), dtype=np.float64)
        start = 0
        for first in range(0, len(inputs.info), runtime.max_batch_size):
            batch = prepare_observations(
                inputs,
                self.preprocessor,
                side_slopes=self.configuration.parameters.side_slopes,
                selection=slice(first, first + runtime.max_batch_size),
            )
            stop = start + len(batch.values)
            result[start:stop] = predict_tcn(self.model, batch, batch_size=runtime.max_batch_size)
            start = stop
        if (
            not np.isfinite(result).all()
            or (result < 0).any()
            or not np.allclose(result.sum(axis=1), 1, rtol=1e-12, atol=1e-12)
        ):
            raise ValueError("TCN returned invalid original-grid probabilities")
        return result


def _read_preprocessing(path: Path, width: int) -> TemporalPreprocessor:
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
                    or member.file_size > width * 8 + 4096
                    or member.flag_bits & 1
                ):
                    raise ValueError("TCN preprocessing archive encoding or size is incompatible")
                with archive.open(member) as stream:
                    if np.lib.format.read_magic(stream) != (1, 0):
                        raise ValueError("TCN preprocessing requires NPY format 1.0")
                    shape, fortran, dtype = np.lib.format.read_array_header_1_0(
                        stream, max_header_size=4096
                    )
                    if shape != (width,) or fortran or dtype != np.dtype(np.float64):
                        raise ValueError("TCN preprocessing dtype or shape is incompatible")
                    data = stream.read(width * 8 + 1)
                    if len(data) != width * 8:
                        raise ValueError("TCN preprocessing array size is incompatible")
                    arrays[member.filename[:-4]] = np.frombuffer(data, dtype=np.float64).copy()
    except (OSError, zipfile.BadZipFile, EOFError) as exc:
        raise ValueError("invalid TCN preprocessing archive") from exc
    return TemporalPreprocessor(arrays["fill"], arrays["mean"], arrays["scale"])


def _read_state(path: Path, parameters: TCNParameters) -> dict[str, Tensor]:
    expected_bytes = _state_bytes(parameters)
    tensor_count = len(state_shapes(parameters))
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
        policy: TrainingPolicySpec,
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
        validate_tcn_policy(policy)
        assert isinstance(policy, (TrainingPolicy, FixedUpdatePolicy))
        if {item.experiment_id for item in training.observations.info} & {
            item.experiment_id for item in validation.info
        }:
            raise ValueError("TCN training and validation experiments must be disjoint")
        rows = training.observations.row_count + validation.row_count
        # Training shuffles sequences, so the longest one can land in a full batch.
        training_activations = _activation_bytes(
            parameters,
            min(policy.batch_size, len(training.observations.info)),
            max(info.step_count for info in training.observations.info),
        )
        validation_activations = max(
            _activation_bytes(parameters, len(batch), max(info.step_count for info in batch))
            for batch in batched(validation.info, SELECTION_BATCH_SIZE, strict=False)
        )
        # Training and validation run in turn. Budget for the larger activation set.
        # Changing the validation batch size could change a close checkpoint score.
        required = (
            rows * len(feature_names(parameters.side_slopes)) * 8 * 8
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

        preprocessing, prepared_training = fit_preprocessing(
            training.observations, side_slopes=parameters.side_slopes
        )
        prepared_validation = prepare_observations(
            validation, preprocessing, side_slopes=parameters.side_slopes
        )
        check_budget()
        if isinstance(policy, FixedUpdatePolicy):
            model, history, selected_seed = fit_tcn_updates(
                prepared_training,
                training.targets,
                prepared_validation,
                parameters=parameters,
                policy=policy,
                runtime=runtime,
                scorer=evaluate,
                check_budget=check_budget,
                on_history=training.on_history,
            )
        else:
            model, history, selected_seed = fit_tcn(
                prepared_training,
                training.targets,
                prepared_validation,
                parameters=parameters,
                policy=policy,
                runtime=runtime,
                scorer=evaluate,
                check_budget=check_budget,
                on_history=training.on_history,
            )
        check_budget()
        selected = max(
            (entry for entry in history if "validation_segment_macro_f1" in entry),
            key=lambda entry: cast(float, entry["validation_segment_macro_f1"]),
        )
        environment = _environment()
        common = _TCNConfigurationBase(
            loss_version="direction-cross-entropy/2.0",
            smoothing_scope="equal_adjacent_targets",
            feature_names=feature_names(parameters.side_slopes),
            boundary_weight_radius=4,
            boundary_weight_multiplier=2,
            parameters=parameters,
            training_runtime=runtime,
            analysis=analysis,
            environment=environment,
        )
        if isinstance(policy, FixedUpdatePolicy):
            configuration: NativeTCNConfiguration = FixedUpdateTCNConfiguration.model_validate(
                {
                    **common.model_dump(),
                    "training_policy": policy,
                    "training_sequence_count": len(training.observations.info),
                }
            )
            selected_settings = {"update": selected["update"]}
        else:
            configuration = TCNConfiguration.model_validate(
                {**common.model_dump(), "training_policy": policy}
            )
            selected_settings = {"epoch": selected["epoch"]}
        return FitResult(
            predictor=TCNPredictor(preprocessing, model, configuration),
            history=tuple(history),
            selected_seed=selected_seed,
            validation_score=cast(float, selected["validation_segment_macro_f1"]),
            selected_settings=selected_settings,
            environment=environment,
        )

    @staticmethod
    def _validate_selection(
        configuration: TCNConfiguration, validation: ValidationRecord, history: HistoryRecord
    ) -> None:
        validate_tcn_policy(configuration.training_policy)
        validate_epoch_selection("TCN", configuration.training_policy, validation, history)

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
        configuration = predictor.configuration
        selected_unit = (
            "update" if isinstance(configuration, FixedUpdateTCNConfiguration) else "epoch"
        )
        if set(result.selected_settings) != {selected_unit}:
            raise ValueError("TCN selected settings must identify its configured checkpoint unit")
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
        validation: ValidationRecord | UpdateValidationRecord
        history: HistoryRecord | UpdateHistoryRecord
        if isinstance(configuration, FixedUpdateTCNConfiguration):
            validation = UpdateValidationRecord(
                score=result.validation_score,
                selected_seed=result.selected_seed,
                selected_update=cast(int, result.selected_settings["update"]),
            )
            history = UpdateHistoryRecord.model_validate({"updates": result.history})
            validate_update_selection(
                configuration.training_policy,
                configuration.training_sequence_count,
                validation,
                history,
            )
        else:
            validation = ValidationRecord(
                score=result.validation_score,
                selected_seed=result.selected_seed,
                selected_epoch=cast(int, result.selected_settings["epoch"]),
            )
            history = HistoryRecord.model_validate({"epochs": result.history})
            self._validate_selection(configuration, validation, history)
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
                feature_version=_feature_version(configuration.parameters),
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
            or bundle.preprocessing_version != PREPROCESSING_VERSION
            or bundle.decoder_version != DECODER_VERSION
        ):
            raise ValueError("TCN bundle uses an incompatible codec or scientific contract")
        configuration = read_bundle_json(
            root / _MEMBERS["configuration"], RootModel[NativeTCNConfiguration], limit=_JSON_LIMIT
        ).root
        validation: ValidationRecord | UpdateValidationRecord
        history: HistoryRecord | UpdateHistoryRecord
        if isinstance(configuration, FixedUpdateTCNConfiguration):
            validation = read_bundle_json(
                root / _MEMBERS["validation"], UpdateValidationRecord, limit=_JSON_LIMIT
            )
            history = read_bundle_json(
                root / _MEMBERS["history"], UpdateHistoryRecord, limit=_JSON_LIMIT
            )
            validate_update_selection(
                configuration.training_policy,
                configuration.training_sequence_count,
                validation,
                history,
            )
        else:
            validation = read_bundle_json(
                root / _MEMBERS["validation"], ValidationRecord, limit=_JSON_LIMIT
            )
            history = read_bundle_json(root / _MEMBERS["history"], HistoryRecord, limit=_JSON_LIMIT)
            self._validate_selection(configuration, validation, history)
        if (
            bundle.feature_version != _feature_version(configuration.parameters)
            or configuration.feature_names != feature_names(configuration.parameters.side_slopes)
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
        if (
            _state_bytes(configuration.parameters) + 3 * len(configuration.feature_names) * 8
            > runtime.model_cache_budget_bytes
        ):
            raise ResourceBudgetExceeded("TCN model exceeds the inference model cache budget")
        preprocessing = _read_preprocessing(
            root / _MEMBERS["preprocessing"], len(configuration.feature_names)
        )
        state = _read_state(root / _MEMBERS["weights"], configuration.parameters)
        torch.set_num_threads(runtime.threads)
        # Loading restores weights; it must not consume the training RNG stream.
        with torch.random.fork_rng(devices=[]):
            model = TCNClassifier(
                len(configuration.feature_names),
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
