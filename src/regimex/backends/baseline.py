"""Real aggregate logistic backend with a numeric, nonexecutable bundle codec."""

import hashlib
import platform
import time
import zipfile
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Literal, cast

import numpy as np
from pydantic import Field, JsonValue

from regimex.baseline import LinearModel, fit_baseline
from regimex.contracts import StrictModel
from regimex.directions import DIRECTION_NAMES, AnalysisConfig
from regimex.features import (
    FEATURE_NAMES,
    FEATURE_VERSION,
    NUMERIC_NAMES,
    DoubleArray,
    Preprocessor,
    compute_features,
)
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
    LogisticParameters,
    TrainingPolicy,
    TrainingRuntime,
)
from regimex.prediction_inputs import PredictionInputs, PreparedPredictionInputs
from regimex.resource_limits import ResourceBudgetExceeded
from regimex.segmentation import DECODER_VERSION
from regimex.storage import digest, json_bytes, output_directory, write_json

ARCHITECTURE_ID = "baseline"
IMPLEMENTATION_VERSION = "0.1.0"
PREPROCESSING_VERSION = "baseline-standardization/1.0"
CODEC_VERSION: Literal["aggregate-logistic/1.0"] = "aggregate-logistic/1.0"
_MEMBERS: dict[BundleRole, str] = {
    "configuration": "configuration.json",
    "preprocessing": "preprocessing.npz",
    "weights": "weights.npz",
    "validation": "validation.json",
    "history": "history.json",
}


class BaselineConfiguration(StrictModel):
    codec_version: Literal["aggregate-logistic/1.0"] = CODEC_VERSION
    parameters: LogisticParameters
    training_policy: TrainingPolicy
    training_runtime: TrainingRuntime
    analysis: AnalysisConfig
    feature_names: tuple[str, ...] = FEATURE_NAMES
    solver: Literal["lbfgs"] = "lbfgs"
    maximum_iterations: Literal[2000] = 2000
    environment: dict[str, JsonValue]


class ValidationRecord(StrictModel):
    metric: Literal["decoded_segment_macro_f1/iou-0.50"] = "decoded_segment_macro_f1/iou-0.50"
    score: float = Field(ge=0, le=1)
    selected_seed: int = Field(ge=0, le=2**32 - 1)
    regularization: float = Field(gt=0)


class CandidateRecord(StrictModel):
    regularization: float = Field(gt=0)
    validation_segment_macro_f1: float = Field(ge=0, le=1)


class HistoryRecord(StrictModel):
    candidates: tuple[CandidateRecord, ...] = Field(min_length=1)


def _validate_policy(policy: TrainingPolicy) -> None:
    TrainingPolicy.model_validate(policy.model_dump())
    if policy.optimizer != "lbfgs":
        raise ValueError("baseline supports only the lbfgs optimizer")
    if len(policy.seeds) != 1:
        raise ValueError("deterministic baseline requires exactly one recorded seed")
    defaults = TrainingPolicy(optimizer="lbfgs")
    unused = (
        "epochs",
        "batch_size",
        "patience",
        "learning_rate",
        "weight_decay",
        "validation_interval",
    )
    changed = [name for name in unused if getattr(policy, name) != getattr(defaults, name)]
    if changed:
        raise ValueError("baseline does not use training policy fields: " + ", ".join(changed))


def _environment() -> dict[str, JsonValue]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: version(name) for name in ("regimex", "numpy", "scikit-learn", "threadpoolctl")
        },
    }


def _feature_inputs(observations: ObservationInputs) -> PredictionInputs:
    require_observation_inputs(observations)
    return PredictionInputs(
        np.concatenate([compute_features(rows) for rows in observations.observations]),
        np.r_[0, np.cumsum([item.step_count for item in observations.info])].astype(np.int64),
        observations.info,
    )


def _freeze(array: DoubleArray, shape: tuple[int, ...], *, positive: bool = False) -> DoubleArray:
    if (
        array.dtype != np.float64
        or array.shape != shape
        or not np.isfinite(array).all()
        or (positive and (array <= 0).any())
    ):
        raise ValueError("baseline arrays have incompatible dtype, shape, finiteness, or scale")
    result = array.copy()
    result.flags.writeable = False
    return result


@dataclass(frozen=True, slots=True)
class BaselinePredictor:
    preprocessor: Preprocessor
    model: LinearModel
    configuration: BaselineConfiguration
    architecture_id: str = ARCHITECTURE_ID
    implementation_version: str = IMPLEMENTATION_VERSION
    class_names: tuple[str, ...] = DIRECTION_NAMES

    def __post_init__(self) -> None:
        size = len(FEATURE_NAMES)
        width = size * 4 * (len(self.configuration.parameters.context_sizes) + 1)
        if self.model.context_sizes != self.configuration.parameters.context_sizes:
            raise ValueError("baseline context configuration disagrees with weights")
        preprocessor = Preprocessor(
            _freeze(self.preprocessor.fill, (size,)),
            _freeze(self.preprocessor.mean, (size,)),
            _freeze(self.preprocessor.scale, (size,), positive=True),
        )
        mask = len(NUMERIC_NAMES)
        if (
            preprocessor.fill[mask:].any()
            or preprocessor.mean[mask:].any()
            or not np.array_equal(preprocessor.scale[mask:], np.ones(mask))
        ):
            raise ValueError("preprocessing must preserve explicit missing-value masks")
        model = LinearModel(
            _freeze(self.model.coefficients, (len(DIRECTION_NAMES), width)),
            _freeze(self.model.intercept, (len(DIRECTION_NAMES),)),
            _freeze(self.model.mean, (width,)),
            _freeze(self.model.scale, (width,), positive=True),
            self.model.context_sizes,
        )
        object.__setattr__(self, "preprocessor", preprocessor)
        object.__setattr__(self, "model", model)

    @property
    def analysis(self) -> AnalysisConfig:
        return self.configuration.analysis

    @property
    def numeric_bytes(self) -> int:
        return sum(
            array.nbytes
            for array in (
                self.preprocessor.fill,
                self.preprocessor.mean,
                self.preprocessor.scale,
                self.model.coefficients,
                self.model.intercept,
                self.model.mean,
                self.model.scale,
            )
        )

    def predict(self, inputs: ObservationInputs, *, runtime: InferenceRuntime) -> DoubleArray:
        from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]

        require_observation_inputs(inputs)
        InferenceRuntime.model_validate(runtime.model_dump())
        if self.numeric_bytes > runtime.model_cache_budget_bytes:
            raise ResourceBudgetExceeded("baseline model exceeds the inference model cache budget")
        features = _feature_inputs(inputs)
        output = np.empty((len(features.values), len(DIRECTION_NAMES)), dtype=np.float64)
        # Process-global BLAS limits belong in an isolated worker computation.
        with threadpool_limits(limits=runtime.threads):
            for first in range(0, len(inputs.info), runtime.max_batch_size):
                last = min(first + runtime.max_batch_size, len(inputs.info))
                start, stop = int(features.offsets[first]), int(features.offsets[last])
                prepared = PreparedPredictionInputs(
                    self.preprocessor.transform(features.values[start:stop]),
                    features.offsets[first : last + 1] - start,
                    inputs.info[first:last],
                )
                output[start:stop] = self.model.predict(prepared)
        if (
            not np.isfinite(output).all()
            or (output < 0).any()
            or not np.allclose(output.sum(axis=1), 1, atol=1e-12, rtol=1e-12)
        ):
            raise ValueError("baseline returned invalid original-grid probabilities")
        return output


def _read_arrays(path: Path, shapes: dict[str, tuple[int, ...]]) -> dict[str, DoubleArray]:
    """Bound and validate NPY headers before allocating any weight array."""
    arrays: dict[str, DoubleArray] = {}
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            if len(members) != len(shapes) or {item.filename for item in members} != {
                name + ".npy" for name in shapes
            }:
                raise ValueError("baseline numeric archive members differ from its codec")
            for member in members:
                name = member.filename[:-4]
                expected_bytes = int(np.prod(shapes[name])) * 8
                if (
                    member.compress_type != zipfile.ZIP_STORED
                    or member.file_size > expected_bytes + 4096
                    or member.flag_bits & 1
                ):
                    raise ValueError("baseline numeric archive encoding or size is incompatible")
                with archive.open(member) as stream:
                    if np.lib.format.read_magic(stream) != (1, 0):
                        raise ValueError("baseline numeric archive requires NPY format 1.0")
                    shape, fortran, dtype = np.lib.format.read_array_header_1_0(
                        stream, max_header_size=4096
                    )
                    if shape != shapes[name] or fortran or dtype != np.dtype(np.float64):
                        raise ValueError("baseline numeric array dtype or shape is incompatible")
                    data = stream.read(expected_bytes + 1)
                    if len(data) != expected_bytes:
                        raise ValueError("baseline numeric array size is incompatible")
                    arrays[name] = np.frombuffer(data, dtype=np.float64).reshape(shape).copy()
        return arrays
    except (OSError, zipfile.BadZipFile, EOFError) as exc:
        raise ValueError("invalid baseline numeric archive") from exc


def _load_json[T: StrictModel](path: Path, model: type[T]) -> T:
    with path.open("rb") as stream:
        data = stream.read(1024 * 1024 + 1)
    if len(data) > 1024 * 1024:
        raise ValueError("baseline JSON member exceeds its codec size limit")
    return model.model_validate_json(data)


class BaselineBackend:
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
        if type(training) is not TrainingInputs or type(parameters) is not LogisticParameters:
            raise TypeError("baseline requires TrainingInputs and LogisticParameters")
        LogisticParameters.model_validate(parameters.model_dump())
        TrainingRuntime.model_validate(runtime.model_dump())
        AnalysisConfig.model_validate(analysis.model_dump())
        _validate_policy(policy)
        if {item.experiment_id for item in training.observations.info} & {
            item.experiment_id for item in validation.info
        }:
            raise ValueError("baseline training and validation experiments must be disjoint")
        width = len(FEATURE_NAMES) * 4 * (len(parameters.context_sizes) + 1)
        # A conservative allocation bound rejects oversized jobs before feature
        # preparation. A worker process enforces the hard OS resource limits.
        rows = training.observations.row_count + validation.row_count
        required = rows * (width * 8 * 3 + len(FEATURE_NAMES) * 8 * 6) + width * 8 * 32
        if required > runtime.memory_budget_bytes:
            raise ResourceBudgetExceeded(
                "baseline feature allocations exceed the training memory budget"
            )
        started = time.monotonic()

        def check_budget() -> None:
            if time.monotonic() - started >= runtime.time_budget_seconds:
                raise ResourceBudgetExceeded("baseline exceeded the training time budget")

        def evaluate(probabilities: DoubleArray) -> float:
            score = evaluator(probabilities)
            if not np.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("baseline validation score must be segment macro F1 in [0, 1]")
            return score

        raw_training = _feature_inputs(training.observations)
        raw_validation = _feature_inputs(validation)
        preprocessor = Preprocessor.fit(raw_training.values)
        train = PreparedPredictionInputs(
            preprocessor.transform(raw_training.values),
            raw_training.offsets,
            training.observations.info,
        )
        valid = PreparedPredictionInputs(
            preprocessor.transform(raw_validation.values), raw_validation.offsets, validation.info
        )
        model, history, selected = fit_baseline(
            train,
            training.targets,
            valid,
            regularization_candidates=parameters.regularization,
            seed=policy.seeds[0],
            threads=runtime.threads,
            context_sizes=parameters.context_sizes,
            scorer=evaluate,
            check_budget=check_budget,
        )
        check_budget()
        scores = [
            cast(float, row["validation_segment_macro_f1"])
            for row in history
            if isinstance(row, dict)
        ]
        best_score = max(scores)
        if not 0 <= best_score <= 1:
            raise ValueError("baseline evaluator must return segment macro F1 between zero and one")
        environment = _environment()
        configuration = BaselineConfiguration(
            parameters=parameters,
            training_policy=policy,
            training_runtime=runtime,
            analysis=analysis,
            environment=environment,
        )
        predictor = BaselinePredictor(preprocessor, model, configuration)
        return FitResult(
            predictor=predictor,
            history=tuple(cast(dict[str, JsonValue], row) for row in history),
            selected_seed=policy.seeds[0],
            validation_score=best_score,
            selected_settings={"regularization": selected},
            environment=environment,
        )

    def write_bundle(
        self,
        result: FitResult,
        root: Path,
        *,
        provenance: BundleProvenance,
        analysis: AnalysisConfig,
    ) -> str:
        if not isinstance(result.predictor, BaselinePredictor):
            raise TypeError("baseline bundles require a fitted baseline predictor")
        predictor = result.predictor
        if analysis != predictor.analysis:
            raise ValueError("saved analysis profile differs from the fitted selection profile")
        analysis_sha256 = hashlib.sha256(json_bytes(analysis.model_dump(mode="json"))).hexdigest()
        environment_sha256 = hashlib.sha256(json_bytes(result.environment)).hexdigest()
        if (
            analysis_sha256 != provenance.analysis_profile_sha256
            or environment_sha256 != provenance.environment_sha256
            or result.environment != predictor.configuration.environment
            or provenance.training.seeds != predictor.configuration.training_policy.seeds
            or provenance.training.selected_seed != result.selected_seed
            or provenance.training.validation_segment_macro_f1 != result.validation_score
        ):
            raise ValueError(
                "bundle provenance differs from fitted analysis, environment, or selection"
            )
        validation = ValidationRecord(
            score=result.validation_score,
            selected_seed=result.selected_seed,
            regularization=cast(float, result.selected_settings["regularization"]),
        )
        history = HistoryRecord.model_validate({"candidates": result.history})
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
            np.savez(
                output / _MEMBERS["weights"],
                coefficients=predictor.model.coefficients,
                intercept=predictor.model.intercept,
                mean=predictor.model.mean,
                scale=predictor.model.scale,
                context_sizes=np.asarray(predictor.model.context_sizes, dtype=np.float64),
            )
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

    @staticmethod
    def _validate_selection(
        configuration: BaselineConfiguration, validation: ValidationRecord, history: HistoryRecord
    ) -> None:
        _validate_policy(configuration.training_policy)
        if (
            tuple(item.regularization for item in history.candidates)
            != configuration.parameters.regularization
            or validation.selected_seed != configuration.training_policy.seeds[0]
        ):
            raise ValueError("baseline selection history differs from the configured search")
        best = max(history.candidates, key=lambda item: item.validation_segment_macro_f1)
        if (best.regularization, best.validation_segment_macro_f1) != (
            validation.regularization,
            validation.score,
        ):
            raise ValueError("baseline selection disagrees with validation history")

    def load_bundle(
        self, root: Path, *, manifest_sha256: str, runtime: InferenceRuntime
    ) -> BaselinePredictor:
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
            raise ValueError("baseline bundle uses an incompatible codec or scientific contract")
        configuration = _load_json(root / _MEMBERS["configuration"], BaselineConfiguration)
        validation = _load_json(root / _MEMBERS["validation"], ValidationRecord)
        history = _load_json(root / _MEMBERS["history"], HistoryRecord)
        if (
            configuration.feature_names != FEATURE_NAMES
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
            raise ValueError("baseline bundle configuration and manifest provenance disagree")
        self._validate_selection(configuration, validation, history)
        size = len(FEATURE_NAMES)
        width = size * 4 * (len(configuration.parameters.context_sizes) + 1)
        state_bytes = 8 * (3 * size + 5 * width + 3)
        if state_bytes > runtime.model_cache_budget_bytes:
            raise ResourceBudgetExceeded("baseline model exceeds the inference model cache budget")
        prepared = _read_arrays(
            root / _MEMBERS["preprocessing"], {name: (size,) for name in ("fill", "mean", "scale")}
        )
        weights = _read_arrays(
            root / _MEMBERS["weights"],
            {
                "coefficients": (len(DIRECTION_NAMES), width),
                "intercept": (len(DIRECTION_NAMES),),
                "mean": (width,),
                "scale": (width,),
                "context_sizes": (len(configuration.parameters.context_sizes),),
            },
        )
        if not np.array_equal(weights["context_sizes"], configuration.parameters.context_sizes):
            raise ValueError("baseline numeric state and configured contexts disagree")
        return BaselinePredictor(
            Preprocessor(prepared["fill"], prepared["mean"], prepared["scale"]),
            LinearModel(
                weights["coefficients"],
                weights["intercept"],
                weights["mean"],
                weights["scale"],
                configuration.parameters.context_sizes,
            ),
            configuration,
        )


def create_backend() -> BaselineBackend:
    return BaselineBackend()


def probe_runtime(device: str) -> None:
    """Exercise actual numeric dependencies without fitting or global changes."""
    from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
    from threadpoolctl import threadpool_info

    if device != "cpu":
        raise ValueError("baseline supports CPU execution only")
    estimator = LogisticRegression(solver="lbfgs")
    estimator.classes_ = np.arange(len(DIRECTION_NAMES))
    estimator.coef_ = np.zeros((len(DIRECTION_NAMES), 2))
    estimator.intercept_ = np.zeros(len(DIRECTION_NAMES))
    estimator.n_features_in_ = 2
    probabilities = estimator.predict_proba(np.zeros((1, 2)))
    if not np.allclose(probabilities, np.full((1, len(DIRECTION_NAMES)), 1 / 3)):
        raise ValueError("baseline logistic numeric runtime is incompatible")
    threadpool_info()
    if not np.array_equal(np.eye(2) @ np.ones(2), np.ones(2)):
        raise ValueError("baseline CPU matrix operation failed")
