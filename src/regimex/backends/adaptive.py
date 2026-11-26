"""Train, validate and restore the adaptive price-direction forest."""

import platform
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from importlib.metadata import version
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, Self

import numpy as np
from pydantic import Field, JsonValue, model_validator

from regimex.backends import multiscale_segments
from regimex.backends.adaptive_forest import (
    FEATURE_NAMES,
    DirectionContextModel,
    features,
    fit_training_rows,
    max_nodes_per_tree,
    training_rows,
)
from regimex.contracts import Observation, Seed, StrictModel
from regimex.data_catalog import analysis_profile_hash, content_hash
from regimex.directions import DECODER_VERSION, DIRECTION_NAMES, AnalysisConfig
from regimex.evaluation import probabilities_valid
from regimex.features import DoubleArray
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
    AdaptiveParameters,
    InferenceRuntime,
    SingleFitPolicy,
    TrainingPolicySpec,
    TrainingRuntime,
    TreeCount,
    TreeDepth,
)
from regimex.resource_limits import ResourceBudgetExceeded
from regimex.storage import digest, output_directory, write_json

if TYPE_CHECKING:
    from regimex.backends.adaptive_blend import BlendedAdaptivePredictor

ARCHITECTURE_ID = "adaptive"
IMPLEMENTATION_VERSION = "0.1.0"
FEATURE_VERSION = "adaptive-segments/1.0"
PREPROCESSING_VERSION: Final = "block-log-price/1.0"
_MEMBERS: dict[BundleRole, str] = {
    "configuration": "configuration.json",
    "preprocessing": "preprocessing.json",
    "weights": "weights.npz",
    "validation": "validation.json",
    "history": "history.json",
}


@dataclass(frozen=True)
class _FeatureSet:
    version: str
    names: tuple[str, ...]
    extract: Callable[[tuple[Observation, ...], AnalysisConfig], DoubleArray]


_FEATURE_SETS = {
    "greedy": _FeatureSet(FEATURE_VERSION, FEATURE_NAMES, features),
    "multiscale": _FeatureSet(
        "adaptive-multiscale/1.0", multiscale_segments.FEATURE_NAMES, multiscale_segments.features
    ),
}


class AdaptiveConfiguration(StrictModel):
    codec_version: Literal["adaptive-forest/1.0"] = "adaptive-forest/1.0"
    parameters: AdaptiveParameters
    training_policy: SingleFitPolicy
    training_runtime: TrainingRuntime
    analysis: AnalysisConfig
    environment: dict[str, JsonValue]


class PreprocessingRecord(StrictModel):
    version: Literal["block-log-price/1.0"]
    feature_names: tuple[str, ...]


class FitRecord(StrictModel):
    seed: Seed
    trees: TreeCount
    max_depth: TreeDepth
    validation_segment_macro_f1: float = Field(ge=0, le=1)


class ValidationRecord(StrictModel):
    metric: Literal["decoded_segment_macro_f1/iou-0.50"] = "decoded_segment_macro_f1/iou-0.50"
    selected_seed: Seed
    score: float = Field(ge=0, le=1)


class HistoryRecord(StrictModel):
    fits: tuple[FitRecord, ...] = Field(min_length=1, max_length=128)

    @model_validator(mode="after")
    def one_forest(self) -> Self:
        for left, right in pairwise(self.fits):
            if (
                left.trees >= right.trees
                or left.seed != right.seed
                or left.max_depth != right.max_depth
            ):
                raise ValueError("Adaptive history must describe increasing prefixes of one forest")
        return self


def environment() -> dict[str, JsonValue]:
    return {
        "python": platform.python_version(),
        "packages": {name: version(name) for name in ("regimex", "numpy", "scikit-learn")},
    }


@dataclass(frozen=True, slots=True)
class AdaptivePredictor:
    model: DirectionContextModel
    configuration: AdaptiveConfiguration
    architecture_id: str = ARCHITECTURE_ID
    implementation_version: str = IMPLEMENTATION_VERSION
    class_names: tuple[str, ...] = DIRECTION_NAMES

    @property
    def analysis(self) -> AnalysisConfig:
        return self.configuration.analysis

    @property
    def numeric_bytes(self) -> int:
        return self.model.numeric_bytes

    def predict_sequence(self, rows: tuple[Observation, ...]) -> DoubleArray:
        features = _FEATURE_SETS[self.configuration.parameters.feature_set].extract
        return self.model.predict(features(rows, self.analysis))

    def predict(self, inputs: ObservationInputs, *, runtime: InferenceRuntime) -> DoubleArray:
        require_observation_inputs(inputs)
        InferenceRuntime.model_validate(runtime.model_dump())
        if self.numeric_bytes > runtime.model_cache_budget_bytes:
            raise ResourceBudgetExceeded("adaptive forest exceeds the inference model cache budget")
        probabilities = np.empty((inputs.row_count, len(DIRECTION_NAMES)), dtype=np.float64)
        offset = 0
        # One sequence at a time keeps both context and temporary arrays within an instrument.
        for rows in inputs.observations:
            end = offset + len(rows)
            probabilities[offset:end] = self.predict_sequence(rows)
            offset = end
        probabilities_valid(probabilities, inputs.row_count)
        return probabilities

    def validation_predictions(
        self,
        inputs: ObservationInputs,
        *,
        memory_budget_bytes: int,
        check_time: Callable[[], None] | None = None,
    ) -> Iterator[tuple[int, DoubleArray]]:
        """Yield probabilities on the original grid for tree prefixes and the complete forest."""
        require_observation_inputs(inputs)
        feature_set = _FEATURE_SETS[self.configuration.parameters.feature_set]
        maximum_sequence = max(info.step_count for info in inputs.info)
        feature_bytes = maximum_sequence * len(feature_set.names) * 8 * 4
        matrix_bytes = inputs.row_count * len(DIRECTION_NAMES) * 8
        slots = (memory_budget_bytes - self.numeric_bytes - feature_bytes) // matrix_bytes
        if slots < 1:
            raise ResourceBudgetExceeded("adaptive validation predictions exceed the memory budget")
        trees = len(self.model.offsets)
        counts = sorted(
            {
                count
                for count in (1, 5, 10, trees // 4, trees // 2, 3 * trees // 4, trees)
                if 1 <= count <= trees
            }
        )
        # Keep the full model under a tight budget. Other checkpoints share the remaining space.
        if slots == 1:
            counts = [trees]
        elif slots < len(counts):
            counts = [counts[i * (len(counts) - 1) // (slots - 1)] for i in range(slots)]
        checkpoints = tuple(counts)
        predictions = [np.empty((inputs.row_count, len(DIRECTION_NAMES))) for _ in counts]
        first = 0
        for rows in inputs.observations:
            if check_time is not None:
                check_time()
            sequence_features = feature_set.extract(rows, self.analysis)
            stop = first + len(rows)
            for matrix, prefix in zip(
                predictions,
                self.model.predict_prefixes(sequence_features, checkpoints),
                strict=True,
            ):
                matrix[first:stop] = prefix
            first = stop
        for count, matrix in zip(counts, predictions, strict=True):
            if check_time is not None:
                check_time()
            probabilities_valid(matrix, inputs.row_count)
            yield count, matrix


def _record(configuration: AdaptiveConfiguration, trees: int, score: float) -> FitRecord:
    return FitRecord(
        seed=configuration.training_policy.seeds[0],
        trees=trees,
        max_depth=configuration.parameters.max_depth,
        validation_segment_macro_f1=score,
    )


def fit_result_from_forest(
    model: DirectionContextModel,
    *,
    parameters: AdaptiveParameters,
    policy: SingleFitPolicy,
    analysis: AnalysisConfig,
    runtime: TrainingRuntime,
    validation_score: float,
    environment: dict[str, JsonValue],
) -> FitResult:
    """Bind an independently evaluated numeric forest to its original fit and selection."""
    config = AdaptiveConfiguration(
        parameters=parameters,
        training_policy=policy,
        analysis=analysis,
        training_runtime=runtime,
        environment=environment,
    )
    return _fit_result(
        AdaptivePredictor(model, config), [_record(config, parameters.trees, validation_score)]
    )


def _fit_result(predictor: AdaptivePredictor, history: list[FitRecord]) -> FitResult:
    config = predictor.configuration
    predictor.model.validate(trees=config.parameters.trees, max_depth=config.parameters.max_depth)
    full = history[-1]
    return FitResult(
        predictor=predictor,
        history=tuple(row.model_dump(mode="json") for row in history),
        selected_seed=full.seed,
        validation_score=full.validation_segment_macro_f1,
        selected_settings=config.parameters.model_dump(mode="json"),
        environment=config.environment.copy(),
    )


class AdaptiveBackend:
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
        if type(training) is not TrainingInputs or type(parameters) is not AdaptiveParameters:
            raise TypeError("adaptive requires TrainingInputs and AdaptiveParameters")
        require_observation_inputs(validation)
        if not isinstance(policy, SingleFitPolicy):
            raise ValueError("adaptive requires a single forest fit")
        parameters = AdaptiveParameters.model_validate(parameters.model_dump())
        policy = SingleFitPolicy.model_validate(policy.model_dump())
        runtime = TrainingRuntime.model_validate(runtime.model_dump())
        analysis = AnalysisConfig.model_validate(analysis.model_dump())
        if {item.experiment_id for item in training.observations.info} & {
            item.experiment_id for item in validation.info
        }:
            raise ValueError("adaptive training and validation experiments must be disjoint")

        train_rows = training.observations.row_count
        feature_set = _FEATURE_SETS[parameters.feature_set]
        width = len(feature_set.names)
        max_nodes = min(max_nodes_per_tree(parameters.max_depth), 2 * train_rows - 1)
        # In the worst case every step has a different feature/target pair. Count the
        # sequence tables, concatenation, sort and unique workspaces before allocating.
        maximum_sequence = max(
            max(item.step_count for item in training.observations.info),
            max(item.step_count for item in validation.info),
        )
        required = (
            train_rows * (width + 1) * 8 * 6
            + max_nodes * parameters.trees * 192
            + maximum_sequence * width * 8 * 4
            + validation.row_count * len(DIRECTION_NAMES) * 8
        )
        if required > runtime.memory_budget_bytes:
            raise ResourceBudgetExceeded(
                "adaptive feature and tree allocations exceed the memory budget"
            )
        deadline = time.monotonic() + runtime.time_budget_seconds

        def check_time() -> None:
            if time.monotonic() >= deadline:
                raise ResourceBudgetExceeded("adaptive exceeded the training time budget")

        examples = []
        first = 0
        for rows in training.observations.observations:
            check_time()
            examples.append(
                training_rows(
                    feature_set.extract(rows, analysis), training.targets[first : first + len(rows)]
                )
            )
            first += len(rows)
        check_time()
        model = fit_training_rows(
            examples,
            interval_weighting=parameters.sample_weighting == "interval",
            trees=parameters.trees,
            max_depth=parameters.max_depth,
            seed=policy.seeds[0],
        )
        del examples
        check_time()
        config = AdaptiveConfiguration(
            parameters=parameters,
            training_policy=policy,
            training_runtime=runtime,
            analysis=analysis,
            environment=environment(),
        )
        predictor = AdaptivePredictor(model, config)
        history = []
        for count, predictions in predictor.validation_predictions(
            validation, memory_budget_bytes=runtime.memory_budget_bytes, check_time=check_time
        ):
            row = _record(config, count, evaluator(predictions))
            check_time()
            history.append(row)
            if training.on_history is not None:
                training.on_history(row.model_dump(mode="json"))
        return _fit_result(predictor, history)

    def write_bundle(
        self,
        result: FitResult,
        root: Path,
        *,
        provenance: BundleProvenance,
        analysis: AnalysisConfig,
    ) -> str:
        if not isinstance(result.predictor, AdaptivePredictor):
            raise TypeError("adaptive bundles require a fitted adaptive predictor")
        predictor = result.predictor
        config = predictor.configuration
        validation = ValidationRecord(
            score=result.validation_score, selected_seed=result.selected_seed
        )
        if (
            analysis != predictor.analysis
            or analysis_profile_hash(analysis) != provenance.analysis_profile_sha256
            or result.environment != config.environment
            or content_hash(result.environment) != provenance.environment_sha256
            or provenance.training.seeds != config.training_policy.seeds
            or provenance.training.selected_seed != validation.selected_seed
            or provenance.training.validation_segment_macro_f1 != validation.score
        ):
            raise ValueError("adaptive bundle provenance differs from the fitted configuration")
        expected = _record(config, config.parameters.trees, validation.score)
        history = HistoryRecord.model_validate({"fits": result.history})
        if (
            history.fits[-1] != expected
            or expected.seed != validation.selected_seed
            or result.selected_settings != config.parameters.model_dump(mode="json")
        ):
            raise ValueError("adaptive validation, history and selected settings disagree")
        predictor.model.validate(
            trees=config.parameters.trees, max_depth=config.parameters.max_depth
        )
        feature_set = _FEATURE_SETS[config.parameters.feature_set]
        preprocessing = PreprocessingRecord(
            version=PREPROCESSING_VERSION, feature_names=feature_set.names
        )
        with output_directory(root) as stage:
            for role, record in (
                ("configuration", config),
                ("preprocessing", preprocessing),
                ("history", history),
                ("validation", validation),
            ):
                write_json(stage / f"{role}.json", record.model_dump(mode="json"))
            predictor.model.save(stage / _MEMBERS["weights"])
            members = tuple(
                BundleMember(
                    role=role_name,
                    path=name,
                    size_bytes=(stage / name).stat().st_size,
                    sha256=digest(stage / name),
                )
                for role_name, name in _MEMBERS.items()
            )
            manifest = ModelBundle(
                bundle_version="1.0",
                model_version_id=provenance.model_version_id,
                architecture_id=ARCHITECTURE_ID,
                implementation_version=IMPLEMENTATION_VERSION,
                interface_version="1.0",
                observation_version="2.1",
                feature_version=feature_set.version,
                preprocessing_version=PREPROCESSING_VERSION,
                decoder_version=DECODER_VERSION,
                analysis_profile=provenance.analysis_profile,
                analysis_profile_sha256=provenance.analysis_profile_sha256,
                class_names=("sideways", "growth", "decline"),
                training=provenance.training,
                build_id=provenance.build_id,
                environment_sha256=provenance.environment_sha256,
                members=members,
            )
            write_json(stage / "manifest.json", manifest.model_dump(mode="json"))
            return digest(stage / "manifest.json")

    def load_bundle(
        self, root: Path, *, manifest_sha256: str, runtime: InferenceRuntime
    ) -> AdaptivePredictor | BlendedAdaptivePredictor:
        from regimex.backends.adaptive_blend import BLEND_VERSION, load_blend

        InferenceRuntime.model_validate(runtime.model_dump())
        bundle = verify_model_bundle(
            root,
            manifest_sha256=manifest_sha256,
            architecture_id=ARCHITECTURE_ID,
            implementation_version=IMPLEMENTATION_VERSION,
        )
        if bundle.feature_version == BLEND_VERSION:
            return load_blend(root, bundle, runtime)
        return self._restore_forest(root, bundle, runtime)

    def _restore_forest(
        self, root: Path, bundle: ModelBundle, runtime: InferenceRuntime
    ) -> AdaptivePredictor:
        """Restore a single forest after its envelope and file digests have been verified."""
        if (
            {member.role: member.path for member in bundle.members} != _MEMBERS
            or bundle.training.parent_models
            or bundle.preprocessing_version != PREPROCESSING_VERSION
            or bundle.decoder_version != DECODER_VERSION
        ):
            raise ValueError("adaptive bundle has an incompatible codec or scientific contract")
        config = read_bundle_json(root / _MEMBERS["configuration"], AdaptiveConfiguration)
        preprocessing = read_bundle_json(root / _MEMBERS["preprocessing"], PreprocessingRecord)
        history = read_bundle_json(root / _MEMBERS["history"], HistoryRecord)
        validation = read_bundle_json(root / _MEMBERS["validation"], ValidationRecord)
        feature_set = _FEATURE_SETS[config.parameters.feature_set]
        if (
            preprocessing.feature_names != feature_set.names
            or bundle.feature_version != feature_set.version
            or bundle.analysis_profile_sha256 != analysis_profile_hash(config.analysis)
            or bundle.environment_sha256 != content_hash(config.environment)
            or bundle.training.seeds != config.training_policy.seeds
            or bundle.training.selected_seed != validation.selected_seed
            or bundle.training.validation_segment_macro_f1 != validation.score
            or validation.selected_seed != config.training_policy.seeds[0]
            or history.fits[-1] != _record(config, config.parameters.trees, validation.score)
        ):
            raise ValueError("adaptive preprocessing, scientific provenance or selection disagree")
        model = DirectionContextModel.load(
            root / _MEMBERS["weights"],
            trees=config.parameters.trees,
            max_depth=config.parameters.max_depth,
            max_bytes=runtime.model_cache_budget_bytes,
        )
        return AdaptivePredictor(model, config)


def create_backend() -> AdaptiveBackend:
    return AdaptiveBackend()


def probe_runtime(device: str) -> None:
    """Import the compiled tree estimator and traverse a known tree without fitting."""
    from sklearn.ensemble import ExtraTreesClassifier  # type: ignore[import-untyped]

    if device != "cpu":
        raise ValueError("adaptive supports CPU execution only")
    ExtraTreesClassifier(n_estimators=1, n_jobs=1)
    model = DirectionContextModel(
        np.array([0]),
        np.array([[1, 2], [-1, -1], [-1, -1]]),
        np.array([0, -2, -2]),
        np.zeros(3),
        np.array([[0.5, 0.5, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    )
    values = np.ones((2, len(FEATURE_NAMES)))
    values[0, 0] = -1
    if not np.array_equal(model.predict(values), model.votes[1:]):
        raise ValueError("adaptive CPU tree traversal failed")
