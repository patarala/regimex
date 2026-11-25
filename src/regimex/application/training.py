"""Train a candidate through a registered backend and verify its saved restoration.

This worker-side operation does not publish a deployment or grant final-reference
access. Public authorization, durable operation commits and process resource
limits belong to the persistent service boundary.
"""

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from pydantic import Field, JsonValue

from regimex.contracts import Identifier, StrictModel
from regimex.data_catalog import (
    LabeledSnapshot,
    NoValidationDirections,
    RawTrainingData,
    analysis_profile_hash,
    content_hash,
)
from regimex.directions import DIRECTION_NAMES, AnalysisConfig, DirectionInterval
from regimex.evaluation import (
    dense_labels,
    probabilities_valid,
    report,
    segment_macro_f1,
    validate_interval_grid,
)
from regimex.features import DoubleArray, IntArray
from regimex.model_bundles import TrainingEvidence, verify_model_bundle
from regimex.model_contracts import (
    BundleProvenance,
    HistoryCallback,
    LoadedPredictor,
    ModelBackend,
    ObservationInputs,
    TrainingInputs,
)
from regimex.model_registry import (
    ArchitectureId,
    ArchitectureRegistry,
    InferenceRuntime,
    RegistryError,
    TrainingPolicySpec,
    TrainingRuntime,
)
from regimex.operations import Sha256
from regimex.prediction_inputs import PredictionSequenceInfo
from regimex.segmentation import DECODER_VERSION, decode
from regimex.storage import output_directory
from regimex.training_statistics import TrainingFitStatistics


class TrainingRequest(StrictModel):
    run_id: Identifier
    model_version_id: Identifier
    architecture_id: ArchitectureId
    parameters: dict[str, JsonValue]
    policy: TrainingPolicySpec
    training_runtime: TrainingRuntime
    verification_runtime: InferenceRuntime
    analysis_profile_id: Identifier
    build_id: Identifier


class CandidateTrainingResult(StrictModel):
    model_version_id: Identifier
    architecture_id: ArchitectureId
    manifest_sha256: Sha256
    dataset_sha256: Sha256
    validation_reference_sha256: Sha256
    validation_segment_macro_f1: Annotated[float, Field(ge=0, le=1)]
    quality_status: Literal["validation_only"] = "validation_only"
    training_runtime: TrainingRuntime
    verification_runtime: InferenceRuntime
    fit_statistics: TrainingFitStatistics


def observation_inputs(snapshots: tuple[LabeledSnapshot, ...]) -> ObservationInputs:
    """Keep original observations, timing and units; backends construct their features."""
    # The namespace and revision align evaluation instruments without exposing seed metadata.
    return ObservationInputs.from_exports(
        (content_hash([item.source_namespace, item.snapshot_id]), item.export) for item in snapshots
    )


def _references(
    snapshots: tuple[LabeledSnapshot, ...],
) -> tuple[tuple[DirectionInterval, ...], ...]:
    return tuple(
        tuple(
            interval
            for interval in item.annotations.intervals
            if interval.instrument_id == instrument.instrument_id
        )
        for item in snapshots
        for instrument in item.export.instruments
    )


@dataclass(frozen=True, slots=True)
class _EvaluationData:
    info: tuple[PredictionSequenceInfo, ...]
    labels: IntArray
    offsets: IntArray
    references: tuple[tuple[DirectionInterval, ...], ...]


@dataclass(frozen=True, slots=True)
class SharedSelectionEvaluator:
    """Use the same availability, decoder and segment metric for every architecture."""

    inputs: ObservationInputs
    references: tuple[tuple[DirectionInterval, ...], ...]
    analysis: AnalysisConfig
    reference_kind: Literal["automatic", "human"]

    def __post_init__(self) -> None:
        if self.reference_kind not in {"automatic", "human"}:
            raise ValueError("unknown reference kind")
        for info, intervals in zip(self.inputs.info, self.references, strict=True):
            validate_interval_grid(intervals, info.instrument_id, info.step_count)

    def _decode(
        self, probabilities: DoubleArray
    ) -> tuple[IntArray, tuple[tuple[DirectionInterval, ...], ...]]:
        if not isinstance(probabilities, np.ndarray) or probabilities.dtype != np.float64:
            raise ValueError("backend probabilities must be a float64 array")
        probabilities_valid(probabilities, self.inputs.row_count)
        offsets = np.concatenate(
            (
                np.zeros(1, dtype=np.int64),
                np.cumsum(
                    [item.step_count for item in self.inputs.info],
                    dtype=np.int64,
                ),
            )
        )
        predicted = tuple(
            decode(probabilities[start:stop], rows, self.analysis)
            for start, stop, rows in zip(
                offsets[:-1],
                offsets[1:],
                self.inputs.observations,
                strict=True,
            )
        )
        return offsets, predicted

    def evaluate(
        self, probabilities: DoubleArray, *, bootstrap_repetitions: int = 0
    ) -> dict[str, JsonValue]:
        offsets, predicted = self._decode(probabilities)
        labels = np.concatenate(
            [
                dense_labels(intervals, item.step_count)
                for intervals, item in zip(self.references, self.inputs.info, strict=True)
            ]
        )
        return report(
            _EvaluationData(self.inputs.info, labels, offsets, self.references),
            predicted,
            reference_kind=self.reference_kind,
            bootstrap_repetitions=bootstrap_repetitions,
        )

    def __call__(self, probabilities: DoubleArray) -> float:
        _, predicted = self._decode(probabilities)
        for info, intervals in zip(self.inputs.info, predicted, strict=True):
            validate_interval_grid(intervals, info.instrument_id, info.step_count)
        return segment_macro_f1(self.references, predicted)


def _validate_predictor(
    predictor: LoadedPredictor,
    request: TrainingRequest,
    implementation_version: str,
    analysis: AnalysisConfig,
) -> None:
    if (
        predictor.architecture_id != request.architecture_id
        or predictor.implementation_version != implementation_version
        or predictor.class_names != DIRECTION_NAMES
        or predictor.analysis != analysis
    ):
        raise ValueError("backend predictor identity, class order or analysis profile differs")


def train_candidate(
    data: RawTrainingData,
    request: TrainingRequest,
    output: Path,
    *,
    registry: ArchitectureRegistry,
    worker_id: str,
    on_history: HistoryCallback | None = None,
) -> CandidateTrainingResult:
    """Create a verified candidate using train/validation, without opening final data.

    The caller establishes the benchmark reservation/selection boundary before
    scheduling scientific searches. Unit behavior fixtures are not final data.
    """
    data = RawTrainingData.model_validate_json(data.model_dump_json())
    request = TrainingRequest.model_validate_json(request.model_dump_json())
    if data.semantics.decoder_version != DECODER_VERSION:
        raise ValueError("training references require an unsupported shared decoder")
    descriptor = registry.get(request.architecture_id)
    parameters = descriptor.validate_parameters(request.parameters)
    descriptor.validate_training_policy(request.policy)
    for item in (*data.train, *data.validation):
        descriptor.validate_observation_fields(item.export.capabilities.observation_fields())
    validation_references = _references(data.validation)
    if not any(
        interval.direction is not None
        for reference in validation_references
        for interval in reference
    ):
        raise NoValidationDirections()
    factory = registry.load_backend_factory_on_worker(
        request.architecture_id,
        request.training_runtime,
        worker_id=worker_id,
    )
    backend = factory()
    if not isinstance(backend, ModelBackend):
        raise RegistryError("installed factory did not return the model backend contract")
    if (backend.architecture_id, backend.implementation_version) != (
        descriptor.architecture_id,
        descriptor.implementation_version,
    ):
        raise RegistryError("backend identity differs from its installed descriptor")
    training = observation_inputs(data.train)
    validation = observation_inputs(data.validation)
    targets = np.concatenate(
        [
            dense_labels(intervals, item.step_count)
            for intervals, item in zip(_references(data.train), training.info, strict=True)
        ]
    )
    evaluator = SharedSelectionEvaluator(
        validation,
        validation_references,
        data.analysis_profile,
        data.validation_references.kind,
    )
    training_inputs = TrainingInputs(training, targets, on_history=on_history)
    fit_started = time.perf_counter()
    fitted = backend.fit(
        training_inputs,
        validation,
        parameters=parameters,
        policy=request.policy,
        analysis=data.analysis_profile,
        runtime=request.training_runtime,
        evaluator=evaluator,
    )
    fit_seconds = time.perf_counter() - fit_started
    _validate_predictor(
        fitted.predictor, request, descriptor.implementation_version, data.analysis_profile
    )
    before = fitted.predictor.predict(validation, runtime=request.verification_runtime)
    verified_score = evaluator(before)
    # A backend may reuse its output buffer during serialization or restored
    # prediction. Retain independent evidence of the fitted model's probabilities.
    before = before.copy()
    if not math.isclose(verified_score, fitted.validation_score, rel_tol=0, abs_tol=1e-12):
        raise ValueError("backend selection score differs from the shared validation evaluator")
    evidence = TrainingEvidence(
        run_id=request.run_id,
        dataset_sha256=data.dataset_sha256,
        split_manifest_sha256=data.split_manifest_sha256,
        labels=data.labels,
        validation_references=data.validation_references,
        seeds=request.policy.seeds,
        selected_seed=fitted.selected_seed,
        selection_metric="decoded_segment_macro_f1/iou-0.50",
        validation_segment_macro_f1=verified_score,
    )
    provenance = BundleProvenance(
        model_version_id=request.model_version_id,
        analysis_profile=request.analysis_profile_id,
        analysis_profile_sha256=analysis_profile_hash(data.analysis_profile),
        training=evidence,
        build_id=request.build_id,
        environment_sha256=content_hash(fitted.environment),
    )
    with output_directory(output) as root:
        manifest_sha256 = backend.write_bundle(
            fitted,
            root,
            provenance=provenance,
            analysis=data.analysis_profile,
        )
        bundle = verify_model_bundle(
            root,
            manifest_sha256=manifest_sha256,
            architecture_id=descriptor.architecture_id,
            implementation_version=descriptor.implementation_version,
        )
        if (
            bundle.training != evidence
            or bundle.model_version_id != request.model_version_id
            or bundle.analysis_profile != provenance.analysis_profile
            or bundle.analysis_profile_sha256 != provenance.analysis_profile_sha256
            or bundle.decoder_version != data.semantics.decoder_version
            or bundle.build_id != provenance.build_id
            or bundle.environment_sha256 != provenance.environment_sha256
        ):
            raise ValueError("written bundle changed the verified training identity or evidence")
        restored = backend.load_bundle(
            root,
            manifest_sha256=manifest_sha256,
            runtime=request.verification_runtime,
        )
        _validate_predictor(
            restored, request, descriptor.implementation_version, data.analysis_profile
        )
        after = restored.predict(validation, runtime=request.verification_runtime)
        evaluator(after)
        if not np.array_equal(before, after):
            raise ValueError("predictions changed after candidate bundle restoration")
    return CandidateTrainingResult(
        model_version_id=request.model_version_id,
        architecture_id=request.architecture_id,
        manifest_sha256=manifest_sha256,
        dataset_sha256=data.dataset_sha256,
        validation_reference_sha256=data.validation_references.content_sha256,
        validation_segment_macro_f1=verified_score,
        training_runtime=request.training_runtime,
        verification_runtime=request.verification_runtime,
        fit_statistics=TrainingFitStatistics(
            fit_seconds=fit_seconds,
            train_rows=training.row_count,
            validation_rows=validation.row_count,
        ),
    )
