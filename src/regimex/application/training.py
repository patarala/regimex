"""Train a candidate through a registered backend and verify its saved restoration.

This worker-side operation does not publish a deployment or grant final-reference
access. Public authorization, durable operation commits and process resource
limits belong to the persistent service boundary.
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, cast

import numpy as np
from pydantic import Field, JsonValue

from regimex.contracts import Identifier, StrictModel
from regimex.data_catalog import (
    LabeledSnapshot,
    RawTrainingData,
    analysis_profile_hash,
    content_hash,
)
from regimex.directions import DIRECTION_NAMES, AnalysisConfig, DirectionInterval
from regimex.evaluation import dense_labels, probabilities_valid, report
from regimex.features import DoubleArray, IntArray
from regimex.model_bundles import TrainingEvidence, verify_model_bundle
from regimex.model_contracts import (
    BundleProvenance,
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
    TrainingPolicy,
    TrainingRuntime,
)
from regimex.operations import Sha256
from regimex.prediction_inputs import PredictionSequenceInfo
from regimex.segmentation import DECODER_VERSION, decode
from regimex.storage import output_directory


class TrainingRequest(StrictModel):
    run_id: Identifier
    model_version_id: Identifier
    architecture_id: ArchitectureId
    parameters: dict[str, JsonValue]
    policy: TrainingPolicy
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


def _available_fields(item: LabeledSnapshot) -> set[str]:
    """Recorded capabilities differ from an occasional missing value in a field."""
    capabilities = item.export.capabilities
    fields: set[str] = set()
    if capabilities.execution_prices != "unavailable":
        fields.add("last_price")
    if capabilities.trade_aggregates != "unavailable":
        fields.update(("trade_count", "trade_volume"))
    if capabilities.accepted_order_aggregates != "unavailable":
        fields.update(
            ("buy_order_count", "buy_order_volume", "sell_order_count", "sell_order_volume")
        )
    if capabilities.historical_quotes.coverage != "unavailable":
        fields.update(("best_bid", "best_ask"))
    return fields


def observation_inputs(snapshots: tuple[LabeledSnapshot, ...]) -> ObservationInputs:
    """Keep original observations, timing and units; backends construct their features."""
    info = []
    observations = []
    grids = []
    instruments = []
    for item in snapshots:
        # Namespace-local IDs are insufficient for grouping/bootstrap identity.
        # This composite ID is for alignment, never a numerical model feature.
        experiment_id = content_hash([item.source_namespace, item.snapshot_id])
        for instrument in item.export.instruments:
            rows = tuple(
                row
                for row in item.export.observations
                if row.instrument_id == instrument.instrument_id
            )
            info.append(
                PredictionSequenceInfo(
                    experiment_id=experiment_id,
                    instrument_id=instrument.instrument_id,
                    step_count=item.export.time_grid.step_count,
                )
            )
            observations.append(rows)
            grids.append(item.export.time_grid)
            instruments.append(instrument)
    return ObservationInputs(
        info=tuple(info),
        observations=tuple(observations),
        time_grids=tuple(grids),
        instruments=tuple(instruments),
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

    def evaluate(
        self, probabilities: DoubleArray, *, bootstrap_repetitions: int = 0
    ) -> dict[str, JsonValue]:
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
        return cast(float, self.evaluate(probabilities)["segment_macro_f1"])


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
        descriptor.validate_observation_fields(_available_fields(item))
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
        _references(data.validation),
        data.analysis_profile,
        data.validation_references.kind,
    )
    fitted = backend.fit(
        TrainingInputs(training, targets),
        validation,
        parameters=parameters,
        policy=request.policy,
        analysis=data.analysis_profile,
        runtime=request.training_runtime,
        evaluator=evaluator,
    )
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
    )
