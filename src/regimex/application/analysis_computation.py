"""Observation-only prediction and rendering inside a bounded computation child."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydantic import JsonValue

from regimex.application.service import AnalysisResult, PredictionSequence
from regimex.contracts import Observation
from regimex.directions import DIRECTION_NAMES, DirectionAnnotations
from regimex.evaluation import probabilities_valid
from regimex.exports import ExportInstrument, ExportV2, parse_export_v2
from regimex.model_bundles import ModelBundle, verify_model_bundle
from regimex.model_contracts import LoadedPredictor, ModelBackend, ObservationInputs
from regimex.model_registry import ArchitectureRegistry, RegistryError
from regimex.prediction_inputs import PredictionSequenceInfo
from regimex.segmentation import DECODER_VERSION, decode
from regimex.service_contracts import AnalysisRecord, ModelRecord, SnapshotRecord
from regimex.storage import json_bytes


def restore_predictor(
    root: Path, model: ModelRecord, registry: ArchitectureRegistry, *, worker_id: str
) -> tuple[LoadedPredictor, ModelBundle]:
    """Validate immutable bytes and let the installed backend restore its codec."""
    descriptor = registry.get(model.architecture_id)
    manifest = verify_model_bundle(
        root,
        manifest_sha256=model.bundle_sha256,
        architecture_id=descriptor.architecture_id,
        implementation_version=descriptor.implementation_version,
    )
    if (
        manifest.model_version_id != model.model_version_id
        or manifest.decoder_version != DECODER_VERSION
        or manifest.class_names != DIRECTION_NAMES
    ):
        raise ValueError("registered bundle identity, class order or decoder is incompatible")
    factory = registry.load_backend_factory_on_worker(
        model.architecture_id, model.inference_runtime, worker_id=worker_id
    )
    backend = factory()
    if not isinstance(backend, ModelBackend) or (
        backend.architecture_id,
        backend.implementation_version,
    ) != (descriptor.architecture_id, descriptor.implementation_version):
        raise RegistryError("installed factory does not implement its declared backend")
    predictor = backend.load_bundle(
        root, manifest_sha256=model.bundle_sha256, runtime=model.inference_runtime
    )
    if (
        predictor.architecture_id != model.architecture_id
        or predictor.implementation_version != descriptor.implementation_version
        or predictor.class_names != DIRECTION_NAMES
        or hashlib.sha256(json_bytes(predictor.analysis.model_dump(mode="json"))).hexdigest()
        != manifest.analysis_profile_sha256
    ):
        raise ValueError("restored predictor identity, class order or analysis differs")
    return predictor, manifest


def _available_fields(export: ExportV2) -> set[str]:
    capabilities = export.capabilities
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


def snapshot_inputs(export: ExportV2) -> ObservationInputs:
    """Construct only the raw original grid, never labels or simulation parameters."""
    return ObservationInputs(
        info=tuple(
            PredictionSequenceInfo(
                experiment_id=export.content_sha256,
                instrument_id=instrument.instrument_id,
                step_count=export.time_grid.step_count,
            )
            for instrument in export.instruments
        ),
        observations=tuple(
            tuple(
                row for row in export.observations if row.instrument_id == instrument.instrument_id
            )
            for instrument in export.instruments
        ),
        time_grids=tuple(export.time_grid for _ in export.instruments),
        instruments=export.instruments,
    )


@dataclass(frozen=True, slots=True)
class _AnnotationGrid:
    """Reuse interval validation without constructing generator metadata."""

    experiment_id: str
    export: ExportV2

    @property
    def step_count(self) -> int:
        return self.export.time_grid.step_count

    @property
    def instruments(self) -> tuple[ExportInstrument, ...]:
        return self.export.instruments

    @property
    def observations(self) -> tuple[Observation, ...]:
        return self.export.observations


def build_analysis_result(
    export: ExportV2,
    snapshot: SnapshotRecord,
    analysis: AnalysisRecord,
    model: ModelRecord,
    predictor: LoadedPredictor,
    manifest: ModelBundle,
) -> AnalysisResult:
    inputs = snapshot_inputs(export)
    values = predictor.predict(inputs, runtime=model.inference_runtime)
    if not isinstance(values, np.ndarray) or values.dtype != np.float64:
        raise ValueError("backend probabilities must be a float64 array")
    probabilities_valid(values, inputs.row_count)
    sequences = []
    start = 0
    for info, rows in zip(inputs.info, inputs.observations, strict=True):
        stop = start + info.step_count
        intervals = decode(values[start:stop], rows, predictor.analysis)
        sequences.append(
            PredictionSequence(
                instrument_id=info.instrument_id,
                probabilities=tuple(
                    (float(row[0]), float(row[1]), float(row[2])) for row in values[start:stop]
                ),
                intervals=intervals,
            )
        )
        start = stop
    # This validates predictions only; no automatic reference is read or scored.
    DirectionAnnotations(
        experiment_id=export.content_sha256,
        policy_config=predictor.analysis.model_dump(mode="json"),
        intervals=tuple(item for sequence in sequences for item in sequence.intervals),
    ).validate_for(_AnnotationGrid(export.content_sha256, export))
    assert analysis.operation.model is not None
    return AnalysisResult(
        analysis_id=analysis.analysis_id,
        snapshot_id=snapshot.snapshot_id,
        snapshot_sha256=export.content_sha256,
        source=export.source,
        model=analysis.operation.model,
        architecture_id=manifest.architecture_id,
        implementation_version=manifest.implementation_version,
        decoder_version=manifest.decoder_version,
        analysis_profile_id=manifest.analysis_profile,
        analysis=predictor.analysis,
        inference_runtime=model.inference_runtime,
        class_names=manifest.class_names,
        time_grid=export.time_grid,
        instruments=export.instruments,
        sequences=tuple(sequences),
    )


def compute(payload: dict[str, JsonValue], root: Path) -> dict[str, JsonValue]:
    """Read private materialized inputs; the child has no database or storage client."""
    from regimex.application.report import render_analysis_report

    export = parse_export_v2((root / "export.json").read_bytes())
    snapshot = SnapshotRecord.model_validate_json(json_bytes(payload["snapshot"]))
    analysis = AnalysisRecord.model_validate_json(json_bytes(payload["analysis"]))
    model = ModelRecord.model_validate_json(json_bytes(payload["model"]))
    bundle_path = payload["bundle_path"]
    if not isinstance(bundle_path, str):
        raise ValueError("analysis requires the verified local model path")
    if (
        snapshot.snapshot_id != analysis.snapshot_id
        or analysis.operation.owner_id != snapshot.owner_id
        or snapshot.content_sha256 != export.content_sha256
        or analysis.operation.model is None
        or analysis.operation.model.model_version_id != model.model_version_id
        or analysis.operation.model.bundle_sha256 != model.bundle_sha256
        or analysis.operation.model.inference_policy_sha256 != model.inference_policy_sha256
    ):
        raise ValueError("analysis input identities differ")
    registry = ArchitectureRegistry.discover()
    descriptor = registry.get(model.architecture_id)
    if (
        descriptor.model_dump(mode="json") != payload["descriptor"]
        or descriptor.parameters_model.model_json_schema() != payload["parameter_schema"]
    ):
        raise RegistryError("installed child metadata differs from the configured worker")
    descriptor.validate_observation_fields(_available_fields(export))
    predictor, manifest = restore_predictor(
        Path(bundle_path), model, registry, worker_id="analysis-child"
    )
    result = build_analysis_result(export, snapshot, analysis, model, predictor, manifest)
    result_bytes = json_bytes(result.model_dump(mode="json"))
    result_sha256 = hashlib.sha256(result_bytes).hexdigest()
    rendered = render_analysis_report(export, result, result_sha256=result_sha256)
    (root / "analysis-result.json").write_bytes(result_bytes)
    (root / "analysis-report.html").write_bytes(rendered.html)
    return {
        "result_sha256": result_sha256,
        "report_sha256": hashlib.sha256(rendered.html).hexdigest(),
        "plotly_version": rendered.plotly_version,
    }
