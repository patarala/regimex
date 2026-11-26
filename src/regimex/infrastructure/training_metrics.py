"""Store an independent, checksummed result for each completed training run."""

import hashlib
from uuid import UUID

import psycopg
from psycopg.types.json import Json

from regimex.admin_contracts import (
    AdminRunRecord,
    ImportedTrainingSpec,
    RepeatOutcome,
    TrainingMetrics,
)
from regimex.direction_quality import DirectionQualityEvidence
from regimex.service_contracts import Conflict, NotFound
from regimex.storage import json_bytes

Connection = psycopg.Connection[tuple[object, ...]]


def verified_metrics(document: object, committed_model_version_id: object) -> TrainingMetrics:
    if not isinstance(document, dict) or document.keys() != {"metrics", "sha256"}:
        raise Conflict("Stored training metrics have an invalid format.")
    try:
        payload = json_bytes(document["metrics"])
        metrics = TrainingMetrics.model_validate_json(payload)
    except (ValueError, RecursionError) as exc:
        raise Conflict("Stored training metrics have an invalid format.") from exc
    if hashlib.sha256(payload).hexdigest() != document["sha256"]:
        raise Conflict("Stored training metrics differ from their saved checksum.")
    if metrics.model_version_id != committed_model_version_id:
        raise Conflict("Stored training metrics refer to a different model version.")
    return metrics


def read_training_metrics(
    connection: Connection, owner_id: str, run_id: UUID
) -> TrainingMetrics | None:
    row = connection.execute(
        "SELECT o.state,t.committed_model_version_id,m.document,a.request "
        "FROM regimex_training_runs t "
        "JOIN regimex_operations o USING (operation_id) "
        "LEFT JOIN regimex_training_metrics m USING (operation_id) "
        "LEFT JOIN regimex_admin_runs a ON a.operation_id=t.operation_id AND t.external "
        "WHERE t.operation_id=%s AND o.owner_id=%s",
        (run_id, owner_id),
    ).fetchone()
    if row is None:
        raise NotFound()
    state, committed_model_version_id, document, imported_request = row
    if document is None:
        return None
    if state != "succeeded":
        raise Conflict("Only a successful training run can have saved metrics.")
    metrics = verified_metrics(document, committed_model_version_id)
    # Earlier imports saved the policy and history separately, even after weight deletion.
    if (
        imported_request is not None
        and not metrics.repeat_outcomes
        and isinstance(metrics.history, dict)
    ):
        spec = ImportedTrainingSpec.model_validate_json(json_bytes(imported_request))
        outcomes = RepeatOutcome.from_history(spec, metrics.history)
        if outcomes:
            return metrics.model_copy(update={"repeat_outcomes": outcomes})
    return metrics


def _metrics_document(metrics: TrainingMetrics) -> Json:
    payload = metrics.model_dump(mode="json")
    return Json({"metrics": payload, "sha256": hashlib.sha256(json_bytes(payload)).hexdigest()})


def store_training_metrics(
    connection: Connection, run: AdminRunRecord, metrics: TrainingMetrics
) -> TrainingMetrics:
    inserted = connection.execute(
        "INSERT INTO regimex_training_metrics (operation_id,document) "
        "SELECT operation_id,%s FROM regimex_training_runs "
        "WHERE operation_id=%s AND committed_model_version_id=%s "
        "ON CONFLICT (operation_id) DO NOTHING",
        (_metrics_document(metrics), run.run_id, metrics.model_version_id),
    )
    if inserted.rowcount:
        return metrics
    # Two readers can open a pre-migration run before either saves its result.
    saved = read_training_metrics(connection, run.operation.owner_id, run.run_id)
    if saved is None:
        raise Conflict("Metrics differ from the committed training result.")
    if metrics.forest_validation_curve is None and metrics.direction_quality is None:
        return saved
    if (
        metrics.forest_validation_curve is not None
        and saved.forest_validation_curve is not None
        and saved.forest_validation_curve != metrics.forest_validation_curve
    ):
        raise Conflict("A different validation curve is already recorded for this forest.")
    # Registration holds the model lock. Keep the original history and its numeric representation.
    supplemental = {"forest_validation_curve", "direction_quality"}
    if json_bytes(saved.model_dump(mode="json", exclude=supplemental)) != json_bytes(
        metrics.model_dump(mode="json", exclude=supplemental)
    ):
        raise Conflict("Additional measurements differ from the saved training result.")
    merged = saved.model_copy(
        update={
            "forest_validation_curve": metrics.forest_validation_curve
            or saved.forest_validation_curve,
            "direction_quality": _merge_direction_quality(
                saved.direction_quality, metrics.direction_quality
            ),
        }
    )
    if merged == saved:
        return saved
    connection.execute(
        "UPDATE regimex_training_metrics SET document=%s WHERE operation_id=%s",
        (_metrics_document(merged), run.run_id),
    )
    return merged


def _merge_direction_quality(
    saved: DirectionQualityEvidence | None, supplied: DirectionQualityEvidence | None
) -> DirectionQualityEvidence | None:
    if saved is None or supplied is None:
        return saved or supplied
    if (saved.protocol_sha256, saved.bundle_sha256) != (
        supplied.protocol_sha256,
        supplied.bundle_sha256,
    ):
        raise Conflict("A different direction quality protocol or bundle is already recorded.")
    old = {(row.reference, row.split): row for row in saved.rows}
    new = {(row.reference, row.split): row for row in supplied.rows}
    if any(old[key] != new[key] for key in old.keys() & new.keys()):
        raise Conflict("A different measurement is already recorded for this reference and split.")
    if new.keys() <= old.keys():
        return saved
    if old.keys() <= new.keys():
        return supplied
    # All groups must be checked together for overlap, including groups from earlier imports.
    raise Conflict("Include the previously recorded rows when adding direction measurements.")
