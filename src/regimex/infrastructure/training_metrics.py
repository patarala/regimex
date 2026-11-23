"""Store an independent, checksummed result for each completed training run."""

import hashlib
from uuid import UUID

import psycopg
from psycopg.types.json import Json

from regimex.admin_contracts import AdminRunRecord, TrainingMetrics
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
        "SELECT o.state,t.committed_model_version_id,m.document FROM regimex_training_runs t "
        "JOIN regimex_operations o USING (operation_id) "
        "LEFT JOIN regimex_training_metrics m USING (operation_id) "
        "WHERE t.operation_id=%s AND o.owner_id=%s",
        (run_id, owner_id),
    ).fetchone()
    if row is None:
        raise NotFound()
    state, committed_model_version_id, document = row
    if document is None:
        return None
    if state != "succeeded":
        raise Conflict("Only a successful training run can have saved metrics.")
    return verified_metrics(document, committed_model_version_id)


def store_training_metrics(
    connection: Connection, run: AdminRunRecord, metrics: TrainingMetrics
) -> TrainingMetrics:
    payload = metrics.model_dump(mode="json")
    document = {"metrics": payload, "sha256": hashlib.sha256(json_bytes(payload)).hexdigest()}
    inserted = connection.execute(
        "INSERT INTO regimex_training_metrics (operation_id,document) "
        "SELECT operation_id,%s FROM regimex_training_runs "
        "WHERE operation_id=%s AND committed_model_version_id=%s "
        "ON CONFLICT (operation_id) DO NOTHING",
        (Json(document), run.run_id, metrics.model_version_id),
    )
    if inserted.rowcount:
        return metrics
    # Two readers can open a pre-migration run before either saves its result.
    saved = read_training_metrics(connection, run.operation.owner_id, run.run_id)
    if saved is None:
        raise Conflict("Metrics differ from the committed training result.")
    if metrics.forest_validation_curve is None:
        return saved
    if (
        saved.forest_validation_curve is not None
        and saved.forest_validation_curve != metrics.forest_validation_curve
    ):
        raise Conflict("A different validation curve is already recorded for this forest.")
    # Registration holds the model lock. Keep the original history and its numeric representation.
    supplemental = {"forest_validation_curve"}
    if json_bytes(saved.model_dump(mode="json", exclude=supplemental)) != json_bytes(
        metrics.model_dump(mode="json", exclude=supplemental)
    ):
        raise Conflict("Forest curve differs from the saved training result.")
    if saved.forest_validation_curve is not None:
        return saved
    connection.execute(
        "UPDATE regimex_training_metrics SET document=%s WHERE operation_id=%s",
        (Json(document), run.run_id),
    )
    return metrics
