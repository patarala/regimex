"""Eligibility checks for releasing one trained candidate's weight object."""

from typing import cast
from uuid import UUID

import psycopg

from regimex.deletion_contracts import DeletionBlocker
from regimex.infrastructure.object_store import ObjectInfo
from regimex.infrastructure.training_metrics import verified_metrics
from regimex.service_contracts import Conflict, ModelRecord, NotFound
from regimex.storage import json_bytes

Connection = psycopg.Connection[tuple[object, ...]]


def review_candidate(
    connection: Connection, owner_id: str, run_id: UUID, weight: ObjectInfo
) -> tuple[str, tuple[DeletionBlocker, ...]]:
    # Evaluation, publication and benchmark selection hold a shared lock when
    # pinning a candidate. After this exclusive lock, reference checks see
    # every earlier pin, including requests from another owner.
    row = connection.execute(
        "SELECT m.document,m.weights_deletion_id,metrics.document "
        "FROM regimex_training_runs t JOIN regimex_operations o USING(operation_id) "
        "JOIN regimex_models m ON m.model_version_id=t.committed_model_version_id "
        "LEFT JOIN regimex_training_metrics metrics USING(operation_id) "
        "WHERE t.operation_id=%s AND o.owner_id=%s AND o.state='succeeded' FOR UPDATE OF m",
        (run_id, owner_id),
    ).fetchone()
    if row is None:
        raise NotFound()
    if row[1] is not None:
        raise Conflict("These weights already have a deletion receipt.")
    if row[2] is None:
        raise Conflict("Save the training history before removing its weights.")
    model = ModelRecord.model_validate_json(json_bytes(row[0]))
    identity = model.model_version_id
    verified_metrics(row[2], identity)
    paths = [path for path, info in model.members.items() if info == weight]
    if len(paths) != 1:
        raise Conflict("The selected weights no longer match a unique registered model member.")
    path = paths[0]
    stored = connection.execute(
        "SELECT o.document FROM regimex_model_members mm JOIN regimex_objects o "
        "USING(object_key) WHERE mm.model_version_id=%s AND mm.member_path=%s",
        (identity, path),
    ).fetchone()
    if stored is None or ObjectInfo.model_validate(stored[0]) != weight:
        raise Conflict("The weight generation differs from its registered model member.")
    row = connection.execute(
        "SELECT "
        "EXISTS (SELECT 1 FROM regimex_publication_audit WHERE model_version_id=%(id)s "
        "OR previous_model_version_id=%(id)s) OR EXISTS (SELECT 1 FROM regimex_deployment "
        "WHERE model_version_id=%(id)s), "
        "EXISTS (SELECT 1 FROM regimex_benchmark_choices WHERE model_version_id=%(id)s), "
        "EXISTS (SELECT 1 FROM regimex_analyses WHERE model_version_id=%(id)s), "
        "(SELECT count(*) FROM regimex_operations o JOIN ("
        "SELECT operation_id FROM regimex_evaluation_requests WHERE model_version_id=%(id)s "
        "UNION ALL SELECT operation_id FROM regimex_publication_requests "
        "WHERE model_version_id=%(id)s) r USING(operation_id) "
        "WHERE o.state IN ('queued','running','cancel_requested')), "
        "EXISTS (SELECT 1 FROM regimex_model_members WHERE object_key=%(key)s "
        "AND (model_version_id,member_path)<>(%(id)s,%(path)s) "
        "UNION ALL SELECT 1 FROM regimex_models WHERE manifest_key=%(key)s "
        "UNION ALL SELECT 1 FROM regimex_storage_reservations WHERE object_key=%(key)s "
        "UNION ALL SELECT 1 FROM regimex_snapshots WHERE object_key=%(key)s "
        "UNION ALL SELECT 1 FROM regimex_analysis_reports WHERE object_key=%(key)s "
        "UNION ALL SELECT 1 FROM regimex_datasets WHERE generation_provenance_key=%(key)s "
        "UNION ALL SELECT 1 FROM regimex_reference_objects WHERE object_key=%(key)s "
        "UNION ALL SELECT 1 FROM regimex_evaluation_records WHERE report_key=%(key)s)",
        {"id": identity, "path": path, "key": weight.key},
    ).fetchone()
    assert row is not None
    published, frozen, analyzed, active, shared = row
    blockers = []
    if published:
        blockers.append(
            DeletionBlocker(
                code="published_model",
                count=1,
                message="Published model versions must remain available.",
            )
        )
    if frozen:
        blockers.append(
            DeletionBlocker(
                code="benchmark_choice",
                count=1,
                message="A benchmark is frozen on this model version.",
            )
        )
    if analyzed:
        blockers.append(
            DeletionBlocker(
                code="analysis_reference", count=1, message="An analysis uses this model version."
            )
        )
    if active:
        blockers.append(
            DeletionBlocker(
                code="active_model_jobs",
                count=cast(int, active),
                message="Wait for evaluation or publication to finish.",
            )
        )
    if shared:
        blockers.append(
            DeletionBlocker(
                code="shared_objects",
                count=1,
                message="Another resource uses the same stored weights.",
            )
        )
    return identity, tuple(blockers)


def detach_candidate_weights(
    connection: Connection, model_id: str, weight: ObjectInfo, plan_id: UUID
) -> None:
    row = connection.execute(
        "DELETE FROM regimex_model_members WHERE model_version_id=%s AND object_key=%s "
        "RETURNING member_path",
        (model_id, weight.key),
    ).fetchone()
    if row is None:
        raise Conflict("The reviewed weight member is no longer registered.")
    # Historical evaluations keep their model foreign key. Only live objects
    # remain in the embedded inventory that a backup must restore and rebind.
    connection.execute(
        "UPDATE regimex_models SET weights_deletion_id=%s, "
        "document=document #- ARRAY['members',%s::text] WHERE model_version_id=%s",
        (plan_id, row[0], model_id),
    )
