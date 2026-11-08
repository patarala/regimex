"""Transactional review, reference checks and durable deletion receipts."""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Annotated, cast
from uuid import UUID, uuid4

import psycopg
from psycopg.types.json import Jsonb
from pydantic import Field, ValidationError

from regimex.contracts import StrictModel
from regimex.data_catalog import content_hash
from regimex.deletion_contracts import (
    DeletionBlocker,
    DeletionKind,
    DeletionPreview,
    DeletionReceipt,
)
from regimex.infrastructure.database import DatabaseRepository
from regimex.infrastructure.object_store import ObjectInfo
from regimex.operations import Sha256
from regimex.service_contracts import (
    CapacityExceeded,
    Conflict,
    NotFound,
    Principal,
    SnapshotRecord,
)
from regimex.storage import json_bytes

Connection = psycopg.Connection[tuple[object, ...]]
MAX_DELETION_ITEMS = 1000


class StoredDeletionPlan(StrictModel):
    preview: DeletionPreview
    owner_id: str
    source_id: str
    run_id: str
    store_scope_id: str
    snapshot_id: UUID
    resources_sha256: Sha256
    analysis_ids: Annotated[tuple[UUID, ...], Field(max_length=MAX_DELETION_ITEMS)]
    objects: Annotated[tuple[ObjectInfo, ...], Field(max_length=MAX_DELETION_ITEMS)]

    def verify(self) -> None:
        preview = self.preview
        payload = self.model_dump(mode="json")
        payload["preview"]["sha256"] = "0" * 64
        if (
            content_hash(payload) != preview.sha256
            or len(set(self.analysis_ids)) != len(self.analysis_ids)
            or len({item.key for item in self.objects}) != len(self.objects)
            or preview.analysis_count != len(self.analysis_ids)
            or preview.experiment_count != int(preview.target_kind == "experiment")
            or preview.object_count != len(self.objects)
            or preview.size_bytes != sum(item.size_bytes for item in self.objects)
            or (preview.target_kind == "experiment" and preview.target_id != self.snapshot_id)
            or (preview.target_kind == "analysis" and self.analysis_ids != (preview.target_id,))
        ):
            raise Conflict("Stored deletion review failed its integrity check.")


@dataclass(frozen=True)
class _Selection:
    snapshot: SnapshotRecord
    analyses: tuple[tuple[UUID, str], ...]
    objects: tuple[ObjectInfo, ...]
    blockers: tuple[DeletionBlocker, ...]

    @property
    def sha256(self) -> str:
        return content_hash(
            {
                "snapshot": self.snapshot.model_dump(mode="json"),
                "analyses": [(str(identity), state) for identity, state in self.analyses],
                "objects": [value.model_dump(mode="json") for value in self.objects],
                "blockers": [value.model_dump(mode="json") for value in self.blockers],
            }
        )


class DeletionRepository:
    def __init__(self, database: DatabaseRepository) -> None:
        self.database = database

    def authorize(self, connection: Connection, principal: Principal) -> None:
        if principal.role == "integration":
            from regimex.infrastructure.integration_limits import lock_integration
            from regimex.infrastructure.integration_repository import require_integration

            lock_integration(self.database, connection, principal)
            require_integration(connection, principal, scope="delete")

    def _selection(
        self, connection: Connection, principal: Principal, kind: DeletionKind, target_id: UUID
    ) -> _Selection:
        snapshot_id = target_id
        if kind == "analysis":
            row = connection.execute(
                "SELECT snapshot_id FROM regimex_analyses WHERE analysis_id=%s", (target_id,)
            ).fetchone()
            if row is None:
                raise NotFound()
            snapshot_id = cast(UUID, row[0])
        row = connection.execute(
            "SELECT document FROM regimex_snapshots WHERE snapshot_id=%s",
            (snapshot_id,),
        ).fetchone()
        if row is None:
            raise NotFound()
        snapshot = SnapshotRecord.model_validate_json(json_bytes(row[0]))
        self.database._authorize_snapshot(principal, snapshot)
        query = (
            "SELECT a.analysis_id,o.state FROM regimex_analyses a JOIN regimex_operations o "
            "ON o.operation_id=a.operation_id WHERE a.snapshot_id=%s"
        )
        parameters: list[object] = [snapshot_id]
        if kind == "analysis":
            query += " AND a.analysis_id=%s"
            parameters.append(target_id)
        rows = connection.execute(
            query + " ORDER BY a.analysis_id LIMIT %s FOR UPDATE OF o",
            [*parameters, MAX_DELETION_ITEMS + 1],
        ).fetchall()
        if len(rows) > MAX_DELETION_ITEMS:
            raise CapacityExceeded("Remove some analyses before deleting this large experiment.")
        if kind == "analysis" and not rows:
            raise NotFound()
        # Worker commits lock the operation before acquiring foreign-key locks
        # on snapshots/reports. Use that same order to avoid a preview deadlock.
        locked = connection.execute(
            "SELECT document FROM regimex_snapshots WHERE snapshot_id=%s FOR UPDATE",
            (snapshot_id,),
        ).fetchone()
        if locked is None:
            raise NotFound()
        snapshot = SnapshotRecord.model_validate_json(json_bytes(locked[0]))
        analyses = tuple((cast(UUID, row[0]), cast(str, row[1])) for row in rows)
        analysis_ids = [identity for identity, _ in analyses]
        object_rows = connection.execute(
            "SELECT o.document FROM regimex_objects o JOIN regimex_analysis_reports r "
            "ON r.object_key=o.object_key WHERE r.analysis_id=ANY(%s)",
            (analysis_ids,),
        ).fetchall()
        objects = {
            value.key: value for value in (ObjectInfo.model_validate(row[0]) for row in object_rows)
        }
        if kind == "experiment":
            objects[snapshot.object.key] = snapshot.object
        if len(objects) > MAX_DELETION_ITEMS:
            raise CapacityExceeded("Remove some analyses before deleting this large experiment.")
        blockers = []
        active = sum(state in {"queued", "running", "cancel_requested"} for _, state in analyses)
        if active:
            blockers.append(
                DeletionBlocker(
                    code="active_analyses",
                    message="Wait for active analyses to finish or finish cancelling them.",
                    count=active,
                )
            )
        if (
            kind == "experiment"
            and connection.execute(
                "SELECT 1 FROM regimex_catalog_bindings WHERE snapshot_id=%s", (snapshot_id,)
            ).fetchone()
        ):
            blockers.append(
                DeletionBlocker(
                    code="catalog_reference",
                    message="This experiment is registered in the model data catalog.",
                    count=1,
                )
            )
        shared = connection.execute(
            "SELECT count(DISTINCT object_key) FROM ("
            "SELECT object_key FROM regimex_snapshots WHERE snapshot_id<>ALL(%s) UNION ALL "
            "SELECT object_key FROM regimex_analysis_reports WHERE analysis_id<>ALL(%s) UNION ALL "
            "SELECT manifest_key FROM regimex_models UNION ALL "
            "SELECT object_key FROM regimex_model_members UNION ALL "
            "SELECT generation_provenance_key FROM regimex_datasets UNION ALL "
            "SELECT object_key FROM regimex_reference_objects UNION ALL "
            "SELECT report_key FROM regimex_evaluation_records) references_to_keep "
            "WHERE object_key=ANY(%s)",
            ([snapshot_id] if kind == "experiment" else [], analysis_ids, list(objects)),
        ).fetchone()
        assert shared is not None
        if shared[0]:
            blockers.append(
                DeletionBlocker(
                    code="shared_objects",
                    message="Stored files are still referenced by another resource.",
                    count=cast(int, shared[0]),
                )
            )
        return _Selection(
            snapshot, analyses, tuple(objects[key] for key in sorted(objects)), tuple(blockers)
        )

    def preview(
        self,
        principal: Principal,
        target_kind: DeletionKind,
        target_id: UUID,
        *,
        store_scope_id: str,
    ) -> DeletionPreview:
        with self.database.connection() as connection:
            self.authorize(connection, principal)
            self.database._lock(connection, f"owner:{principal.owner_id}")
            connection.execute(
                "DELETE FROM regimex_deletion_plans WHERE owner_id=%s "
                "AND applied_at IS NULL AND expires_at<=clock_timestamp()",
                (principal.owner_id,),
            )
            outstanding = connection.execute(
                "SELECT count(*) FROM regimex_deletion_plans "
                "WHERE owner_id=%s AND applied_at IS NULL",
                (principal.owner_id,),
            ).fetchone()
            if outstanding is not None and cast(int, outstanding[0]) >= 100:
                raise CapacityExceeded("Too many open deletion previews; wait for them to expire.")
            selected = self._selection(connection, principal, target_kind, target_id)
            now = datetime.now(UTC)
            preview = DeletionPreview(
                plan_id=uuid4(),
                sha256="0" * 64,
                target_kind=target_kind,
                target_id=target_id,
                analysis_count=len(selected.analyses),
                experiment_count=int(target_kind == "experiment"),
                object_count=len(selected.objects),
                size_bytes=sum(item.size_bytes for item in selected.objects),
                blockers=selected.blockers,
                created_at=now,
                expires_at=now + timedelta(minutes=15),
            )
            plan = StoredDeletionPlan(
                preview=preview,
                owner_id=principal.owner_id,
                source_id=selected.snapshot.source_id,
                run_id=selected.snapshot.run_id,
                store_scope_id=store_scope_id,
                snapshot_id=selected.snapshot.snapshot_id,
                resources_sha256=selected.sha256,
                analysis_ids=tuple(identity for identity, _ in selected.analyses),
                objects=selected.objects,
            )
            digest = content_hash(plan.model_dump(mode="json"))
            preview = preview.model_copy(update={"sha256": digest})
            plan = plan.model_copy(update={"preview": preview})
            plan.verify()
            connection.execute(
                "INSERT INTO regimex_deletion_plans (plan_id,owner_id,source_id,run_id,"
                "target_kind,target_id,sha256,document,created_at,expires_at) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (
                    preview.plan_id,
                    principal.owner_id,
                    plan.source_id,
                    plan.run_id,
                    target_kind,
                    target_id,
                    digest,
                    Jsonb(plan.model_dump(mode="json")),
                    now,
                    preview.expires_at,
                ),
            )
            return preview

    @staticmethod
    def _scope(principal: Principal) -> tuple[str, list[object]]:
        clause = "owner_id=%s"
        parameters: list[object] = [principal.owner_id]
        if principal.role == "integration":
            clause += " AND source_id=%s"
            parameters.append(principal.source_namespace)
            if principal.run_scope is not None:
                clause += " AND run_id=%s"
                parameters.append(principal.run_scope)
        return clause, parameters

    def plan(
        self, connection: Connection, principal: Principal, plan_id: UUID, *, lock: bool = False
    ) -> tuple[StoredDeletionPlan, bool]:
        clause, parameters = self._scope(principal)
        row = connection.execute(
            "SELECT document,applied_at,plan_id,owner_id,source_id,run_id,target_kind,"
            "target_id,sha256,created_at,expires_at FROM regimex_deletion_plans "
            "WHERE plan_id=%s AND " + clause + (" FOR UPDATE" if lock else ""),
            [plan_id, *parameters],
        ).fetchone()
        if row is None:
            raise NotFound()
        try:
            plan = StoredDeletionPlan.model_validate_json(json_bytes(row[0]))
        except ValidationError as exc:
            raise Conflict("Stored deletion review failed schema validation.") from exc
        plan.verify()
        preview = plan.preview
        if row[2:] != (
            preview.plan_id,
            plan.owner_id,
            plan.source_id,
            plan.run_id,
            preview.target_kind,
            preview.target_id,
            preview.sha256,
            preview.created_at,
            preview.expires_at,
        ):
            raise Conflict("Stored deletion identity differs from its reviewed document.")
        return plan, row[1] is not None

    def apply_metadata(
        self,
        connection: Connection,
        principal: Principal,
        plan_id: UUID,
        *,
        expected_sha256: str,
        store_scope_id: str,
    ) -> StoredDeletionPlan:
        self.authorize(connection, principal)
        self.database._lock(connection, f"owner:{principal.owner_id}")
        plan, applied = self.plan(connection, principal, plan_id, lock=True)
        if expected_sha256 != plan.preview.sha256:
            raise Conflict("The confirmation does not match the reviewed deletion preview.")
        if applied:
            return plan
        if plan.preview.expires_at <= datetime.now(UTC):
            raise Conflict("The deletion preview expired; review the current resources again.")
        if plan.store_scope_id != store_scope_id:
            raise Conflict(
                "Storage changed since this deletion was reviewed; create a new preview."
            )
        selected = self._selection(
            connection, principal, plan.preview.target_kind, plan.preview.target_id
        )
        if selected.sha256 != plan.resources_sha256:
            raise Conflict("The experiment changed; review a new deletion preview.")
        if selected.blockers:
            raise Conflict(
                "This deletion is blocked by active work or retained resource references."
            )
        ids = list(plan.analysis_ids)
        for identity in ids:
            connection.execute(
                "INSERT INTO regimex_deleted_analyses(analysis_id,deletion_id) VALUES (%s,%s)",
                (identity, plan_id),
            )
        if plan.preview.target_kind == "experiment":
            snapshot = selected.snapshot
            connection.execute(
                "INSERT INTO regimex_deleted_snapshots (snapshot_id,owner_id,source_id,run_id,"
                "source_snapshot_id,deletion_id) VALUES (%s,%s,%s,%s,%s,%s)",
                (
                    snapshot.snapshot_id,
                    snapshot.owner_id,
                    snapshot.source_id,
                    snapshot.run_id,
                    snapshot.source_snapshot_id,
                    plan_id,
                ),
            )
        for info in plan.objects:
            connection.execute(
                "INSERT INTO regimex_deleted_object_keys(object_key,deletion_id) VALUES (%s,%s)",
                (info.key, plan_id),
            )
            connection.execute(
                "INSERT INTO regimex_deletion_objects(deletion_id,object_key,document) "
                "VALUES (%s,%s,%s)",
                (plan_id, info.key, Jsonb(info.model_dump(mode="json"))),
            )
        connection.execute("DELETE FROM regimex_analysis_reports WHERE analysis_id=ANY(%s)", (ids,))
        connection.execute("DELETE FROM regimex_analyses WHERE analysis_id=ANY(%s)", (ids,))
        if plan.preview.target_kind == "experiment":
            connection.execute(
                "DELETE FROM regimex_snapshots WHERE snapshot_id=%s", (plan.snapshot_id,)
            )
        connection.execute(
            "DELETE FROM regimex_objects WHERE object_key=ANY(%s)",
            ([item.key for item in plan.objects],),
        )
        connection.execute(
            "UPDATE regimex_deletion_plans SET applied_at=clock_timestamp() WHERE plan_id=%s",
            (plan_id,),
        )
        return plan

    def _objects(
        self, connection: Connection, plan: StoredDeletionPlan
    ) -> tuple[tuple[ObjectInfo, str, str | None], ...]:
        removed_analyses = {
            cast(UUID, row[0])
            for row in connection.execute(
                "SELECT analysis_id FROM regimex_deleted_analyses WHERE deletion_id=%s",
                (plan.preview.plan_id,),
            )
        }
        removed_snapshots = connection.execute(
            "SELECT snapshot_id,owner_id,source_id,run_id FROM regimex_deleted_snapshots "
            "WHERE deletion_id=%s",
            (plan.preview.plan_id,),
        ).fetchall()
        expected_snapshots = (
            [(plan.snapshot_id, plan.owner_id, plan.source_id, plan.run_id)]
            if plan.preview.target_kind == "experiment"
            else []
        )
        if removed_analyses != set(plan.analysis_ids) or removed_snapshots != expected_snapshots:
            raise Conflict("Deletion progress differs from its reviewed resource identities.")
        rows = connection.execute(
            "SELECT d.object_key,d.document,d.state,d.failure,k.deletion_id,"
            "q.owner_id,q.source_id,q.sha256,q.size_bytes,o.object_key "
            "FROM regimex_deletion_objects d "
            "LEFT JOIN regimex_deleted_object_keys k ON k.object_key=d.object_key "
            "LEFT JOIN regimex_storage_reservations q ON q.object_key=d.object_key "
            "LEFT JOIN regimex_objects o ON o.object_key=d.object_key "
            "WHERE d.deletion_id=%s ORDER BY d.object_key",
            (plan.preview.plan_id,),
        ).fetchall()
        expected = {info.key: info for info in plan.objects}
        if len(rows) != len(expected):
            raise Conflict("Deletion progress differs from its reviewed object inventory.")
        result = []
        for (
            key,
            document,
            state,
            failure,
            deletion_id,
            owner,
            source,
            digest,
            size,
            registered,
        ) in rows:
            try:
                info = ObjectInfo.model_validate(document)
            except ValidationError as exc:
                raise Conflict("Deletion progress failed object schema validation.") from exc
            if (
                expected.get(cast(str, key)) != info
                or key != info.key
                or deletion_id != plan.preview.plan_id
                or registered is not None
                or (
                    state == "pending"
                    and (owner, source, digest, size)
                    != (plan.owner_id, plan.source_id, info.sha256, info.size_bytes)
                )
                or (state == "deleted" and owner is not None)
            ):
                raise Conflict("Deletion progress differs from its reviewed object inventory.")
            result.append((info, cast(str, state), cast(str | None, failure)))
        return tuple(result)

    def pending_objects(
        self, connection: Connection, plan: StoredDeletionPlan
    ) -> tuple[ObjectInfo, ...]:
        return tuple(
            info for info, state, _ in self._objects(connection, plan) if state == "pending"
        )

    def record_object(
        self, connection: Connection, plan_id: UUID, info: ObjectInfo, *, failure: str | None
    ) -> None:
        connection.execute(
            "UPDATE regimex_deletion_objects SET state=%s,failure=%s "
            "WHERE deletion_id=%s AND object_key=%s",
            ("deleted" if failure is None else "pending", failure, plan_id, info.key),
        )
        if failure is None:
            connection.execute(
                "DELETE FROM regimex_storage_reservations "
                "WHERE object_key=%s AND sha256=%s AND size_bytes=%s",
                (info.key, info.sha256, info.size_bytes),
            )

    def finish(self, connection: Connection, plan_id: UUID) -> None:
        connection.execute(
            "UPDATE regimex_deletion_plans "
            "SET completed_at=COALESCE(completed_at,clock_timestamp()) "
            "WHERE plan_id=%s AND applied_at IS NOT NULL AND NOT EXISTS ("
            "SELECT 1 FROM regimex_deletion_objects WHERE deletion_id=%s AND state='pending')",
            (plan_id, plan_id),
        )

    def _receipt(
        self, connection: Connection, principal: Principal, plan_id: UUID
    ) -> DeletionReceipt:
        plan, applied = self.plan(connection, principal, plan_id)
        if not applied:
            raise NotFound("This deletion has not been confirmed.")
        rows = self._objects(connection, plan)
        pending = [row for row in rows if row[1] == "pending"]
        completed = connection.execute(
            "SELECT completed_at IS NOT NULL FROM regimex_deletion_plans WHERE plan_id=%s",
            (plan_id,),
        ).fetchone()
        return DeletionReceipt(
            deletion_id=plan_id,
            sha256=plan.preview.sha256,
            target_kind=plan.preview.target_kind,
            target_id=plan.preview.target_id,
            state="completed" if completed == (True,) and not pending else "pending",
            analysis_count=plan.preview.analysis_count,
            experiment_count=plan.preview.experiment_count,
            reclaimed_bytes=sum(row[0].size_bytes for row in rows if row[1] == "deleted"),
            pending_bytes=sum(row[0].size_bytes for row in pending),
            pending_objects=len(pending),
            failure=next((row[2] for row in pending if row[2]), None),
        )

    def receipt(self, principal: Principal, plan_id: UUID) -> DeletionReceipt:
        with self.database.connection() as connection:
            self.authorize(connection, principal)
            return self._receipt(connection, principal, plan_id)

    def list_pending(
        self, principal: Principal, *, limit: int = 50, offset: int = 0
    ) -> tuple[DeletionReceipt, ...]:
        self.database._page(limit, offset)
        with self.database.connection() as connection:
            self.authorize(connection, principal)
            clause, parameters = self._scope(principal)
            rows = connection.execute(
                "SELECT plan_id FROM regimex_deletion_plans WHERE applied_at IS NOT NULL "
                "AND completed_at IS NULL AND "
                + clause
                + " ORDER BY created_at DESC,plan_id DESC LIMIT %s OFFSET %s",
                [*parameters, limit, offset],
            ).fetchall()
            return tuple(self._receipt(connection, principal, cast(UUID, row[0])) for row in rows)
