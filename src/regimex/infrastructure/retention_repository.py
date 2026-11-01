"""PostgreSQL maintenance authority, immutable plans and deletion fencing."""

from collections.abc import Iterator
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from typing import cast
from uuid import UUID

import psycopg
from psycopg.types.json import Jsonb

from regimex.infrastructure.database import DatabaseRepository
from regimex.infrastructure.maintenance import (
    ServicesStoppedAttestation,
    maintenance_lock,
    maintenance_unlock,
    quiescent_transaction,
)
from regimex.retention_contracts import CleanupEvent, CleanupStatus, OrphanPlan
from regimex.service_contracts import CapacityExceeded, Conflict, NotFound, Principal
from regimex.storage import json_bytes

Connection = psycopg.Connection[tuple[object, ...]]


class RetentionRepository:
    def __init__(self, database: DatabaseRepository) -> None:
        self.database = database

    @contextmanager
    def maintenance(
        self, principal: Principal, attestation: ServicesStoppedAttestation
    ) -> Iterator[Connection]:
        self.database._admin(principal)
        attestation = ServicesStoppedAttestation.model_validate_json(attestation.model_dump_json())
        if attestation.stopped_at > datetime.now(UTC):
            raise Conflict("A stopped-services assertion cannot come from the future.")
        with self.database.connection() as connection:
            connection.autocommit = True
            connection.execute("SET application_name='regimex-offline-maintenance'")
            maintenance_lock(connection, self.database.schema)
            try:
                yield connection
            finally:
                with suppress(psycopg.Error):
                    maintenance_unlock(connection, self.database.schema)

    @staticmethod
    @contextmanager
    def quiescent_transaction(
        connection: Connection, *, maximum_idle_seconds: int
    ) -> Iterator[None]:
        with quiescent_transaction(connection, maximum_idle_seconds=maximum_idle_seconds):
            yield

    @staticmethod
    def installation_id(connection: Connection) -> UUID:
        row = connection.execute(
            "SELECT installation_id FROM regimex_maintenance_identity WHERE singleton"
        ).fetchone()
        if row is None:
            raise Conflict("Maintenance identity is not initialized.")
        return cast(UUID, row[0])

    @staticmethod
    def registered_keys(connection: Connection, candidates: tuple[str, ...]) -> frozenset[str]:
        # No age, publication or reachability heuristic permits collecting a
        # registered artifact. Older and otherwise unreferenced ledger rows stay.
        return frozenset(
            cast(str, row[0])
            for row in connection.execute(
                "SELECT object_key FROM regimex_objects WHERE object_key=ANY(%s)",
                (list(candidates),),
            )
        )

    @staticmethod
    def registered_count(connection: Connection) -> int:
        row = connection.execute("SELECT count(*) FROM regimex_objects").fetchone()
        if row is None:
            raise Conflict("Object registration inventory is unavailable.")
        return cast(int, row[0])

    @staticmethod
    def registered(connection: Connection, key: str) -> bool:
        return (
            connection.execute(
                "SELECT 1 FROM regimex_objects WHERE object_key=%s", (key,)
            ).fetchone()
            is not None
        )

    @staticmethod
    def add_plan(connection: Connection, plan: OrphanPlan) -> None:
        plan = OrphanPlan.model_validate_json(plan.model_dump_json())
        connection.execute(
            "INSERT INTO regimex_orphan_plans "
            "(plan_id,installation_id,owner_id,store_scope_id,sha256,document) "
            "VALUES (%s,%s,%s,%s,%s,%s)",
            (
                plan.plan_id,
                plan.installation_id,
                plan.owner_id,
                plan.store_scope_id,
                plan.sha256,
                Jsonb(plan.model_dump(mode="json")),
            ),
        )

    @staticmethod
    def plan(connection: Connection, principal: Principal, plan_id: UUID) -> OrphanPlan:
        DatabaseRepository._admin(principal)
        row = connection.execute(
            "SELECT document,sha256 FROM regimex_orphan_plans WHERE plan_id=%s AND owner_id=%s",
            (plan_id, principal.owner_id),
        ).fetchone()
        if row is None:
            raise NotFound()
        plan = OrphanPlan.model_validate_json(json_bytes(row[0]))
        if plan.sha256 != row[1]:
            raise Conflict("The stored orphan plan differs from its review digest.")
        return plan

    @staticmethod
    def fence(connection: Connection, plan: OrphanPlan) -> None:
        for item in plan.objects:
            info = item.object
            if RetentionRepository.registered(connection, info.key):
                raise Conflict("An orphan candidate became registered; create a fresh review plan.")
            connection.execute(
                "INSERT INTO regimex_object_version_claims(object_key,version_id,state) "
                "VALUES (%s,%s,'fenced') ON CONFLICT (object_key,version_id) DO NOTHING",
                (info.key, info.version_id),
            )
            if connection.execute(
                "SELECT state FROM regimex_object_version_claims "
                "WHERE object_key=%s AND version_id=%s",
                (info.key, info.version_id),
            ).fetchone() != ("fenced",):
                raise Conflict("The object version has already been admitted by a writer.")
            connection.execute(
                "INSERT INTO regimex_object_delete_fences "
                "(object_key,version_id,sha256,size_bytes,plan_id,store_scope_id) "
                "VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT (object_key,version_id) DO NOTHING",
                (
                    info.key,
                    info.version_id,
                    info.sha256,
                    info.size_bytes,
                    plan.plan_id,
                    plan.store_scope_id,
                ),
            )
            row = connection.execute(
                "SELECT sha256,size_bytes,store_scope_id FROM regimex_object_delete_fences "
                "WHERE object_key=%s AND version_id=%s",
                (info.key, info.version_id),
            ).fetchone()
            if row != (info.sha256, info.size_bytes, plan.store_scope_id):
                raise Conflict("An object version has conflicting maintenance evidence.")

    @staticmethod
    def append_event(
        connection: Connection,
        plan_id: UUID,
        *,
        kind: str,
        key: str,
        identity: str,
        status: CleanupStatus,
    ) -> None:
        connection.execute(
            "INSERT INTO regimex_orphan_cleanup_events "
            "(plan_id,kind,object_key,identity,status) VALUES (%s,%s,%s,%s,%s)",
            (plan_id, kind, key, identity, status),
        )

    @staticmethod
    def events(connection: Connection, plan_id: UUID) -> tuple[CleanupEvent, ...]:
        rows = connection.execute(
            "SELECT event_id,kind,object_key,identity,status,occurred_at "
            "FROM regimex_orphan_cleanup_events WHERE plan_id=%s ORDER BY event_id LIMIT 10001",
            (plan_id,),
        ).fetchall()
        if len(rows) > 10_000:
            raise CapacityExceeded("This plan has reached its bounded audit retrieval limit.")
        return tuple(
            CleanupEvent.model_validate(
                {
                    "event_id": row[0],
                    "plan_id": plan_id,
                    "kind": row[1],
                    "key": row[2],
                    "identity": row[3],
                    "status": row[4],
                    "occurred_at": row[5],
                },
                strict=True,
            )
            for row in rows
        )
