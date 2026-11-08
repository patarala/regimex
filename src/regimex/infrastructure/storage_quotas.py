"""Durable object reservations shared by API processes and workers.

An upload is charged before touching storage. Failed and interrupted uploads
remain charged: their bytes might exist even when the caller saw an error.
Only offline maintenance with a complete inventory can release absent objects.
"""

from datetime import datetime
from typing import TYPE_CHECKING, cast
from uuid import UUID

from regimex.infrastructure.object_store import ObjectInfo
from regimex.service_contracts import Conflict, NotFound, Principal
from regimex.storage_contracts import StorageQuotaExceeded, StorageUsage

if TYPE_CHECKING:
    import psycopg

    from regimex.infrastructure.database import DatabaseRepository


def integration_for_owner(
    connection: psycopg.Connection[tuple[object, ...]], owner_id: str, source_id: str
) -> UUID | None:
    row = connection.execute(
        "SELECT i.integration_id FROM regimex_integrations i "
        "WHERE i.source_namespace=%s AND (i.owner_id=%s OR EXISTS ("
        "SELECT 1 FROM regimex_integration_subjects s "
        "WHERE s.integration_owner_id=i.owner_id AND s.source_namespace=i.source_namespace "
        "AND s.effective_owner_id=%s))",
        (source_id, owner_id, owner_id),
    ).fetchone()
    return None if row is None else cast(UUID, row[0])


def record_owned_object(
    connection: psycopg.Connection[tuple[object, ...]],
    owner_id: str,
    source_id: str,
    info: ObjectInfo,
) -> None:
    """Record a verified resource's charge in its visibility transaction.

    Application writers reserve first. This also accounts for trusted repository
    imports and fixtures, whose callers already control admission.
    """
    integration_id = integration_for_owner(connection, owner_id, source_id)
    connection.execute(
        "INSERT INTO regimex_storage_reservations "
        "(object_key,owner_id,source_id,integration_id,sha256,size_bytes) "
        "VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT(object_key) DO NOTHING",
        (info.key, owner_id, source_id, integration_id, info.sha256, info.size_bytes),
    )
    _verify_reservation(connection, owner_id, source_id, integration_id, info)


def _verify_reservation(
    connection: psycopg.Connection[tuple[object, ...]],
    owner_id: str,
    source_id: str,
    integration_id: UUID | None,
    info: ObjectInfo,
) -> None:
    row = connection.execute(
        "SELECT owner_id,source_id,integration_id,sha256,size_bytes "
        "FROM regimex_storage_reservations WHERE object_key=%s",
        (info.key,),
    ).fetchone()
    if row != (owner_id, source_id, integration_id, info.sha256, info.size_bytes):
        raise Conflict("The object key already has a different storage reservation.")


class StorageQuotaRepository:
    def __init__(
        self,
        database: DatabaseRepository,
        *,
        max_owner_storage_bytes: int = 1024**3,
        max_integration_storage_bytes: int = 2 * 1024**3,
    ) -> None:
        if (
            any(
                type(value) is not int or not 1 <= value <= 2**63 - 1
                for value in (max_owner_storage_bytes, max_integration_storage_bytes)
            )
            or max_owner_storage_bytes > max_integration_storage_bytes
        ):
            raise ValueError("Storage limits must be positive, with owner <= connection.")
        self.database = database
        self.max_owner_storage_bytes = max_owner_storage_bytes
        self.max_integration_storage_bytes = max_integration_storage_bytes

    @staticmethod
    def _usage(
        connection: psycopg.Connection[tuple[object, ...]],
        identity: str | UUID,
        maximum: int,
        *,
        integration: bool = False,
    ) -> StorageUsage:
        field = "integration_id" if integration else "owner_id"
        row = connection.execute(
            "SELECT COALESCE(sum(r.size_bytes) FILTER (WHERE o.object_key IS NOT NULL),0), "
            "COALESCE(sum(r.size_bytes) FILTER (WHERE o.object_key IS NULL),0) "
            "FROM regimex_storage_reservations r LEFT JOIN regimex_objects o "
            "ON o.object_key=r.object_key WHERE r." + field + "=%s",
            (identity,),
        ).fetchone()
        assert row is not None
        used, reserved = int(cast(int, row[0])), int(cast(int, row[1]))
        return StorageUsage(
            used_bytes=used,
            reserved_bytes=reserved,
            limit_bytes=maximum,
            available_bytes=max(0, maximum - used - reserved),
        )

    def owner_usage(self, principal: Principal) -> StorageUsage:
        with self.database.connection() as connection:
            return self._usage(connection, principal.owner_id, self.max_owner_storage_bytes)

    def integration_usage(self, principal: Principal, integration_id: UUID) -> StorageUsage:
        self.database._admin(principal)
        with self.database.connection() as connection:
            if (
                connection.execute(
                    "SELECT 1 FROM regimex_integrations WHERE integration_id=%s", (integration_id,)
                ).fetchone()
                is None
            ):
                raise NotFound()
            return self._usage(
                connection, integration_id, self.max_integration_storage_bytes, integration=True
            )

    def reserve(
        self, owner_id: str, source_id: str, info: ObjectInfo, *, principal: Principal | None = None
    ) -> None:
        """Admit one immutable payload; retries reuse its original reservation."""
        with self.database.connection() as connection:
            from regimex.infrastructure.resource_lifecycle import require_object_key

            require_object_key(connection, info.key)
            integration_id = integration_for_owner(connection, owner_id, source_id)
            if principal is not None and principal.role == "integration":
                from regimex.infrastructure.integration_limits import lock_integration
                from regimex.infrastructure.integration_repository import require_integration

                if principal.owner_id != owner_id or principal.integration_id != integration_id:
                    raise Conflict("The storage owner differs from the authenticated connection.")
                lock_integration(self.database, connection, principal)
                require_integration(connection, principal)
            if integration_id is not None:
                self.database._lock(connection, f"storage-integration:{integration_id}")
            self.database._lock(connection, f"storage-owner:{owner_id}")
            self.database._lock(connection, f"storage-object:{info.key}")
            if (
                connection.execute(
                    "SELECT 1 FROM regimex_storage_reservations WHERE object_key=%s", (info.key,)
                ).fetchone()
                is not None
            ):
                _verify_reservation(connection, owner_id, source_id, integration_id, info)
                return
            owner = self._usage(connection, owner_id, self.max_owner_storage_bytes)
            if info.size_bytes > owner.available_bytes:
                raise StorageQuotaExceeded()
            if integration_id is not None:
                integration = self._usage(
                    connection, integration_id, self.max_integration_storage_bytes, integration=True
                )
                if info.size_bytes > integration.available_bytes:
                    raise StorageQuotaExceeded(integration=True)
            record_owned_object(connection, owner_id, source_id, info)


def release_absent_reservations(
    connection: psycopg.Connection[tuple[object, ...]],
    existing_keys: frozenset[str],
    *,
    created_before: datetime,
) -> int:
    """Use only under offline maintenance after a complete object/MPU inventory.

    Registered objects remain charged even if missing from a damaged store. The
    grace cutoff keeps a recent failed upload available for an idempotent retry.
    """
    result = connection.execute(
        "DELETE FROM regimex_storage_reservations r WHERE created_at<%s "
        "AND NOT (r.object_key=ANY(%s)) AND NOT EXISTS "
        "(SELECT 1 FROM regimex_objects o WHERE o.object_key=r.object_key) AND NOT EXISTS "
        "(SELECT 1 FROM regimex_deletion_objects d WHERE d.object_key=r.object_key "
        "AND d.state='pending')",
        (created_before, list(existing_keys)),
    )
    return result.rowcount
