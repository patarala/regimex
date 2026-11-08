"""Reviewed removal with resumable, exact-generation storage cleanup."""

import time
from typing import TYPE_CHECKING
from uuid import UUID

from regimex.deletion_contracts import DeletionKind, DeletionPreview, DeletionReceipt
from regimex.infrastructure.object_store import ObjectConflictError, ObjectInfo, ObjectStore
from regimex.infrastructure.resource_lifecycle import owner_payload_lock
from regimex.service_contracts import Conflict, Principal

if TYPE_CHECKING:
    from regimex.infrastructure.deletion_repository import DeletionRepository


class DeletionService:
    def __init__(self, repository: DeletionRepository, store: ObjectStore) -> None:
        self.repository = repository
        self.store = store

    def preview(
        self, principal: Principal, target_kind: DeletionKind, target_id: UUID
    ) -> DeletionPreview:
        return self.repository.preview(
            principal, target_kind, target_id, store_scope_id=self.store.scope_id
        )

    def receipt(self, principal: Principal, plan_id: UUID) -> DeletionReceipt:
        return self.repository.receipt(principal, plan_id)

    def list_pending(
        self, principal: Principal, *, limit: int = 50, offset: int = 0
    ) -> tuple[DeletionReceipt, ...]:
        return self.repository.list_pending(principal, limit=limit, offset=offset)

    def _remove(self, info: ObjectInfo) -> None:
        try:
            current = self.store.stat_version(info.key, info.version_id)
        except FileNotFoundError:
            current = None
        if current is not None:
            if current != info:
                raise ObjectConflictError("The reviewed object generation changed.")
            self.store.delete_version(info)
        # A successful provider response is not itself absence evidence. Verify
        # the exact generation and refuse to release bytes for a replacement key.
        try:
            self.store.stat_version(info.key, info.version_id)
        except FileNotFoundError:
            pass
        else:
            raise ObjectConflictError("The reviewed object generation is still present.")
        try:
            self.store.stat(info.key)
        except FileNotFoundError:
            return
        raise ObjectConflictError("Another object generation occupies the removed key.")

    def apply(
        self, principal: Principal, plan_id: UUID, *, expected_sha256: str
    ) -> DeletionReceipt:
        from psycopg.errors import ForeignKeyViolation, UniqueViolation

        deadline = time.monotonic() + 60
        with owner_payload_lock(self.repository.database, principal.owner_id) as connection:
            try:
                with connection.transaction():
                    plan = self.repository.apply_metadata(
                        connection,
                        principal,
                        plan_id,
                        expected_sha256=expected_sha256,
                        store_scope_id=self.store.scope_id,
                    )
            except (ForeignKeyViolation, UniqueViolation) as exc:
                raise Conflict(
                    "Resource references changed; review a new deletion preview."
                ) from exc
            # Tombstones and visibility changes must commit before any object
            # disappears. The owner's upload lock remains held across this work.
            pending = self.repository.pending_objects(connection, plan)
            if pending and plan.store_scope_id != self.store.scope_id:
                raise Conflict("Pending deletion belongs to another storage location.")
            for info in pending:
                if time.monotonic() >= deadline:
                    break
                failure = None
                try:
                    self._remove(info)
                except ObjectConflictError:
                    failure = "Stored files changed. Ask the administrator to inspect storage."
                except OSError, ValueError:
                    failure = "Some files could not be removed. Retry to finish freeing storage."
                with connection.transaction():
                    self.repository.record_object(connection, plan_id, info, failure=failure)
            with connection.transaction():
                self.repository.finish(connection, plan_id)
                return self.repository._receipt(connection, principal, plan_id)
