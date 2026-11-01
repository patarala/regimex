"""Conservative offline orphan cleanup with reviewed plans and durable fences."""

import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol
from uuid import UUID, uuid4

from regimex.data_catalog import content_hash
from regimex.infrastructure.maintenance import ServicesStoppedAttestation
from regimex.infrastructure.object_inventory import IncompleteUpload, StoredVersion
from regimex.infrastructure.object_store import (
    ObjectConflictError,
    ObjectInfo,
    ObjectIntegrityError,
)
from regimex.infrastructure.retention_repository import Connection, RetentionRepository
from regimex.retention_contracts import (
    CleanupEvent,
    CleanupReceipt,
    CleanupStatus,
    OrphanObject,
    OrphanPlan,
    OrphanPlanPayload,
)
from regimex.service_contracts import CapacityExceeded, Conflict, Principal
from regimex.storage import json_bytes


class MaintenanceStore(Protocol):
    @property
    def scope_id(self) -> str: ...

    def inventory(self) -> tuple[StoredVersion, ...]: ...
    def stat_version(self, key: str, version_id: str) -> ObjectInfo: ...
    def delete_version(self, expected: ObjectInfo) -> None: ...
    def list_multipart_uploads(self) -> tuple[IncompleteUpload, ...]: ...
    def abort_multipart(
        self, expected: IncompleteUpload, *, initiated_before: datetime
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class RetentionLimits:
    minimum_grace_seconds: int = 86400
    max_inventory_entries: int = 10_000
    max_plan_entries: int = 1000
    max_verified_bytes: int = 2 * 1024**3
    max_plan_bytes: int = 4 * 1024**2
    max_wall_seconds: int = 600

    def __post_init__(self) -> None:
        for name in self.__slots__:
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError("Retention limits must be positive integers.")
        if self.max_plan_entries > 1000:
            raise ValueError("An offline orphan plan supports at most 1000 candidates.")


DEFAULT_RETENTION_LIMITS = RetentionLimits()


class RetentionService:
    def __init__(
        self,
        repository: RetentionRepository,
        store: MaintenanceStore,
        *,
        limits: RetentionLimits = DEFAULT_RETENTION_LIMITS,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.repository = repository
        self.store = store
        self.limits = limits
        self.clock = clock

    def _deadline(self) -> float:
        return time.monotonic() + self.limits.max_wall_seconds

    @staticmethod
    def _check_time(deadline: float) -> None:
        if time.monotonic() >= deadline:
            raise CapacityExceeded("Offline retention exceeded its wall-time limit.")

    def _check_plan(self, connection: Connection, plan: OrphanPlan) -> None:
        if plan.installation_id != self.repository.installation_id(connection):
            raise Conflict("The reviewed plan belongs to another database installation.")
        if plan.store_scope_id != self.store.scope_id:
            raise Conflict("The reviewed plan belongs to another object-store location.")
        if plan.grace_seconds < self.limits.minimum_grace_seconds:
            raise Conflict("The reviewed plan no longer satisfies the configured grace period.")
        if len(plan.objects) + len(plan.multipart) > self.limits.max_plan_entries:
            raise CapacityExceeded("The reviewed plan exceeds the configured candidate limit.")
        if sum(item.object.size_bytes for item in plan.objects) > self.limits.max_verified_bytes:
            raise CapacityExceeded(
                "The reviewed plan exceeds the configured verification byte limit."
            )

    def plan(
        self,
        principal: Principal,
        *,
        attestation: ServicesStoppedAttestation,
        grace_seconds: int = 86400,
    ) -> OrphanPlan:
        if type(grace_seconds) is not int or grace_seconds < self.limits.minimum_grace_seconds:
            raise ValueError("Orphan cleanup requires the configured minimum grace period.")
        deadline = self._deadline()
        with (
            self.repository.maintenance(principal, attestation) as connection,
            self.repository.quiescent_transaction(
                connection, maximum_idle_seconds=self.limits.max_wall_seconds
            ),
        ):
            created_at = self.clock()
            cutoff = created_at - timedelta(seconds=grace_seconds)
            inventory = self.store.inventory()
            multipart = self.store.list_multipart_uploads()
            if len(inventory) + len(multipart) > self.limits.max_inventory_entries:
                raise CapacityExceeded("Object inventory exceeds the configured entry limit.")
            registered = self.repository.registered_keys(
                connection,
                tuple({item.key for item in inventory} | {item.key for item in multipart}),
            )
            objects: list[OrphanObject] = []
            uploads: list[IncompleteUpload] = []
            verified_bytes = 0
            for version in sorted(inventory, key=lambda item: (item.key, item.version_id)):
                self._check_time(deadline)
                if (
                    version.is_delete_marker
                    or version.key in registered
                    or version.last_modified >= cutoff
                ):
                    continue
                verified_bytes += version.size_bytes
                if verified_bytes > self.limits.max_verified_bytes:
                    raise CapacityExceeded("Orphan verification exceeds its byte limit.")
                if len(objects) >= self.limits.max_plan_entries:
                    raise CapacityExceeded("Orphan inventory exceeds its review plan limit.")
                info = self.store.stat_version(version.key, version.version_id)
                if (info.key, info.version_id, info.size_bytes) != (
                    version.key,
                    version.version_id,
                    version.size_bytes,
                ):
                    raise ObjectIntegrityError("Object inventory changed during verification.")
                objects.append(OrphanObject(object=info, last_modified=version.last_modified))
            for upload in sorted(multipart, key=lambda item: (item.key, item.upload_id)):
                self._check_time(deadline)
                if upload.key in registered or upload.initiated_at >= cutoff:
                    continue
                if len(objects) + len(uploads) >= self.limits.max_plan_entries:
                    raise CapacityExceeded("Orphan inventory exceeds its review plan limit.")
                uploads.append(upload)
            payload = OrphanPlanPayload(
                plan_id=uuid4(),
                installation_id=self.repository.installation_id(connection),
                owner_id=principal.owner_id,
                store_scope_id=self.store.scope_id,
                created_at=created_at,
                cutoff_at=cutoff,
                grace_seconds=grace_seconds,
                attestation=attestation,
                registered_objects=self.repository.registered_count(connection),
                objects=tuple(objects),
                multipart=tuple(uploads),
            ).model_dump(mode="json")
            plan = OrphanPlan.model_validate_json(
                json_bytes({**payload, "sha256": content_hash(payload)})
            )
            if len(plan.model_dump_json().encode()) > self.limits.max_plan_bytes:
                raise CapacityExceeded("The review plan exceeds its serialized byte limit.")
            self._check_time(deadline)
            self.repository.add_plan(connection, plan)
            return plan

    def get_plan(self, principal: Principal, plan_id: UUID) -> OrphanPlan:
        self.repository.database._admin(principal)
        with self.repository.database.connection() as connection:
            return self.repository.plan(connection, principal, plan_id)

    def export_plan(self, principal: Principal, plan_id: UUID, destination: Path) -> OrphanPlan:
        plan = self.get_plan(principal, plan_id)
        data = json_bytes(plan.model_dump(mode="json"))
        if len(data) > self.limits.max_plan_bytes:
            raise CapacityExceeded("The review plan exceeds its serialized byte limit.")
        descriptor = os.open(
            destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
        )
        try:
            with os.fdopen(descriptor, "wb", closefd=False) as stream:
                stream.write(data)
                stream.flush()
                os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return plan

    def apply(
        self,
        principal: Principal,
        plan_id: UUID,
        *,
        expected_sha256: str,
        attestation: ServicesStoppedAttestation,
    ) -> CleanupReceipt:
        deadline = self._deadline()
        with self.repository.maintenance(principal, attestation) as connection:
            with self.repository.quiescent_transaction(
                connection, maximum_idle_seconds=self.limits.max_wall_seconds
            ):
                plan = self.repository.plan(connection, principal, plan_id)
                if expected_sha256 != plan.sha256:
                    raise Conflict("The supplied review digest does not identify this orphan plan.")
                self._check_plan(connection, plan)
                for upload in plan.multipart:
                    if self.repository.registered(connection, upload.key):
                        raise Conflict(
                            "A multipart candidate became registered; create a fresh plan."
                        )
                self.repository.fence(connection, plan)
                self._check_time(deadline)
            # The transaction above MUST commit before any external deletion.
            # Registration inserts every ObjectInfo; the SQL trigger fences its
            # exact version before an external object can be deleted.
            successful = {
                (item.kind, item.key, item.identity)
                for item in self.repository.events(connection, plan_id)
                if item.status
                in {"deleted", "already_missing", "protected", "mpu_aborted", "mpu_already_missing"}
            }
            for item in plan.objects:
                self._check_time(deadline)
                info = item.object
                if ("object", info.key, info.version_id) in successful:
                    continue
                if self.repository.registered(connection, info.key):
                    status: CleanupStatus = "protected"
                else:
                    try:
                        current = self.store.stat_version(info.key, info.version_id)
                        if current != info:
                            raise ObjectConflictError("A reviewed object has changed.")
                        self.store.delete_version(info)
                        status = "deleted"
                    except FileNotFoundError:
                        status = "already_missing"
                    except ObjectConflictError, ObjectIntegrityError:
                        status = "changed"
                    except OSError, ValueError:
                        status = "failed"
                self.repository.append_event(
                    connection,
                    plan_id,
                    kind="object",
                    key=info.key,
                    identity=info.version_id,
                    status=status,
                )
            for upload in plan.multipart:
                self._check_time(deadline)
                if ("multipart", upload.key, upload.upload_id) in successful:
                    continue
                if self.repository.registered(connection, upload.key):
                    status = "protected"
                else:
                    try:
                        self.store.abort_multipart(upload, initiated_before=plan.cutoff_at)
                        status = "mpu_aborted"
                    except FileNotFoundError:
                        status = "mpu_already_missing"
                    except ObjectConflictError, ObjectIntegrityError:
                        status = "mpu_changed"
                    except OSError, ValueError:
                        status = "mpu_failed"
                self.repository.append_event(
                    connection,
                    plan_id,
                    kind="multipart",
                    key=upload.key,
                    identity=upload.upload_id,
                    status=status,
                )
            return self._receipt(connection, plan)

    def _receipt(self, connection: Connection, plan: OrphanPlan) -> CleanupReceipt:
        events = self.repository.events(connection, plan.plan_id)
        latest: dict[tuple[str, str, str], CleanupEvent] = {}
        for event in events:
            latest[(event.kind, event.key, event.identity)] = event
        return CleanupReceipt(
            plan_id=plan.plan_id,
            plan_sha256=plan.sha256,
            installation_id=plan.installation_id,
            store_scope_id=plan.store_scope_id,
            inspected_at=self.clock(),
            events=events,
            complete=len(latest) == len(plan.objects) + len(plan.multipart)
            and all(
                item.status not in {"changed", "failed", "mpu_changed", "mpu_failed"}
                for item in latest.values()
            ),
        )

    def get_receipt(self, principal: Principal, plan_id: UUID) -> CleanupReceipt:
        self.repository.database._admin(principal)
        with self.repository.database.connection() as connection:
            return self._receipt(connection, self.repository.plan(connection, principal, plan_id))
