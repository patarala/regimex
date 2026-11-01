"""Bounded contracts for explicitly offline database/object recovery."""

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal
from uuid import UUID

from pydantic import AwareDatetime, Field, model_validator

from regimex.contracts import StrictModel
from regimex.infrastructure.maintenance import ServicesStoppedAttestation
from regimex.infrastructure.object_store import ObjectInfo

RecoveryRevision = Literal["0001_initial"]
RECOVERY_REVISION: RecoveryRevision = "0001_initial"
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class RecoveryLimits(StrictModel):
    max_objects: Annotated[int, Field(ge=1, le=100_000)] = 100_000
    max_rows_per_table: Annotated[int, Field(ge=1, le=1_000_000)] = 1_000_000
    max_manifest_bytes: Annotated[int, Field(ge=1)] = 32 * 1024**2
    max_dump_bytes: Annotated[int, Field(ge=1)] = 8 * 1024**3
    max_total_object_bytes: Annotated[int, Field(ge=1)] = 32 * 1024**3
    timeout_seconds: Annotated[int, Field(ge=1)] = 3600


class TableEvidence(StrictModel):
    rows: Annotated[int, Field(ge=0)]
    portable_sha256: Sha256


class MaintenanceIdentity(StrictModel):
    singleton: Literal[True]
    installation_id: UUID
    created_at: AwareDatetime
    restored_from_installation_id: UUID | None
    restored_from_backup_sha256: Sha256 | None

    @model_validator(mode="after")
    def paired_lineage(self) -> MaintenanceIdentity:
        if (self.restored_from_installation_id is None) != (
            self.restored_from_backup_sha256 is None
        ):
            raise ValueError("restore lineage requires both source installation and backup digest")
        return self


class BackupManifest(StrictModel):
    version: Literal["offline-backup/1.0"] = "offline-backup/1.0"
    backup_id: UUID
    schema_name: Annotated[str, Field(pattern=r"^[a-z_][a-z0-9_]{0,62}$")]
    migration_revision: RecoveryRevision
    created_at: AwareDatetime
    attestation: ServicesStoppedAttestation
    postgres_version: Annotated[str, Field(min_length=1, max_length=256)]
    dump_tool_version: Annotated[str, Field(min_length=1, max_length=256)]
    dump_size_bytes: Annotated[int, Field(ge=0)]
    dump_sha256: Sha256
    objects: Annotated[tuple[ObjectInfo, ...], Field(max_length=100_000)]
    tables: Annotated[dict[str, TableEvidence], Field(min_length=1, max_length=1000)]
    source_maintenance_identity: MaintenanceIdentity
    scope: Literal["all_registered_objects"] = "all_registered_objects"

    @model_validator(mode="after")
    def complete_inventory(self) -> BackupManifest:
        if self.attestation.stopped_at > self.created_at:
            raise ValueError("the stopped-services attestation cannot come from the future")
        if len({item.key for item in self.objects}) != len(self.objects):
            raise ValueError("backup object keys must be unique")
        return self


class BackupResult(StrictModel):
    manifest: BackupManifest
    manifest_sha256: Sha256


class RestoreReceipt(StrictModel):
    version: Literal["offline-restore/1.0"] = "offline-restore/1.0"
    backup_id: UUID
    backup_manifest_sha256: Sha256
    restored_at: AwareDatetime
    attestation: ServicesStoppedAttestation
    schema_name: str
    object_count: Annotated[int, Field(ge=0)]
    changed_versions: Annotated[int, Field(ge=0)]
    table_evidence: dict[str, TableEvidence]
    new_installation_id: UUID
    new_registered_claims: Annotated[int, Field(ge=0)] = 0
    new_claims_sha256: Sha256 | None = None


@dataclass(frozen=True, slots=True)
class PostgresTools:
    pg_dump: Path
    pg_restore: Path

    @classmethod
    def discover(
        cls, *, pg_dump: Path | None = None, pg_restore: Path | None = None
    ) -> PostgresTools:
        def executable(explicit: Path | None, environment: str, name: str) -> Path:
            selected = str(explicit) if explicit is not None else os.environ.get(environment)
            selected = selected or shutil.which(name)
            if selected is None:
                raise ValueError(f"Set {environment} or install {name} on PATH.")
            path = Path(selected).resolve(strict=True)
            if not path.is_file() or not os.access(path, os.X_OK):
                raise ValueError(f"The configured {name} is not an executable file.")
            return path

        return cls(
            executable(pg_dump, "PGDUMP_BIN", "pg_dump"),
            executable(pg_restore, "PGRESTORE_BIN", "pg_restore"),
        )
