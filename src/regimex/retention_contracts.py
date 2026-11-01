"""Bounded offline retention policy; registered artifacts are never collected."""

from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import AwareDatetime, Field, model_validator

from regimex.contracts import Identifier, StrictModel
from regimex.data_catalog import content_hash
from regimex.infrastructure.maintenance import ServicesStoppedAttestation
from regimex.infrastructure.object_inventory import IncompleteUpload
from regimex.infrastructure.object_store import ObjectInfo
from regimex.operations import Sha256

StoreScope = Annotated[str, Field(pattern=r"^(local|s3):[0-9a-f]{64}$")]


class OrphanObject(StrictModel):
    object: ObjectInfo
    last_modified: AwareDatetime


class OrphanPlanPayload(StrictModel):
    version: Literal["offline-orphan-plan/1.0"] = "offline-orphan-plan/1.0"
    plan_id: UUID
    installation_id: UUID
    owner_id: Identifier
    store_scope_id: StoreScope
    created_at: AwareDatetime
    cutoff_at: AwareDatetime
    grace_seconds: Annotated[int, Field(ge=1)]
    attestation: ServicesStoppedAttestation
    policy: Literal["preserve_all_registered_v1"] = "preserve_all_registered_v1"
    registered_objects: Annotated[int, Field(ge=0)]
    objects: Annotated[tuple[OrphanObject, ...], Field(max_length=1000)]
    multipart: Annotated[tuple[IncompleteUpload, ...], Field(max_length=1000)]

    @model_validator(mode="after")
    def integrity(self) -> Self:
        if self.cutoff_at >= self.created_at:
            raise ValueError("An orphan plan requires an earlier grace cutoff.")
        if (self.created_at - self.cutoff_at).total_seconds() != self.grace_seconds:
            raise ValueError("The grace cutoff differs from the recorded retention policy.")
        if self.attestation.stopped_at > self.created_at:
            raise ValueError("A stopped-services assertion cannot come from the future.")
        identities = [(item.object.key, item.object.version_id) for item in self.objects]
        if len(set(identities)) != len(identities):
            raise ValueError("An orphan plan cannot contain duplicate object versions.")
        uploads = [(item.key, item.upload_id) for item in self.multipart]
        if len(set(uploads)) != len(uploads):
            raise ValueError("An orphan plan cannot contain duplicate multipart uploads.")
        if any(item.last_modified >= self.cutoff_at for item in self.objects):
            raise ValueError("An object has not passed its grace period.")
        if any(item.initiated_at >= self.cutoff_at for item in self.multipart):
            raise ValueError("A multipart upload has not passed its grace period.")
        return self


class OrphanPlan(OrphanPlanPayload):
    sha256: Sha256

    @model_validator(mode="after")
    def review_digest(self) -> Self:
        if self.sha256 != content_hash(self.model_dump(mode="json", exclude={"sha256"})):
            raise ValueError("The orphan plan differs from its review digest.")
        return self


CleanupStatus = Literal[
    "deleted",
    "already_missing",
    "changed",
    "failed",
    "protected",
    "mpu_aborted",
    "mpu_already_missing",
    "mpu_changed",
    "mpu_failed",
]


class CleanupEvent(StrictModel):
    event_id: int
    plan_id: UUID
    kind: Literal["object", "multipart"]
    key: str
    identity: str
    status: CleanupStatus
    occurred_at: AwareDatetime


class CleanupReceipt(StrictModel):
    plan_id: UUID
    plan_sha256: Sha256
    installation_id: UUID
    store_scope_id: StoreScope
    inspected_at: AwareDatetime
    events: tuple[CleanupEvent, ...]
    complete: bool
