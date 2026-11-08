"""Reviewable removal of saved experiments and completed analyses."""

from typing import Annotated, Literal
from uuid import UUID

from pydantic import AwareDatetime, Field

from regimex.contracts import StrictModel
from regimex.operations import Sha256

DeletionKind = Literal["analysis", "experiment"]
Count = Annotated[int, Field(ge=0)]


class DeletionBlocker(StrictModel):
    code: Literal["active_analyses", "catalog_reference", "shared_objects"]
    message: str
    count: Count


class DeletionPreview(StrictModel):
    plan_id: UUID
    sha256: Sha256
    target_kind: DeletionKind
    target_id: UUID
    analysis_count: Count
    experiment_count: Count
    object_count: Count
    size_bytes: Count
    blockers: tuple[DeletionBlocker, ...]
    created_at: AwareDatetime
    expires_at: AwareDatetime


class DeletionReceipt(StrictModel):
    deletion_id: UUID
    sha256: Sha256
    target_kind: DeletionKind
    target_id: UUID
    state: Literal["pending", "completed"]
    analysis_count: Count
    experiment_count: Count
    reclaimed_bytes: Count
    pending_bytes: Count
    pending_objects: Count
    failure: str | None = None
