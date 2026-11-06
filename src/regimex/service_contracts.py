"""Lightweight persistent-service records; importing these never imports science."""

from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import AwareDatetime, Field, JsonValue, model_validator

from regimex.access_contracts import IntegrationScope
from regimex.contracts import MAX_STEPS, Identifier, StrictModel
from regimex.infrastructure.object_store import ObjectInfo
from regimex.model_registry import InferenceRuntime
from regimex.operations import Operation, OperationState, Sha256

Role = Literal["user", "admin", "integration"]
HistoryState = OperationState | Literal["active"]


class AnalysisHistoryFilter(StrictModel):
    q: Annotated[str, Field(max_length=128, pattern=r"^[^\x00]*$")] = ""
    state: HistoryState | None = None


class Principal(StrictModel):
    owner_id: Identifier
    role: Role
    token_id: UUID
    source_namespace: Identifier | None = None
    run_scope: Annotated[str, Field(pattern=r"^[A-Za-z0-9._:@-]{1,128}$")] | None = None
    integration_id: UUID | None = None
    integration_scopes: tuple[IntegrationScope, ...] = ()

    @model_validator(mode="after")
    def trusted_run_scope(self) -> Self:
        if self.role != "integration" and (self.integration_id or self.integration_scopes):
            raise ValueError("Only integration principals can carry integration permissions.")
        if self.integration_id is not None and (
            self.source_namespace is None or not self.integration_scopes
        ):
            raise ValueError("Managed integrations require a source and explicit permissions.")
        if self.run_scope is not None and (
            self.role != "integration" or self.source_namespace is None
        ):
            raise ValueError("Only a source-bound integration principal can carry a run scope.")
        return self


class SnapshotRecord(StrictModel):
    snapshot_id: UUID
    owner_id: Identifier
    source_id: Identifier
    run_id: Identifier
    source_snapshot_id: Identifier
    content_sha256: Sha256
    object: ObjectInfo
    created_at: AwareDatetime
    instrument_ids: Annotated[tuple[Identifier, ...], Field(max_length=2)] = ()
    step_count: Annotated[int, Field(ge=1, le=MAX_STEPS)] | None = None

    @model_validator(mode="after")
    def complete_display_metadata(self) -> Self:
        # Earlier saved snapshots have neither field. Their history remains
        # usable without opening raw objects or inventing instrument metadata.
        if bool(self.instrument_ids) != (self.step_count is not None):
            raise ValueError("instrument identifiers and step count must be supplied together")
        if len(set(self.instrument_ids)) != len(self.instrument_ids):
            raise ValueError("snapshot instrument identifiers must be unique")
        return self


class ModelRecord(StrictModel):
    model_version_id: Identifier
    architecture_id: Identifier
    bundle_sha256: Sha256
    manifest: ObjectInfo
    members: dict[str, ObjectInfo]
    inference_runtime: InferenceRuntime
    inference_policy_sha256: Sha256
    quality_status: Literal["validation_only"] = "validation_only"
    created_at: AwareDatetime


class AnalysisRecord(StrictModel):
    analysis_id: UUID
    snapshot_id: UUID
    operation: Operation
    result: dict[str, JsonValue] | None = None


class AnalysisSummary(StrictModel):
    """Pollable metadata; result bytes are loaded only by the result endpoint."""

    analysis_id: UUID
    snapshot_id: UUID
    source_id: Identifier
    run_id: Identifier
    source_snapshot_id: Identifier
    instrument_ids: tuple[Identifier, ...] = ()
    step_count: Annotated[int, Field(ge=1, le=MAX_STEPS)] | None = None
    operation: Operation
    result_available: bool
    report_available: bool = False


class ServiceError(ValueError):
    def __init__(self, code: str, message: str, status_code: int) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.integration_id: UUID | None = None
        self.token_id: UUID | None = None


class NotFound(ServiceError):
    def __init__(self, message: str = "Resource was not found.") -> None:
        super().__init__("not_found", message, 404)


class Conflict(ServiceError):
    def __init__(self, message: str = "The request conflicts with persisted state.") -> None:
        super().__init__("conflict", message, 409)


class UnsupportedContract(ServiceError):
    def __init__(self) -> None:
        super().__init__(
            "unsupported_contract",
            "The stored resource uses an unsupported contract version.",
            409,
        )


class Forbidden(ServiceError):
    def __init__(self, message: str = "This credential cannot perform this action.") -> None:
        super().__init__("forbidden", message, 403)


class Unauthorized(ServiceError):
    def __init__(self) -> None:
        super().__init__("unauthorized", "A valid bearer credential is required.", 401)


class CapacityExceeded(ServiceError):
    def __init__(self, message: str = "The owner's pending operation quota is exhausted.") -> None:
        super().__init__("capacity_limit", message, 429)


class StaleAttempt(Conflict):
    def __init__(self) -> None:
        super().__init__("The attempt is stale or cancellation prevents this commit.")
