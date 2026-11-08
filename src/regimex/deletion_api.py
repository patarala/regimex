"""Reviewed deletion of owned resources and resumable storage cleanup."""

from collections.abc import Callable
from enum import Enum
from typing import Annotated, Literal, Protocol
from uuid import UUID

from fastapi import Depends, FastAPI, Query, Response

from regimex.api import _json_dependency, _request_body
from regimex.contracts import StrictModel
from regimex.deletion_contracts import DeletionPreview, DeletionReceipt
from regimex.operations import Sha256
from regimex.service_contracts import Principal


class DeletionAPIService(Protocol):
    def preview(
        self, principal: Principal, target_kind: Literal["analysis", "experiment"], target_id: UUID
    ) -> DeletionPreview: ...

    def apply(
        self, principal: Principal, plan_id: UUID, *, expected_sha256: str
    ) -> DeletionReceipt: ...

    def receipt(self, principal: Principal, plan_id: UUID) -> DeletionReceipt: ...

    def list_pending(
        self, principal: Principal, *, limit: int = 50, offset: int = 0
    ) -> tuple[DeletionReceipt, ...]: ...


class DeletionConfirmation(StrictModel):
    sha256: Sha256


class DeletionPage(StrictModel):
    items: tuple[DeletionReceipt, ...]
    limit: int
    offset: int


def install_deletion_routes(
    app: FastAPI, service: DeletionAPIService, *, authenticate: Callable[..., Principal]
) -> None:
    Identity = Annotated[Principal, Depends(authenticate)]
    Confirmation = Annotated[DeletionConfirmation, Depends(_json_dependency(DeletionConfirmation))]
    Limit = Annotated[int, Query(ge=1, le=100)]
    Offset = Annotated[int, Query(ge=0, le=1_000_000)]
    tags: list[str | Enum] = ["Resource deletion"]
    app.state.request_models += (DeletionConfirmation,)

    @app.post("/api/v1/analyses/{analysis_id}/deletion-preview", tags=tags)
    def preview_analysis(analysis_id: UUID, identity: Identity) -> DeletionPreview:
        return service.preview(identity, "analysis", analysis_id)

    @app.post("/api/v1/experiments/{snapshot_id}/deletion-preview", tags=tags)
    def preview_experiment(snapshot_id: UUID, identity: Identity) -> DeletionPreview:
        return service.preview(identity, "experiment", snapshot_id)

    @app.post(
        "/api/v1/deletions/{plan_id}",
        tags=tags,
        openapi_extra=_request_body(DeletionConfirmation),
    )
    def confirm_deletion(
        plan_id: UUID, identity: Identity, confirmation: Confirmation, response: Response
    ) -> DeletionReceipt:
        receipt = service.apply(identity, plan_id, expected_sha256=confirmation.sha256)
        response.status_code = 200 if receipt.state == "completed" else 202
        return receipt

    @app.get("/api/v1/deletions/{plan_id}", tags=tags)
    def deletion_receipt(plan_id: UUID, identity: Identity) -> DeletionReceipt:
        return service.receipt(identity, plan_id)

    @app.get("/api/v1/deletions", tags=tags)
    def pending_deletions(
        identity: Identity, limit: Limit = 50, offset: Offset = 0
    ) -> DeletionPage:
        return DeletionPage(
            items=service.list_pending(identity, limit=limit, offset=offset),
            limit=limit,
            offset=offset,
        )
