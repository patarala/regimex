"""Administrator routes for platform connections and their service credentials."""

from collections.abc import Callable
from enum import Enum
from typing import Annotated
from uuid import UUID

from fastapi import Depends, FastAPI, Query, Response
from fastapi.responses import JSONResponse
from pydantic import Field

from regimex.access_contracts import (
    AccessEventView,
    IntegrationCreate,
    IntegrationKeyCreate,
    IntegrationKeyView,
    IntegrationUpdate,
    IntegrationView,
)
from regimex.api import _json_dependency, _request_body
from regimex.contracts import StrictModel
from regimex.infrastructure.integration_repository import IntegrationRepository
from regimex.service_contracts import Principal


class IntegrationPage(StrictModel):
    items: tuple[IntegrationView, ...]
    limit: int
    offset: int


class IntegrationKeyPage(StrictModel):
    items: tuple[IntegrationKeyView, ...]
    limit: int
    offset: int


class AccessEventPage(StrictModel):
    items: tuple[AccessEventView, ...]
    limit: int
    offset: int


class IntegrationKeyIssued(StrictModel):
    key: IntegrationKeyView
    token: Annotated[str, Field(min_length=47, max_length=47, repr=False)]


def install_integration_routes(
    app: FastAPI,
    repository: IntegrationRepository,
    *,
    authenticate: Callable[..., Principal],
) -> None:
    Identity = Annotated[Principal, Depends(authenticate)]
    Limit = Annotated[int, Query(ge=1, le=100)]
    Offset = Annotated[int, Query(ge=0, le=1_000_000)]
    CreateBody = Annotated[IntegrationCreate, Depends(_json_dependency(IntegrationCreate))]
    UpdateBody = Annotated[IntegrationUpdate, Depends(_json_dependency(IntegrationUpdate))]
    KeyBody = Annotated[IntegrationKeyCreate, Depends(_json_dependency(IntegrationKeyCreate))]
    app.state.request_models += (IntegrationCreate, IntegrationUpdate, IntegrationKeyCreate)
    prefix = "/api/v1/admin/integrations"
    tags: list[str | Enum] = ["Integration administration"]

    @app.get(prefix, tags=tags)
    def connections(identity: Identity, limit: Limit = 50, offset: Offset = 0) -> IntegrationPage:
        return IntegrationPage(
            items=repository.list(identity, limit=limit, offset=offset), limit=limit, offset=offset
        )

    @app.post(prefix, status_code=201, tags=tags, openapi_extra=_request_body(IntegrationCreate))
    def create_connection(
        identity: Identity, intake: CreateBody, response: Response
    ) -> IntegrationView:
        record = repository.create(identity, intake)
        response.headers["Location"] = f"{prefix}/{record.integration_id}"
        return record

    @app.get(prefix + "/{integration_id}", tags=tags)
    def connection(integration_id: UUID, identity: Identity) -> IntegrationView:
        return repository.get(identity, integration_id)

    @app.patch(
        prefix + "/{integration_id}", tags=tags, openapi_extra=_request_body(IntegrationUpdate)
    )
    def update_connection(
        integration_id: UUID, identity: Identity, intake: UpdateBody
    ) -> IntegrationView:
        return repository.update(identity, integration_id, intake)

    @app.get(prefix + "/{integration_id}/keys", tags=tags)
    def keys(
        integration_id: UUID, identity: Identity, limit: Limit = 50, offset: Offset = 0
    ) -> IntegrationKeyPage:
        return IntegrationKeyPage(
            items=repository.list_keys(identity, integration_id, limit=limit, offset=offset),
            limit=limit,
            offset=offset,
        )

    @app.post(
        prefix + "/{integration_id}/keys",
        status_code=201,
        response_model=IntegrationKeyIssued,
        tags=tags,
        openapi_extra=_request_body(IntegrationKeyCreate),
    )
    def issue_key(integration_id: UUID, identity: Identity, intake: KeyBody) -> JSONResponse:
        issued = repository.issue_key(identity, integration_id, intake)
        # The only response containing this secret. Metadata and audit routes
        # never retrieve or reproduce a credential.
        return JSONResponse(
            {"key": issued.key.model_dump(mode="json"), "token": issued.token.get_secret_value()},
            status_code=201,
            headers={"Cache-Control": "private, no-store", "Pragma": "no-cache"},
        )

    @app.post(prefix + "/{integration_id}/keys/{token_id}/revoke", tags=tags, status_code=204)
    def revoke_key(integration_id: UUID, token_id: UUID, identity: Identity) -> Response:
        repository.revoke_key(identity, integration_id, token_id)
        return Response(status_code=204, headers={"Cache-Control": "private, no-store"})

    @app.get(prefix + "/{integration_id}/events", tags=tags)
    def events(
        integration_id: UUID, identity: Identity, limit: Limit = 50, offset: Offset = 0
    ) -> AccessEventPage:
        return AccessEventPage(
            items=repository.events(identity, integration_id, limit=limit, offset=offset),
            limit=limit,
            offset=offset,
        )
