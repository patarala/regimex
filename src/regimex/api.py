"""Authenticated HTTP intake and reads over the persistent analysis application.

The API validates bounded JSON and exposes committed state. It does not import a
backend, probe computation dependencies, fit a model, or accept local file paths.
"""

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Annotated, Any, Literal, Protocol
from uuid import UUID

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response, Security
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from psycopg import Error as DatabaseError
from pydantic import AwareDatetime, BaseModel, JsonValue, ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from regimex.contracts import MAX_INPUT_BYTES, Identifier, StrictModel
from regimex.exports import ExportV2
from regimex.infrastructure.object_store import ObjectIntegrityError
from regimex.model_registry import ArchitectureInfo
from regimex.operations import (
    Operation,
    OperationFailure,
    OperationKind,
    OperationState,
    ResolvedModel,
)
from regimex.service_contracts import (
    AnalysisRecord,
    AnalysisSummary,
    Forbidden,
    ModelRecord,
    NotFound,
    Principal,
    ServiceError,
    SnapshotRecord,
    Unauthorized,
)

if TYPE_CHECKING:
    from regimex.admin_api import AdminAPIService
    from regimex.application.auth import AuthService
    from regimex.benchmark_api import BenchmarkAPIService


class AnalysisAPIService(Protocol):
    """The HTTP boundary depends on application operations, never SQL or models."""

    def authenticate(self, raw_token: str) -> Principal: ...

    def readiness(self) -> dict[str, JsonValue]: ...

    def create_snapshot(self, principal: Principal, export: ExportV2) -> SnapshotRecord: ...

    def get_snapshot(self, principal: Principal, snapshot_id: UUID) -> SnapshotRecord: ...

    def read_snapshot(self, principal: Principal, snapshot_id: UUID) -> ExportV2: ...

    def submit_analysis(
        self,
        principal: Principal,
        snapshot_id: UUID,
        *,
        model_version_id: str = "active",
        idempotency_key: str,
    ) -> AnalysisRecord: ...

    def list_analysis_summaries(
        self, principal: Principal, *, limit: int = 100, offset: int = 0
    ) -> tuple[AnalysisSummary, ...]: ...

    def get_analysis_summary(self, principal: Principal, analysis_id: UUID) -> AnalysisSummary: ...

    def get_result(self, principal: Principal, analysis_id: UUID) -> dict[str, JsonValue]: ...

    def read_report(self, principal: Principal, analysis_id: UUID) -> bytes: ...

    def get_operation(self, principal: Principal, operation_id: UUID) -> Operation: ...

    def cancel_operation(self, principal: Principal, operation_id: UUID) -> Operation: ...

    def list_models(
        self, principal: Principal, *, limit: int = 100, offset: int = 0
    ) -> tuple[ModelRecord, ...]: ...

    def get_model(self, principal: Principal, model_version_id: str) -> ModelRecord: ...

    def list_architectures(self, principal: Principal) -> tuple[ArchitectureInfo, ...]: ...


class AnalysisRequest(StrictModel):
    snapshot_id: UUID
    model_version_id: Identifier = "active"


class ValidationResult(StrictModel):
    schema_version: str
    content_sha256: str
    source_id: str
    run_id: str
    source_snapshot_id: str
    step_count: int
    instrument_ids: tuple[str, ...]


class SnapshotView(StrictModel):
    snapshot_id: UUID
    source_id: str
    run_id: str
    source_snapshot_id: str
    content_sha256: str
    created_at: AwareDatetime
    export_url: str


class OperationView(StrictModel):
    operation_id: UUID
    kind: OperationKind
    resource_id: str
    state: OperationState
    stage: str
    attempt: int
    max_attempts: int
    model: ResolvedModel | None
    created_at: AwareDatetime
    updated_at: AwareDatetime
    heartbeat_at: AwareDatetime | None
    failure: OperationFailure | None
    polling_url: str


class AnalysisView(StrictModel):
    analysis_id: UUID
    snapshot_id: UUID
    job: OperationView
    result_url: str
    result_available: bool
    report_url: str
    report_available: bool = False
    polling_url: str


class AnalysisPage(StrictModel):
    items: tuple[AnalysisView, ...]
    limit: int
    offset: int


class ModelView(StrictModel):
    model_version_id: str
    architecture_id: str
    bundle_sha256: str
    inference_policy_sha256: str
    quality_status: Literal["validation_only"]
    created_at: AwareDatetime


class ModelPage(StrictModel):
    items: tuple[ModelView, ...]
    limit: int
    offset: int


class ArchitecturePage(StrictModel):
    items: tuple[ArchitectureInfo, ...]
    limit: int
    offset: int


def _error(code: str, message: str) -> dict[str, JsonValue]:
    return {"error": {"code": code, "message": message}}


class BoundedJSONBody:
    """Reject excess bytes before handing a request to JSON parsing or storage."""

    def __init__(self, app: ASGIApp, *, max_bytes: int = MAX_INPUT_BYTES) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] != "http"
            or not scope["path"].startswith("/api/v1/")
            or scope["method"] not in {"POST", "PUT", "PATCH"}
        ):
            await self.app(scope, receive, send)
            return
        headers = scope.get("headers", [])
        lengths = [value for key, value in headers if key.lower() == b"content-length"]
        try:
            if len(lengths) > 1:
                raise ValueError
            declared = int(lengths[0]) if lengths else None
            if declared is not None and declared < 0:
                raise ValueError
        except ValueError:
            await JSONResponse(_error("invalid_request", "Invalid Content-Length."), 400)(
                scope, receive, send
            )
            return
        if declared is not None and declared > self.max_bytes:
            await JSONResponse(_error("capacity_limit", "JSON exceeds the 16 MiB limit."), 413)(
                scope, receive, send
            )
            return
        encodings = [value for key, value in headers if key.lower() == b"content-encoding"]
        if encodings and encodings != [b"identity"]:
            await JSONResponse(
                _error("invalid_request", "Encoded request bodies are unsupported."), 415
            )(scope, receive, send)
            return
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            if message["type"] != "http.request":
                continue
            chunk = message.get("body", b"")
            if len(body) + len(chunk) > self.max_bytes:
                await JSONResponse(_error("capacity_limit", "JSON exceeds the 16 MiB limit."), 413)(
                    scope, receive, send
                )
                return
            body.extend(chunk)
            if not message.get("more_body", False):
                break
        if declared is not None and declared != len(body):
            await JSONResponse(
                _error("invalid_request", "Content-Length does not match the body."), 400
            )(scope, receive, send)
            return
        delivered = False

        async def replay() -> Message:
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        await self.app(scope, replay, send)


def _snapshot(record: SnapshotRecord, *, prefix: str = "/api/v1") -> SnapshotView:
    return SnapshotView(
        snapshot_id=record.snapshot_id,
        source_id=record.source_id,
        run_id=record.run_id,
        source_snapshot_id=record.source_snapshot_id,
        content_sha256=record.content_sha256,
        created_at=record.created_at,
        export_url=f"{prefix}/snapshots/{record.snapshot_id}/export",
    )


def _operation(operation: Operation, *, prefix: str = "/api/v1") -> OperationView:
    # Fencing tokens and request/idempotency material remain internal to workers.
    return OperationView(
        operation_id=operation.operation_id,
        kind=operation.kind,
        resource_id=operation.resource_id,
        state=operation.state,
        stage=operation.stage,
        attempt=operation.attempt,
        max_attempts=operation.max_attempts,
        model=operation.model,
        created_at=operation.created_at,
        updated_at=operation.updated_at,
        heartbeat_at=operation.heartbeat_at,
        failure=operation.failure,
        polling_url=f"{prefix}/jobs/{operation.operation_id}",
    )


def _analysis(record: AnalysisSummary, *, prefix: str = "/api/v1") -> AnalysisView:
    return AnalysisView(
        analysis_id=record.analysis_id,
        snapshot_id=record.snapshot_id,
        job=_operation(record.operation, prefix=prefix),
        result_url=f"{prefix}/analyses/{record.analysis_id}/result",
        result_available=record.result_available,
        report_url=f"{prefix}/analyses/{record.analysis_id}/report",
        report_available=record.report_available,
        polling_url=f"{prefix}/analyses/{record.analysis_id}",
    )


def _model(record: ModelRecord) -> ModelView:
    return ModelView(
        model_version_id=record.model_version_id,
        architecture_id=record.architecture_id,
        bundle_sha256=record.bundle_sha256,
        inference_policy_sha256=record.inference_policy_sha256,
        quality_status=record.quality_status,
        created_at=record.created_at,
    )


def _prefix(identity: Principal) -> str:
    return "/api/v1/integration" if identity.run_scope is not None else "/api/v1"


def _json_dependency[T: BaseModel](model: type[T]) -> Callable[[Request], Awaitable[T]]:
    async def parse(request: Request) -> T:
        content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if content_type != "application/json":
            raise HTTPException(415, "Content-Type must be application/json.")
        body = await request.body()
        try:
            # Strict tuples and UUIDs use JSON validation rules. Validating an
            # already-decoded dict would reject their ordinary JSON representation.
            return await run_in_threadpool(model.model_validate_json, body)
        except ValidationError as exc:
            raise RequestValidationError(exc.errors(), body=None) from exc

    return parse


def _request_body(model: type[BaseModel]) -> dict[str, Any]:
    return {
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {"schema": {"$ref": f"#/components/schemas/{model.__name__}"}}
            },
        }
    }


class _API(FastAPI):
    def openapi(self) -> dict[str, Any]:
        if self.openapi_schema is None:
            schema = get_openapi(
                title=self.title,
                version=self.version,
                description=self.description,
                routes=self.routes,
            )
            components = schema.setdefault("components", {}).setdefault("schemas", {})
            request_models: tuple[type[BaseModel], ...] = self.state.request_models
            for model in request_models:
                body_schema = model.model_json_schema(ref_template="#/components/schemas/{model}")
                components.update(body_schema.pop("$defs", {}))
                components[model.__name__] = body_schema
            self.openapi_schema = schema
        return self.openapi_schema


def create_app(
    service: AnalysisAPIService,
    *,
    admin_service: AdminAPIService | None = None,
    benchmark_service: BenchmarkAPIService | None = None,
    auth_service: AuthService | None = None,
) -> FastAPI:
    """Create an app over configured persistence; startup never auto-migrates."""
    app = _API(
        title="Regimex analysis API",
        version="1.0",
        description=(
            "Upload a completed schema-2.1 snapshot, submit a saved-model analysis with an "
            "Idempotency-Key, and poll its job and committed JSON result. Opaque bearer "
            "credentials are provisioned operationally. The standalone page exchanges a "
            "token for a short-lived session. Dedicated integration routes bind a trusted "
            "backend's subject and run assertions to its configured source scope."
        ),
    )
    app.add_middleware(BoundedJSONBody)
    app.state.request_models = (ExportV2, AnalysisRequest)
    bearer = HTTPBearer(auto_error=False)

    @app.exception_handler(ServiceError)
    async def service_error(_request: Request, error: ServiceError) -> JSONResponse:
        headers = {"Cache-Control": "private, no-store"}
        if error.status_code == 401:
            headers["WWW-Authenticate"] = "Bearer"
        return JSONResponse(_error(error.code, error.message), error.status_code, headers=headers)

    @app.exception_handler(DatabaseError)
    async def database_error(_request: Request, _failure: DatabaseError) -> JSONResponse:
        return JSONResponse(
            _error("infrastructure_failure", "The database is temporarily unavailable."),
            503,
            headers={"Cache-Control": "private, no-store"},
        )

    @app.exception_handler(OSError)
    async def storage_error(_request: Request, _failure: OSError) -> JSONResponse:
        return JSONResponse(
            _error("infrastructure_failure", "Committed storage is temporarily unavailable."),
            503,
            headers={"Cache-Control": "private, no-store"},
        )

    @app.exception_handler(ObjectIntegrityError)
    async def integrity_error(_request: Request, _failure: ObjectIntegrityError) -> JSONResponse:
        return JSONResponse(
            _error("incompatible_artifact", "Committed content failed integrity verification."),
            409,
            headers={"Cache-Control": "private, no-store"},
        )

    @app.exception_handler(HTTPException)
    async def http_error(_request: Request, error: HTTPException) -> JSONResponse:
        return JSONResponse(
            _error("invalid_request", str(error.detail)),
            error.status_code,
            headers={"Cache-Control": "private, no-store", **(error.headers or {})},
        )

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_request: Request, error: RequestValidationError) -> JSONResponse:
        # Pydantic's default errors include submitted values. Never echo tokens,
        # complete exports, or unexpected credential fields into a response.
        details: list[JsonValue] = [
            {
                "location": [str(part) for part in item["loc"]],
                "type": item["type"],
                "message": item["msg"],
            }
            for item in error.errors()
        ]
        return JSONResponse(
            {
                "error": {
                    "code": "invalid_data",
                    "message": "Request validation failed.",
                    "details": details,
                }
            },
            422,
            headers={"Cache-Control": "private, no-store"},
        )

    def principal(
        request: Request,
        response: Response,
        credentials: Annotated[HTTPAuthorizationCredentials | None, Security(bearer)],
    ) -> Principal:
        response.headers["Cache-Control"] = "private, no-store"
        if auth_service is not None:
            from regimex.auth_api import request_auth

            return auth_service.resolve(
                request_auth(request),
                delegated=request.url.path.startswith("/api/v1/integration/"),
            )
        if credentials is None or credentials.scheme.lower() != "bearer":
            raise Unauthorized
        token = credentials.credentials
        if len(token) > 1024 or not token or any(character.isspace() for character in token):
            raise Unauthorized
        identity = service.authenticate(token)
        if any(
            name in request.headers
            for name in (
                "x-external-user-id",
                "x-external-owner-id",
                "x-owner-id",
                "x-source-namespace",
                "x-regimex-subject",
                "x-regimex-run-id",
            )
        ):
            raise Forbidden("External owner or source headers are unsupported by this endpoint.")
        return identity

    Identity = Annotated[Principal, Depends(principal)]
    Export = Annotated[ExportV2, Depends(_json_dependency(ExportV2))]
    Intake = Annotated[AnalysisRequest, Depends(_json_dependency(AnalysisRequest))]
    Limit = Annotated[int, Query(ge=1, le=100)]
    Offset = Annotated[int, Query(ge=0, le=1_000_000)]

    @app.get("/health/live", include_in_schema=False)
    def live() -> dict[str, str]:
        return {"status": "alive"}

    @app.get("/health/ready", include_in_schema=False)
    def ready(response: Response) -> dict[str, JsonValue]:
        state = service.readiness()
        database = state.get("database") is True
        response.status_code = 200 if database else 503
        return {"status": "ready" if database else "unavailable", "database": database}

    @app.get("/health/models", include_in_schema=False)
    def model_readiness(response: Response) -> dict[str, JsonValue]:
        state = service.readiness()
        configured = state.get("database") is True and state.get("active_model") is True
        response.status_code = 200 if configured else 503
        return {"deployment_configured": configured, "worker_capability": "not_probed"}

    @app.post("/api/v1/validate", openapi_extra=_request_body(ExportV2))
    def validate_export(identity: Identity, export: Export) -> ValidationResult:
        if identity.role == "integration" and export.source.source_id != identity.source_namespace:
            raise Forbidden("The export source is outside this integration credential's scope.")
        if identity.run_scope is not None and export.source.run_id != identity.run_scope:
            raise Forbidden("The export run is outside this integration request's scope.")
        return ValidationResult(
            schema_version=export.schema_version,
            content_sha256=export.content_sha256,
            source_id=export.source.source_id,
            run_id=export.source.run_id,
            source_snapshot_id=export.source.snapshot_id,
            step_count=export.time_grid.step_count,
            instrument_ids=tuple(item.instrument_id for item in export.instruments),
        )

    @app.post("/api/v1/snapshots", status_code=201, openapi_extra=_request_body(ExportV2))
    def create_snapshot(identity: Identity, export: Export, response: Response) -> SnapshotView:
        record = service.create_snapshot(identity, export)
        response.headers["Location"] = f"{_prefix(identity)}/snapshots/{record.snapshot_id}"
        return _snapshot(record, prefix=_prefix(identity))

    @app.get("/api/v1/snapshots/{snapshot_id}")
    def get_snapshot(snapshot_id: UUID, identity: Identity) -> SnapshotView:
        return _snapshot(service.get_snapshot(identity, snapshot_id), prefix=_prefix(identity))

    @app.get("/api/v1/snapshots/{snapshot_id}/export")
    def download_snapshot(snapshot_id: UUID, identity: Identity, response: Response) -> ExportV2:
        response.headers["Content-Disposition"] = f'attachment; filename="{snapshot_id}.json"'
        return service.read_snapshot(identity, snapshot_id)

    @app.post("/api/v1/analyses", status_code=202, openapi_extra=_request_body(AnalysisRequest))
    def submit_analysis(
        identity: Identity,
        intake: Intake,
        response: Response,
        idempotency_key: Annotated[Identifier, Header(alias="Idempotency-Key")],
    ) -> AnalysisView:
        record = service.submit_analysis(
            identity,
            intake.snapshot_id,
            model_version_id=intake.model_version_id,
            idempotency_key=idempotency_key,
        )
        response.headers["Location"] = f"{_prefix(identity)}/analyses/{record.analysis_id}"
        return _analysis(
            service.get_analysis_summary(identity, record.analysis_id), prefix=_prefix(identity)
        )

    @app.get("/api/v1/analyses")
    def list_analyses(identity: Identity, limit: Limit = 100, offset: Offset = 0) -> AnalysisPage:
        return AnalysisPage(
            items=tuple(
                _analysis(item, prefix=_prefix(identity))
                for item in service.list_analysis_summaries(identity, limit=limit, offset=offset)
            ),
            limit=limit,
            offset=offset,
        )

    @app.get("/api/v1/analyses/{analysis_id}")
    def get_analysis(analysis_id: UUID, identity: Identity) -> AnalysisView:
        return _analysis(
            service.get_analysis_summary(identity, analysis_id), prefix=_prefix(identity)
        )

    @app.get("/api/v1/analyses/{analysis_id}/result")
    def get_result(
        analysis_id: UUID, identity: Identity, response: Response
    ) -> dict[str, JsonValue]:
        response.headers["Content-Disposition"] = f'attachment; filename="{analysis_id}.json"'
        return service.get_result(identity, analysis_id)

    @app.get("/api/v1/analyses/{analysis_id}/report")
    def get_report(analysis_id: UUID, identity: Identity) -> Response:
        return Response(
            service.read_report(identity, analysis_id),
            media_type="text/html",
            headers={
                "Content-Disposition": (
                    f'attachment; filename="regimex-analysis-{analysis_id}.html"'
                ),
                "Cache-Control": "private, no-store",
                "X-Content-Type-Options": "nosniff",
                "Referrer-Policy": "no-referrer",
            },
        )

    @app.get("/api/v1/analyses/{analysis_id}/report/preview", include_in_schema=False)
    def preview_report(analysis_id: UUID, identity: Identity) -> Response:
        # A separate document keeps its own strict embedded CSP. Blob/srcdoc
        # documents inherit the workspace policy, which excludes inline scripts.
        return Response(
            service.read_report(identity, analysis_id),
            media_type="text/html",
            headers={
                "Cache-Control": "private, no-store",
                "X-Content-Type-Options": "nosniff",
                "Referrer-Policy": "no-referrer",
                "Content-Security-Policy": (
                    "sandbox allow-scripts allow-downloads; frame-ancestors 'self'"
                ),
            },
        )

    @app.get("/api/v1/jobs/{operation_id}")
    def get_job(operation_id: UUID, identity: Identity) -> OperationView:
        return _operation(service.get_operation(identity, operation_id), prefix=_prefix(identity))

    @app.post("/api/v1/jobs/{operation_id}/cancel")
    def cancel_job(operation_id: UUID, identity: Identity) -> OperationView:
        return _operation(
            service.cancel_operation(identity, operation_id), prefix=_prefix(identity)
        )

    @app.get("/api/v1/models")
    def list_models(identity: Identity, limit: Limit = 100, offset: Offset = 0) -> ModelPage:
        return ModelPage(
            items=tuple(
                _model(item) for item in service.list_models(identity, limit=limit, offset=offset)
            ),
            limit=limit,
            offset=offset,
        )

    @app.get("/api/v1/models/{model_version_id}")
    def get_model(model_version_id: Identifier, identity: Identity) -> ModelView:
        return _model(service.get_model(identity, model_version_id))

    @app.get("/api/v1/architectures")
    def list_architectures(
        identity: Identity, limit: Limit = 100, offset: Offset = 0
    ) -> ArchitecturePage:
        records = service.list_architectures(identity)
        return ArchitecturePage(items=records[offset : offset + limit], limit=limit, offset=offset)

    @app.get("/api/v1/architectures/{architecture_id}")
    def get_architecture(architecture_id: str, identity: Identity) -> ArchitectureInfo:
        for item in service.list_architectures(identity):
            if item.architecture_id == architecture_id:
                return item
        raise NotFound

    if admin_service is not None:
        from regimex.admin_api import install_admin_routes

        install_admin_routes(app, admin_service, authenticate=principal)
    if benchmark_service is not None:
        from regimex.benchmark_api import install_benchmark_routes

        if admin_service is None:
            raise ValueError("Benchmark routes require the administrative evaluation service")
        install_benchmark_routes(app, benchmark_service, admin_service, authenticate=principal)
    if auth_service is not None:
        from regimex.auth_api import install_session_routes

        install_session_routes(app, auth_service)
        delegated_paths = {
            "/api/v1/validate",
            "/api/v1/snapshots",
            "/api/v1/snapshots/{snapshot_id}",
            "/api/v1/snapshots/{snapshot_id}/export",
            "/api/v1/analyses",
            "/api/v1/analyses/{analysis_id}",
            "/api/v1/analyses/{analysis_id}/result",
            "/api/v1/analyses/{analysis_id}/report",
            "/api/v1/analyses/{analysis_id}/report/preview",
            "/api/v1/jobs/{operation_id}",
            "/api/v1/jobs/{operation_id}/cancel",
        }
        for route in tuple(app.routes):
            if (
                isinstance(route, APIRoute)
                and route.path in delegated_paths
                and route.methods is not None
            ):
                app.add_api_route(
                    "/api/v1/integration" + route.path.removeprefix("/api/v1"),
                    route.endpoint,
                    methods=sorted(route.methods),
                    response_model=route.response_model,
                    status_code=route.status_code,
                    openapi_extra=route.openapi_extra,
                    include_in_schema=route.include_in_schema,
                    tags=["Integration backend"],
                    name="integration_" + route.name,
                )
    from regimex.web import install_web_routes

    install_web_routes(app)
    return app
