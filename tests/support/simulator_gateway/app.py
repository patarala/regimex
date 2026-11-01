"""Deliberate contract simulator and backend gateway, not the real website.

Only immutable pre-existing fixture records are served. No trading, scheduling,
balances or simulator database access is implemented here.
"""

import hashlib
import re
import threading
from dataclasses import dataclass, field
from typing import Annotated
from uuid import UUID

from fastapi import Depends, FastAPI, Query, Request
from fastapi.responses import JSONResponse, Response
from pydantic import Field, JsonValue, ValidationError
from starlette.concurrency import run_in_threadpool

from regimex.application.service import AnalysisResult
from regimex.contracts import MAX_INPUT_BYTES, StrictModel
from regimex.exports import ExportV2, parse_export_v2

from .client import GatewayError, GatewaySettings, RegimexClient
from .jwt import IDENTITY_PATTERN, InvalidToken, SimulatorJWT, json_bytes, strict_json

Identity = Annotated[str, Field(pattern=f"^{IDENTITY_PATTERN}$")]


@dataclass(frozen=True)
class FixtureRun:
    simulation_id: str
    run_id: str
    owner: str
    results: bytes = field(repr=False)
    export: bytes | None = field(default=None, repr=False)

    @classmethod
    def completed(
        cls, simulation_id: str, owner: str, export: ExportV2, *, results: JsonValue
    ) -> FixtureRun:
        return cls(
            simulation_id,
            export.source.run_id,
            owner,
            json_bytes(results),
            json_bytes(export.model_dump(mode="json")),
        )


class FixtureCatalog:
    """Atomic immutable fixture storage; the real site must supply its own adapter."""

    def __init__(self, source_namespace: str) -> None:
        self.source_namespace = source_namespace
        self._runs: dict[tuple[str, str], FixtureRun] = {}
        self._lock = threading.RLock()

    def register(self, run: FixtureRun) -> None:
        if any(
            not re.fullmatch(IDENTITY_PATTERN, value)
            for value in (run.simulation_id, run.run_id, run.owner)
        ):
            raise ValueError("Invalid fixture identity.")
        strict_json(run.results)
        if len(run.results) > MAX_INPUT_BYTES:
            raise ValueError("Fixture results exceed their size limit.")
        if run.export is not None:
            export = parse_export_v2(run.export)
            if (
                export.source.source_id != self.source_namespace
                or export.source.run_id != run.run_id
                or export.source.configuration_id != run.simulation_id
            ):
                raise ValueError("Fixture export identity does not match the run.")
        with self._lock:
            key = (run.simulation_id, run.run_id)
            if key in self._runs and self._runs[key] != run:
                raise ValueError("An immutable fixture run cannot be overwritten.")
            self._runs[key] = run

    def owned(self, simulation_id: str, run_id: str, subject: str) -> FixtureRun:
        with self._lock:
            run = self._runs.get((simulation_id, run_id))
            if run is None or run.owner != subject:
                raise GatewayError(
                    "not_found", "Simulator run was not found.", 404, retryable=False
                )
            return run


class AnalysisRequest(StrictModel):
    run_id: Identity


def _completed(run: FixtureRun) -> ExportV2:
    if run.export is None:
        raise GatewayError(
            "run_incomplete",
            "Classification requires a completed immutable run.",
            409,
            retryable=False,
        )
    return parse_export_v2(run.export)


def _unverified() -> GatewayError:
    return GatewayError(
        "invalid_classification_response",
        "The classification response could not be verified.",
        502,
        retryable=True,
    )


def idempotency_key(subject: str, export: ExportV2) -> str:
    identity = {
        "gateway_operation": "saved-model-analysis/1.0",
        "subject": subject,
        "source_id": export.source.source_id,
        "run_id": export.source.run_id,
        "snapshot_sha256": export.content_sha256,
    }
    return "gateway-" + hashlib.sha256(json_bytes(identity)).hexdigest()


def _snapshot_matches(snapshot: dict[str, object], export: ExportV2) -> UUID:
    try:
        snapshot_id = UUID(str(snapshot["snapshot_id"]))
    except KeyError, ValueError, TypeError:
        raise _unverified() from None
    if any(
        snapshot.get(key) != value
        for key, value in {
            "source_id": export.source.source_id,
            "run_id": export.source.run_id,
            "source_snapshot_id": export.source.snapshot_id,
            "content_sha256": export.content_sha256,
        }.items()
    ):
        raise _unverified()
    return snapshot_id


def _analysis_ids(value: dict[str, object], *, expected: UUID | None = None) -> tuple[UUID, UUID]:
    try:
        analysis_id, snapshot_id = UUID(str(value["analysis_id"])), UUID(str(value["snapshot_id"]))
        if expected is not None and analysis_id != expected:
            raise ValueError
        job = value["job"]
        if not isinstance(job, dict) or job.get("resource_id") != str(analysis_id):
            raise ValueError
        UUID(str(job["operation_id"]))
        if job.get("kind") != "analysis" or job.get("state") not in {
            "queued",
            "running",
            "cancel_requested",
            "succeeded",
            "failed",
            "cancelled",
        }:
            raise ValueError
    except KeyError, ValueError, TypeError:
        raise _unverified() from None
    return analysis_id, snapshot_id


def _view(value: dict[str, object], simulation_id: str, run_id: str) -> dict[str, object]:
    analysis_id, _ = _analysis_ids(value)
    # Return only the deliberate public fields and gateway-local links. Internal
    # upstream URLs are never used as destinations or copied to the browser.
    job = value["job"]
    assert isinstance(job, dict)
    base = f"/api/simulations/{simulation_id}/regime-analyses/{analysis_id}"
    query = f"?run_id={run_id}"
    return {
        "analysis_id": str(analysis_id),
        "run_id": run_id,
        "state": job["state"],
        "stage": job.get("stage"),
        "failure": job.get("failure"),
        "result_available": value.get("result_available") is True,
        "report_available": value.get("report_available") is True,
        "polling_url": base + query,
        "result_url": base + "/result" + query,
        "report_url": base + "/report" + query,
    }


def create_gateway(
    settings: GatewaySettings,
    jwt: SimulatorJWT,
    catalog: FixtureCatalog,
) -> FastAPI:
    if settings.source_namespace != catalog.source_namespace:
        raise ValueError("Gateway and fixture source namespaces must match.")
    client = RegimexClient(settings)
    app = FastAPI(title="Regimex contract simulator gateway", docs_url="/docs")

    @app.exception_handler(GatewayError)
    async def gateway_error(request: Request, error: GatewayError) -> JSONResponse:
        headers = {"Cache-Control": "no-store"}
        if error.status == 401:
            headers["WWW-Authenticate"] = "Bearer"
        return JSONResponse(
            {"error": {"code": error.code, "message": error.message, "retryable": error.retryable}},
            error.status,
            headers=headers,
        )

    def authenticate(request: Request) -> str:
        forbidden = {
            "x-regimex-subject",
            "x-regimex-run-id",
            "x-external-user-id",
            "x-external-owner-id",
            "x-owner-id",
            "x-source-namespace",
        }
        if forbidden.intersection(request.headers):
            raise GatewayError(
                "untrusted_identity",
                "The browser cannot assert a trusted identity.",
                400,
                retryable=False,
            )
        values = request.headers.getlist("authorization")
        try:
            if len(values) != 1 or not values[0].startswith("Bearer "):
                raise InvalidToken
            return jwt.verify(values[0][7:])
        except InvalidToken:
            raise GatewayError(
                "unauthorized", "Valid simulator authentication is required.", 401, retryable=False
            ) from None

    def owned_export(simulation_id: str, run_id: str, subject: str) -> ExportV2:
        return _completed(catalog.owned(simulation_id, run_id, subject))

    def bound_analysis(
        simulation_id: str,
        run_id: str,
        analysis_id: UUID,
        subject: str,
    ) -> tuple[dict[str, object], ExportV2, UUID]:
        export = owned_export(simulation_id, run_id, subject)
        analysis = client.json(
            "GET",
            f"/api/v1/integration/analyses/{analysis_id}",
            subject=subject,
            run_id=run_id,
        )
        _, snapshot_id = _analysis_ids(analysis, expected=analysis_id)
        snapshot = client.json(
            "GET",
            f"/api/v1/integration/snapshots/{snapshot_id}",
            subject=subject,
            run_id=run_id,
        )
        if _snapshot_matches(snapshot, export) != snapshot_id:
            raise _unverified()
        return analysis, export, snapshot_id

    @app.get("/api/simulations/{simulation_id}/results")
    def results(
        simulation_id: Identity,
        run_id: Annotated[Identity, Query()],
        subject: Annotated[str, Depends(authenticate)],
    ) -> Response:
        run = catalog.owned(simulation_id, run_id, subject)
        return Response(
            run.results, media_type="application/json", headers={"Cache-Control": "no-store"}
        )

    @app.get("/api/simulations/{simulation_id}/export")
    def export_run(
        simulation_id: Identity,
        run_id: Annotated[Identity, Query()],
        subject: Annotated[str, Depends(authenticate)],
        schema_version: Annotated[str, Query(pattern=r"^2\.1$")] = "2.1",
    ) -> Response:
        export = owned_export(simulation_id, run_id, subject)
        return Response(
            json_bytes(export.model_dump(mode="json")),
            media_type="application/json",
            headers={
                "Content-Disposition": f'attachment; filename="{run_id}-export-v2.json"',
                "Cache-Control": "no-store",
            },
        )

    @app.post("/api/simulations/{simulation_id}/regime-analyses", status_code=202)
    async def submit(
        simulation_id: Identity,
        request: Request,
        subject: Annotated[str, Depends(authenticate)],
    ) -> JSONResponse:
        data = bytearray()
        if request.headers.get("content-type", "").split(";", 1)[0].lower() != "application/json":
            raise GatewayError(
                "invalid_request", "Use an application/json body.", 415, retryable=False
            )
        if request.headers.get("content-encoding", "identity") != "identity":
            raise GatewayError(
                "invalid_request", "Encoded request bodies are unsupported.", 415, retryable=False
            )
        async for block in request.stream():
            if len(data) + len(block) > 4096:
                raise GatewayError(
                    "invalid_request", "Gateway request is too large.", 413, retryable=False
                )
            data.extend(block)
        try:
            body = AnalysisRequest.model_validate(strict_json(bytes(data)))
        except ValidationError, ValueError, UnicodeError, RecursionError:
            raise GatewayError(
                "invalid_request", "A valid run_id is required.", 422, retryable=False
            ) from None
        export = owned_export(simulation_id, body.run_id, subject)
        return await run_in_threadpool(submit_export, simulation_id, body.run_id, subject, export)

    def submit_export(
        simulation_id: str,
        run_id: str,
        subject: str,
        export: ExportV2,
    ) -> JSONResponse:
        snapshot = client.json(
            "POST",
            "/api/v1/integration/snapshots",
            subject=subject,
            run_id=run_id,
            payload=export.model_dump(mode="json"),
        )
        snapshot_id = _snapshot_matches(snapshot, export)
        analysis = client.json(
            "POST",
            "/api/v1/integration/analyses",
            subject=subject,
            run_id=run_id,
            payload={"snapshot_id": str(snapshot_id), "model_version_id": "active"},
            idempotency_key=idempotency_key(subject, export),
        )
        if _analysis_ids(analysis)[1] != snapshot_id:
            raise _unverified()
        view = _view(analysis, simulation_id, run_id)
        return JSONResponse(
            view, 202, headers={"Location": str(view["polling_url"]), "Cache-Control": "no-store"}
        )

    @app.get("/api/simulations/{simulation_id}/regime-analyses/{analysis_id}")
    def status(
        simulation_id: Identity,
        analysis_id: UUID,
        run_id: Annotated[Identity, Query()],
        subject: Annotated[str, Depends(authenticate)],
    ) -> JSONResponse:
        analysis, _, _ = bound_analysis(simulation_id, run_id, analysis_id, subject)
        return JSONResponse(
            _view(analysis, simulation_id, run_id), headers={"Cache-Control": "no-store"}
        )

    @app.get("/api/simulations/{simulation_id}/regime-analyses/{analysis_id}/result")
    def result(
        simulation_id: Identity,
        analysis_id: UUID,
        run_id: Annotated[Identity, Query()],
        subject: Annotated[str, Depends(authenticate)],
    ) -> Response:
        _, export, snapshot_id = bound_analysis(simulation_id, run_id, analysis_id, subject)
        value = client.json(
            "GET",
            f"/api/v1/integration/analyses/{analysis_id}/result",
            subject=subject,
            run_id=run_id,
        )
        try:
            parsed = AnalysisResult.model_validate_json(json_bytes(value))
        except ValidationError:
            raise _unverified() from None
        if (
            parsed.analysis_id != analysis_id
            or parsed.snapshot_id != snapshot_id
            or parsed.snapshot_sha256 != export.content_sha256
            or parsed.source != export.source
            or parsed.time_grid != export.time_grid
            or parsed.instruments != export.instruments
        ):
            raise _unverified()
        return Response(
            json_bytes(value), media_type="application/json", headers={"Cache-Control": "no-store"}
        )

    @app.get("/api/simulations/{simulation_id}/regime-analyses/{analysis_id}/report")
    def report(
        simulation_id: Identity,
        analysis_id: UUID,
        run_id: Annotated[Identity, Query()],
        subject: Annotated[str, Depends(authenticate)],
    ) -> Response:
        bound_analysis(simulation_id, run_id, analysis_id, subject)
        response = client.request(
            "GET",
            f"/api/v1/integration/analyses/{analysis_id}/report",
            subject=subject,
            run_id=run_id,
        )
        try:
            response.body.decode("utf-8")
            if response.content_type.split(";", 1)[0].strip().lower() != "text/html":
                raise ValueError
        except ValueError, UnicodeError:
            raise _unverified() from None
        return Response(
            response.body,
            media_type="text/html; charset=utf-8",
            headers={
                "Content-Disposition": (
                    f'attachment; filename="regimex-analysis-{analysis_id}.html"'
                ),
                "Cache-Control": "no-store",
                "Content-Security-Policy": "sandbox",
                "X-Content-Type-Options": "nosniff",
            },
        )

    return app
