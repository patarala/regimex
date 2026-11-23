"""Trusted subject delegation and owner/source/run boundaries over actual PostgreSQL."""

import hashlib
import json
import os
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from uuid import UUID, uuid4
from zipfile import ZIP_STORED, ZipFile

import pytest

pytest.importorskip("psycopg", reason="install the service extra")
pytest.importorskip("procrastinate", reason="install the service extra")
pytest.importorskip("httpx2", reason="install the service-test group")

import psycopg
from fastapi.testclient import TestClient
from pydantic import ValidationError
from test_api import analysis_result_fixture
from test_service_database import _model

from regimex.api import create_app
from regimex.application.auth import AuthService
from regimex.application.service import AnalysisService
from regimex.auth_contracts import AuthPolicy
from regimex.exports import ExportPayloadV2, ExportV2, seal_export
from regimex.infrastructure.auth_repository import AuthRepository
from regimex.infrastructure.database import DatabaseRepository
from regimex.infrastructure.object_store import LocalObjectStore
from regimex.model_registry import ArchitectureRegistry, baseline_descriptor
from regimex.report_contracts import AnalysisReportRecord
from regimex.service_contracts import CapacityExceeded, Conflict, Forbidden, NotFound, Principal
from regimex.storage import json_bytes

ORIGIN = "https://regimex.example"
PREFIX = "/api/v1/integration"


@dataclass
class IntegrationContext:
    database: DatabaseRepository
    authentication: AuthService
    service: AnalysisService
    client: TestClient
    gateway_token: str
    user_token: str

    def headers(
        self, subject: str = "alice", run: str = "run-1", *, token: str | None = None
    ) -> dict[str, str]:
        return {
            "Authorization": "Bearer " + (token or self.gateway_token),
            "X-Regimex-Subject": subject,
            "X-Regimex-Run-ID": run,
        }

    def identity(self, subject: str = "alice", run: str = "run-1") -> Principal:
        return self.authentication.repository.bind_subject(self.gateway_token, subject, run)


@pytest.fixture
def context(tmp_path: Path) -> Iterator[IntegrationContext]:
    dsn = os.environ.get("REGIMEX_TEST_DATABASE_DSN")
    if not dsn:
        pytest.skip("set REGIMEX_TEST_DATABASE_DSN to a disposable PostgreSQL database")
    database = DatabaseRepository(dsn, schema="regimex_scope_" + uuid4().hex)
    try:
        database.migrate()
        gateway_token = database.create_token(
            "simulator-account", "integration", source_namespace="test-simulator"
        )
        user_token = database.create_token("ordinary-user", "user")
        authentication = AuthService(AuthRepository(database), AuthPolicy(public_origin=ORIGIN))
        service = AnalysisService(
            database,
            LocalObjectStore(
                tmp_path / "objects", max_object_bytes=1024**2, max_total_bytes=16 * 1024**2
            ),
            ArchitectureRegistry((baseline_descriptor(),)),
            tmp_path / "work",
        )
        # A deliberately non-executable model record tests authorization and SQL
        # queue transitions. Actual fitting and report rendering have other tests.
        model = _model(database)
        admin = database.authenticate(database.create_token("administrator", "admin"))
        database.activate_model(admin, model.model_version_id, expected_previous=None)
        with (
            database.queue_app.open(),
            TestClient(create_app(service, auth_service=authentication), base_url=ORIGIN) as client,
        ):
            yield IntegrationContext(
                database, authentication, service, client, gateway_token, user_token
            )
    finally:
        with psycopg.connect(dsn, autocommit=True) as connection:
            connection.execute(
                psycopg.sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                    psycopg.sql.Identifier(database.schema)
                )
            )


def export(run: str = "run-1", source: str = "test-simulator") -> ExportV2:
    value = json.loads((Path(__file__).parent / "fixtures" / "experiment-v2.json").read_bytes())
    value.pop("content_sha256")
    value["source"]["run_id"] = run
    value["source"]["source_id"] = source
    return seal_export(ExportPayloadV2.model_validate_json(json.dumps(value)))


def test_single_request_intake_respects_the_subject_rate_limit(context: IntegrationContext) -> None:
    context.authentication.integration_limits.subject_requests_per_minute = 1
    payload = export()
    headers = {
        **context.headers(),
        "Idempotency-Key": "single-call",
        "Content-Type": "application/json",
    }
    reply = context.client.post(
        PREFIX + "/analysis-intake", content=payload.model_dump_json(), headers=headers
    )
    assert reply.status_code == 202, reply.text
    snapshot, analysis = reply.json()["snapshot"], reply.json()["analysis"]
    assert snapshot["snapshot_id"] == analysis["snapshot_id"]
    assert snapshot["content_sha256"] == analysis["snapshot_sha256"] == payload.content_sha256
    assert reply.headers["Location"] == analysis["polling_url"]
    limited = context.client.post(
        PREFIX + "/analysis-intake", content=payload.model_dump_json(), headers=headers
    )
    assert limited.status_code == 429 and limited.json()["error"]["code"] == "rate_limit"


def test_analysis_intake_replays_and_checks_historical_retry(context: IntegrationContext) -> None:
    payload = export().model_dump(mode="json")
    url = PREFIX + "/analysis-intake"
    headers = {**context.headers(), "Idempotency-Key": "first", "X-Regimex-Command": "analyze"}
    first = context.client.post(url, json=payload, headers=headers).json()["analysis"]
    assert first["job"]["launch_source"] == {"channel": "integration", "command": None}
    assert context.client.post(url, json=payload, headers=headers).json()["analysis"] == first
    params = {"retry_of": first["analysis_id"]}
    headers["Idempotency-Key"] = "retry"
    refused = context.client.post(url, params=params, json=payload, headers=headers)
    assert refused.status_code == 409 and refused.json()["error"]["code"] == "retry_not_available"
    context.database.cancel_operation(context.identity(), UUID(first["analysis_id"]))
    retry = context.client.post(url, params=params, json=payload, headers=headers)
    assert retry.status_code == 202, retry.text
    assert retry.json()["analysis"]["analysis_id"] != first["analysis_id"]
    assert (
        context.client.post(url, params=params, json=payload, headers=headers).json()
        == retry.json()
    )


def test_combined_intake_recovers_an_accepted_two_request_retry(
    context: IntegrationContext,
) -> None:
    snapshot = upload(context)
    first = submit(context, snapshot)
    context.database.cancel_operation(context.identity(), first)
    headers = {**context.headers(), "Idempotency-Key": "same-retry"}
    alternate = context.service.submit_analysis(
        context.identity(), snapshot, idempotency_key="alternate"
    )
    context.database.cancel_operation(context.identity(), alternate.operation.operation_id)
    stored_digest = hashlib.sha256(
        json_bytes({"snapshot_id": str(snapshot), "model_version_id": "active"})
    ).hexdigest()
    accepted = context.database.submit_analysis(
        context.identity(), snapshot, idempotency_key="same-retry", request_sha256=stored_digest
    )
    replay = context.client.post(
        PREFIX + "/analysis-intake",
        params={"retry_of": str(first)},
        json=export().model_dump(mode="json"),
        headers=headers,
    )
    assert replay.status_code == 202
    assert replay.json()["analysis"]["analysis_id"] == str(accepted.analysis_id)
    switched = context.client.post(
        PREFIX + "/analysis-intake",
        params={"retry_of": str(alternate.analysis_id)},
        json=export().model_dump(mode="json"),
        headers=headers,
    )
    assert switched.status_code == 409 and switched.json()["error"]["code"] == "conflict"
    assert (
        context.database.get_operation(
            context.identity(), accepted.operation.operation_id
        ).request_sha256
        == stored_digest
    )
    different_run = context.client.post(
        PREFIX + "/analysis-intake",
        params={"retry_of": str(first)},
        json=export("run-2").model_dump(mode="json"),
        headers={**context.headers(run="run-2"), "Idempotency-Key": "different-run"},
    )
    assert different_run.status_code == 404


def test_intake_distinguishes_storage_and_model_configuration(context: IntegrationContext) -> None:
    url = PREFIX + "/analysis-intake"
    payload = export().model_dump(mode="json")
    headers = {**context.headers(), "Idempotency-Key": "single-call"}
    context.service.storage_quotas.max_owner_storage_bytes = 1
    quota = context.client.post(url, json=payload, headers=headers)
    assert quota.status_code == 429 and quota.json()["error"]["code"] == "storage_quota_exceeded"
    assert "Retry-After" not in quota.headers
    context.service.storage_quotas.max_owner_storage_bytes = 1024**2
    with context.database.connection() as connection:
        connection.execute("DELETE FROM regimex_deployment")
    model = context.client.post(url, json=payload, headers=headers)
    assert model.status_code == 409 and model.json()["error"]["code"] == "model_unavailable"


def upload(context: IntegrationContext, run: str = "run-1") -> UUID:
    response = context.client.post(
        PREFIX + "/snapshots",
        json=export(run).model_dump(mode="json"),
        headers=context.headers(run=run),
    )
    assert response.status_code == 201, response.text
    assert response.json()["export_url"].startswith(PREFIX)
    return UUID(response.json()["snapshot_id"])


def submit(context: IntegrationContext, snapshot_id: UUID, run: str = "run-1") -> UUID:
    response = context.client.post(
        PREFIX + "/analyses",
        json={"snapshot_id": str(snapshot_id)},
        headers={**context.headers(run=run), "Idempotency-Key": "analysis-" + run},
    )
    assert response.status_code == 202, response.text
    assert response.json()["job"]["polling_url"].startswith(PREFIX)
    return UUID(response.json()["analysis_id"])


def test_delegation_mapping_is_stable_private_and_partitioned_by_integration_source(
    context: IntegrationContext,
) -> None:
    first = context.identity()
    rotated = context.database.create_token(
        "simulator-account", "integration", source_namespace="test-simulator"
    )
    same = context.authentication.repository.bind_subject(rotated, "alice", "run-2")
    assert first.owner_id == same.owner_id and same.run_scope == "run-2"
    assert first.owner_id != context.identity("bob").owner_id
    for account, source in (
        ("simulator-account", "another-source"),
        ("another-account", "test-simulator"),
    ):
        token = context.database.create_token(account, "integration", source_namespace=source)
        different = context.authentication.repository.bind_subject(token, "alice", "run-1")
        assert different.owner_id != first.owner_id
    assert first.role == same.role == "integration"
    with context.database.connection() as connection:
        rows = connection.execute("SELECT row_to_json(s)::text FROM regimex_integration_subjects s")
        assert all('"alice"' not in str(row[0]) for row in rows)
        with pytest.raises(psycopg.Error, match="immutable"):
            connection.execute(
                "UPDATE regimex_integration_subjects SET subject_sha256=%s", ("f" * 64,)
            )


def test_subject_bound_is_atomic_and_existing_subject_remains_usable(
    context: IntegrationContext,
) -> None:
    def bind(index: int) -> bool:
        try:
            context.authentication.repository.bind_subject(
                context.gateway_token, "subject-" + str(index), "run-1", maximum_subjects=1
            )
        except CapacityExceeded:
            return False
        return True

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert sum(pool.map(bind, range(4))) == 1
    with context.database.connection() as connection:
        row = connection.execute(
            "SELECT subject_sha256 FROM regimex_integration_subjects"
        ).fetchall()
        assert len(row) == 1
    # Re-authentication of a known subject does not consume another identity slot.
    winner = next(index for index in range(4) if bind(index))
    context.authentication.repository.bind_subject(
        context.gateway_token, "subject-" + str(winner), "run-2", maximum_subjects=1
    )


def test_browser_user_and_unbound_account_cannot_assert_delegated_ownership(
    context: IntegrationContext,
) -> None:
    target = PREFIX + "/analyses"
    user_headers = context.headers(token=context.user_token)
    assert context.client.get(target, headers=user_headers).status_code == 403
    for extra in ({"Origin": ORIGIN}, {"Sec-Fetch-Site": "same-origin"}):
        assert context.client.get(target, headers={**context.headers(), **extra}).status_code == 403
    assert context.client.get("/api/v1/analyses", headers=context.headers()).status_code == 403
    assert (
        context.client.get(
            target, headers={"Authorization": "Bearer " + context.gateway_token}
        ).status_code
        == 403
    )
    for subject in ("", "has spaces", "contains/slash", "a" * 129):
        assert context.client.get(target, headers=context.headers(subject)).status_code == 422
    login = context.client.post(
        "/api/v1/session", json={"token": context.user_token}, headers={"Origin": ORIGIN}
    )
    assert login.status_code == 200
    assert (
        context.client.get(
            target, headers={"X-Regimex-Subject": "alice", "X-Regimex-Run-ID": "run-1"}
        ).status_code
        == 403
    )


def test_delegated_upload_validation_and_reads_are_bound_before_object_access(
    context: IntegrationContext,
) -> None:
    for wrong in (export("run-2"), export(source="another-simulator")):
        for route in ("/validate", "/snapshots"):
            response = context.client.post(
                PREFIX + route, json=wrong.model_dump(mode="json"), headers=context.headers()
            )
            assert response.status_code == 403, response.text
    assert not context.service.store.inventory()
    first = upload(context)
    second = upload(context, "run-2")
    for suffix in ("", "/export"):
        target = PREFIX + f"/snapshots/{first}" + suffix
        assert context.client.get(target, headers=context.headers()).status_code == 200
        for headers in (context.headers(run="run-2"), context.headers("bob")):
            assert context.client.get(target, headers=headers).status_code == 404
        assert (
            context.client.get(
                target, headers={"Authorization": "Bearer " + context.gateway_token}
            ).status_code
            == 403
        )
        # The unbound service account sees only its own resources via ordinary routes.
        assert (
            context.client.get(
                target.replace(PREFIX, "/api/v1"),
                headers={"Authorization": "Bearer " + context.gateway_token},
            ).status_code
            == 404
        )
    assert [item.snapshot_id for item in context.database.list_snapshots(context.identity())] == [
        first
    ]
    assert [
        item.snapshot_id for item in context.database.list_snapshots(context.identity(run="run-2"))
    ] == [second]
    # An earlier server-owned snapshot with the same effective owner and a
    # different source must remain hidden from the integration credential.
    with pytest.raises(Conflict, match="reserved"):
        context.database.create_token(context.identity().owner_id, "user")
    # Insert a deliberate stored-data fixture without minting a conflicting credential.
    local = Principal(owner_id=context.identity().owner_id, role="user", token_id=uuid4())
    foreign = context.service.create_snapshot(local, export(source="another-simulator"))
    with pytest.raises(NotFound):
        context.database.get_snapshot(context.identity(), foreign.snapshot_id)
    assert len(context.database.list_snapshots(context.identity())) == 1


def test_every_analysis_job_result_and_report_route_hides_other_run_or_subject(
    context: IntegrationContext,
) -> None:
    first_snapshot = upload(context)
    first = submit(context, first_snapshot)
    second = submit(context, upload(context, "run-2"), "run-2")
    identity = context.identity()
    assert [item.analysis_id for item in context.database.list_analyses(identity)] == [first]
    assert [
        item.analysis_id for item in context.database.list_analysis_summaries(identity, limit=1)
    ] == [first]
    assert not context.database.list_analysis_summaries(identity, limit=1, offset=1)
    for headers in (context.headers(run="run-2"), context.headers("bob")):
        for route in (
            f"/analyses/{first}",
            f"/analyses/{first}/result",
            f"/analyses/{first}/report",
            f"/analyses/{first}/delivery?include_report=true",
            f"/jobs/{first}",
        ):
            assert context.client.get(PREFIX + route, headers=headers).status_code == 404
        assert (
            context.client.post(PREFIX + f"/jobs/{first}/cancel", headers=headers).status_code
            == 404
        )
        assert (
            context.client.post(
                PREFIX + "/analyses",
                json={"snapshot_id": str(first_snapshot)},
                headers={**headers, "Idempotency-Key": "analysis-run-1"},
            ).status_code
            == 404
        )
    for run, expected in (("run-1", first), ("run-2", second)):
        response = context.client.get(
            PREFIX + "/analyses?limit=1", headers=context.headers(run=run)
        )
        assert [item["analysis_id"] for item in response.json()["items"]] == [str(expected)]
    # Source filtering must precede pagination even for an older resource that
    # happens to share this delegated effective owner and the same run ID.
    with pytest.raises(Conflict, match="reserved"):
        context.database.create_token(identity.owner_id, "user")
    local = Principal(owner_id=identity.owner_id, role="user", token_id=uuid4())
    foreign_snapshot = context.service.create_snapshot(local, export(source="foreign-source"))
    foreign = context.service.submit_analysis(
        local, foreign_snapshot.snapshot_id, idempotency_key="foreign-source-analysis"
    )
    assert [item.analysis_id for item in context.database.list_analyses(identity)] == [first]
    assert [
        item.analysis_id for item in context.database.list_analysis_summaries(identity, limit=1)
    ] == [first]
    for route in (
        f"/analyses/{foreign.analysis_id}",
        f"/analyses/{foreign.analysis_id}/result",
        f"/analyses/{foreign.analysis_id}/report",
        f"/analyses/{foreign.analysis_id}/delivery",
        f"/jobs/{foreign.analysis_id}",
    ):
        assert context.client.get(PREFIX + route, headers=context.headers()).status_code == 404
    assert context.database.get_operation(identity, first).state == "queued"
    assert (
        context.client.post(PREFIX + f"/jobs/{first}/cancel", headers=context.headers()).status_code
        == 200
    )


def test_delivery_rejects_pending_analysis_before_creating_a_response(
    context: IntegrationContext,
) -> None:
    queued = submit(context, upload(context))
    response = context.client.get(
        PREFIX + f"/analyses/{queued}/delivery",
        params={"include_report": True},
        headers=context.headers(),
    )
    assert response.status_code == 409 and response.json()["error"]["code"] == "result_pending"


@pytest.mark.parametrize("include_report", [False, True])
def test_delivery_binds_saved_documents_in_one_request(
    context: IntegrationContext,
    tmp_path: Path,
    include_report: bool,
) -> None:
    identity = context.identity()
    observations = export()
    snapshot = context.service.create_snapshot(identity, observations)
    analysis = context.service.submit_analysis(
        identity, snapshot.snapshot_id, idempotency_key="delivery"
    )
    operation = context.database.claim_operation(analysis.analysis_id)
    assert (
        operation is not None
        and operation.attempt_token is not None
        and operation.model is not None
    )
    result = analysis_result_fixture(observations, snapshot.snapshot_id, operation).model_dump(
        mode="json"
    )
    # The committed numeric spelling survives validation and delivery.
    result["sequences"][0]["probabilities"][0] = [1, 0, 0]
    result_hash = hashlib.sha256(json_bytes(result)).hexdigest()
    html = b"<!doctype html><title>Bound report</title>"
    html_path = tmp_path / "report.html"
    html_path.write_bytes(html)
    stored = context.service.store.put(
        "reports/bound.html", html_path, expected_sha256=hashlib.sha256(html).hexdigest()
    )
    context.database.complete_analysis(
        analysis.analysis_id,
        operation.attempt_token,
        result,
        report=AnalysisReportRecord(
            analysis_id=analysis.analysis_id,
            owner_id=identity.owner_id,
            snapshot_id=snapshot.snapshot_id,
            snapshot_sha256=observations.content_sha256,
            source=observations.source,
            model=operation.model,
            result_sha256=result_hash,
            plotly_version="fixture",
            object=stored,
            created_at=datetime.now(UTC),
        ),
    )
    context.authentication.integration_limits.subject_requests_per_minute = 1
    url = PREFIX + f"/analyses/{analysis.analysis_id}/delivery"
    response = context.client.get(
        url, params={"include_report": True} if include_report else {}, headers=context.headers()
    )
    assert response.status_code == 200, response.text[:256]
    assert response.headers["Content-Type"] == "application/zip"
    assert response.headers["Cache-Control"] == "private, no-store"
    with ZipFile(BytesIO(response.content)) as archive:
        names = ["analysis.json", "result.json"] + (["report.html"] if include_report else [])
        assert archive.namelist() == names
        assert all(member.compress_type == ZIP_STORED for member in archive.infolist())
        detail = json.loads(archive.read("analysis.json"))
        assert detail["analysis_id"] == str(analysis.analysis_id)
        assert detail["snapshot_id"] == str(snapshot.snapshot_id)
        assert detail["snapshot_sha256"] == observations.content_sha256
        assert detail["job"]["state"] == "succeeded" and detail["polling_url"].startswith(PREFIX)
        assert archive.read("result.json") == json_bytes(result)
        if include_report:
            assert archive.read("report.html") == html
    limited = context.client.get(url, headers=context.headers())
    assert limited.status_code == 429 and limited.json()["error"]["code"] == "rate_limit"


def test_delivery_distinguishes_successful_analysis_without_html(
    context: IntegrationContext,
) -> None:
    from psycopg.types.json import Json

    from regimex.operations import succeed

    snapshot_id = upload(context)
    analysis_id = submit(context, snapshot_id)
    operation = context.database.claim_operation(analysis_id)
    assert operation is not None and operation.attempt_token is not None
    result = analysis_result_fixture(export(), snapshot_id, operation).model_dump(mode="json")
    result["sequences"][0]["probabilities"][0] = [1, 0, 0]
    # A successful analysis created before report publication has only its JSON.
    with context.database.connection() as connection:
        context.database._save(
            connection,
            succeed(
                operation,
                operation.attempt_token,
                hashlib.sha256(json_bytes(result)).hexdigest(),
                datetime.now(UTC),
            ),
        )
        connection.execute(
            "UPDATE regimex_analyses SET result=%s WHERE analysis_id=%s",
            (Json(result), analysis_id),
        )
    url = PREFIX + f"/analyses/{analysis_id}/delivery"
    response = context.client.get(url, params={"include_report": True}, headers=context.headers())
    assert response.status_code == 404 and response.json()["error"]["code"] == "report_unavailable"
    response = context.client.get(url, headers=context.headers())
    assert response.status_code == 200
    with ZipFile(BytesIO(response.content)) as archive:
        direct = context.client.get(
            PREFIX + f"/analyses/{analysis_id}/result", headers=context.headers()
        )
        assert direct.status_code == 200
        assert direct.content == archive.read("result.json") == json_bytes(result)


def test_cookie_authentication_guards_real_api_upload_submit_and_cancel(
    context: IntegrationContext,
) -> None:
    login = context.client.post(
        "/api/v1/session", json={"token": context.user_token}, headers={"Origin": ORIGIN}
    )
    csrf = {"Origin": ORIGIN, "X-CSRF-Token": login.json()["csrf_token"]}
    value = export().model_dump(mode="json")
    assert (
        context.client.post("/api/v1/snapshots", json=value, headers={"Origin": ORIGIN}).status_code
        == 403
    )
    assert not context.service.store.inventory()
    snapshot = context.client.post("/api/v1/snapshots", json=value, headers=csrf)
    assert snapshot.status_code == 201
    intake = {"snapshot_id": snapshot.json()["snapshot_id"]}
    assert (
        context.client.post(
            "/api/v1/analyses", json=intake, headers={"Idempotency-Key": "browser"}
        ).status_code
        == 403
    )
    queued = context.client.post(
        "/api/v1/analyses",
        json=intake,
        headers={**csrf, "Idempotency-Key": "browser", "X-Regimex-Command": "analyze"},
    )
    assert queued.status_code == 202
    assert queued.json()["job"]["launch_source"] == {"channel": "web", "command": None}
    job = queued.json()["job"]["operation_id"]
    assert context.client.get(f"/api/v1/jobs/{job}").status_code == 200
    assert context.client.post(f"/api/v1/jobs/{job}/cancel").status_code == 403
    assert context.client.post(f"/api/v1/jobs/{job}/cancel", headers=csrf).status_code == 200
    schema = context.client.get("/openapi.json").json()
    exchange = schema["paths"]["/api/v1/session"]["post"]["requestBody"]
    assert (
        exchange["required"]
        and "token" in exchange["content"]["application/json"]["schema"]["properties"]
    )
    with pytest.raises(ValidationError):
        Principal(owner_id="user", role="user", token_id=uuid4(), run_scope="run-1")
    with pytest.raises(Forbidden):
        context.service.create_snapshot(context.identity(), export("wrong-run"))
    # Direct repository reads share the same authorization as HTTP downloads.
    with pytest.raises(NotFound):
        context.database.get_analysis(context.identity(), UUID(queued.json()["analysis_id"]))
