"""Trusted subject delegation and owner/source/run boundaries over actual PostgreSQL."""

import json
import os
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID, uuid4

import pytest

pytest.importorskip("psycopg", reason="install the service extra")
pytest.importorskip("procrastinate", reason="install the service extra")
pytest.importorskip("httpx2", reason="install the service-test group")

import psycopg
from fastapi.testclient import TestClient
from pydantic import ValidationError
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
from regimex.service_contracts import CapacityExceeded, Forbidden, NotFound, Principal

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
    local = context.database.authenticate(
        context.database.create_token(context.identity().owner_id, "user")
    )
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
    local = context.database.authenticate(context.database.create_token(identity.owner_id, "user"))
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
        f"/jobs/{foreign.analysis_id}",
    ):
        assert context.client.get(PREFIX + route, headers=context.headers()).status_code == 404
    assert context.database.get_operation(identity, first).state == "queued"
    assert (
        context.client.post(PREFIX + f"/jobs/{first}/cancel", headers=context.headers()).status_code
        == 200
    )


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
        "/api/v1/analyses", json=intake, headers={**csrf, "Idempotency-Key": "browser"}
    )
    assert queued.status_code == 202
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
