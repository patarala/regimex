"""HTTP boundary checks and owner isolation against disposable PostgreSQL."""

import asyncio
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Never, cast
from uuid import UUID, uuid4

import pytest

pytest.importorskip("fastapi", reason="install the service extra")
pytest.importorskip("httpx2", reason="install the service-test dependency group")
psycopg = pytest.importorskip("psycopg", reason="install the service extra")

from fastapi.testclient import TestClient
from psycopg.types.json import Jsonb
from pydantic import JsonValue
from starlette.types import Message, Scope

from regimex.api import AnalysisAPIService, BoundedJSONBody, create_app
from regimex.application.service import AnalysisResult, AnalysisService, PredictionSequence
from regimex.directions import AnalysisConfig, DirectionInterval, summarize_interval
from regimex.exports import ExportPayloadV2, ExportV2, seal_export
from regimex.infrastructure import database
from regimex.infrastructure.database import DatabaseRepository
from regimex.infrastructure.object_store import LocalObjectStore, ObjectInfo
from regimex.model_registry import ArchitectureRegistry, InferenceRuntime, baseline_descriptor
from regimex.operations import Operation
from regimex.report_contracts import AnalysisReportRecord
from regimex.service_contracts import ModelRecord, NotFound, Principal, Unauthorized
from regimex.storage import json_bytes

EXPORT = Path(__file__).parent / "fixtures" / "experiment-v2.json"
TOKEN = "boundary-test-credential"


def export_fixture() -> ExportV2:
    return ExportV2.model_validate_json(EXPORT.read_bytes())


class _ValidationService:
    """Only validation/health are exercised here; persistence tests use PostgreSQL."""

    def __init__(self, *, integration: bool = False) -> None:
        self.principal = Principal(
            owner_id="boundary-user",
            role="integration" if integration else "user",
            token_id=uuid4(),
            source_namespace="bound-source" if integration else None,
        )

    def authenticate(self, raw_token: str) -> Principal:
        if raw_token != TOKEN:
            raise Unauthorized()
        return self.principal

    def readiness(self) -> dict[str, bool]:
        return {"database": True, "active_model": False}


@pytest.fixture
def boundary_client() -> Iterator[TestClient]:
    # This fixture deliberately has no storage/model methods. Any accidental
    # computation or persistence in /validate makes the test fail immediately.
    service = cast(AnalysisAPIService, _ValidationService())
    with TestClient(create_app(service)) as client:
        yield client


def authorization(token: str = TOKEN) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_validation_uses_strict_json_contract_without_model_or_storage(
    boundary_client: TestClient,
) -> None:
    export = export_fixture()
    response = boundary_client.post(
        "/api/v1/validate", json=export.model_dump(mode="json"), headers=authorization()
    )
    assert response.status_code == 200
    assert response.json()["content_sha256"] == export.content_sha256
    assert response.json()["instrument_ids"] == ["A/Q"]
    assert response.headers["Cache-Control"] == "private, no-store"


@pytest.mark.parametrize("credential", [None, "Basic abc", "Bearer wrong-token"])
def test_validation_requires_opaque_bearer_authentication(
    boundary_client: TestClient, credential: str | None
) -> None:
    response = boundary_client.post(
        "/api/v1/validate",
        json=export_fixture().model_dump(mode="json"),
        headers={} if credential is None else {"Authorization": credential},
    )
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"
    assert response.json()["error"]["code"] == "unauthorized"


def test_validation_errors_do_not_echo_values_and_do_not_coerce_numbers(
    boundary_client: TestClient,
) -> None:
    value = export_fixture().model_dump(mode="json")
    value["time_grid"]["step_count"] = "2"
    value["external_user_id"] = "must-never-appear-in-error"
    response = boundary_client.post("/api/v1/validate", json=value, headers=authorization())
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_data"
    assert "must-never-appear-in-error" not in response.text
    assert "int_type" in response.text


@pytest.mark.parametrize("header", ["X-External-User-Id", "X-Owner-Id", "X-Source-Namespace"])
def test_http_caller_cannot_assert_owner_or_source(
    boundary_client: TestClient, header: str
) -> None:
    response = boundary_client.post(
        "/api/v1/validate",
        json=export_fixture().model_dump(mode="json"),
        headers={**authorization(), header: "another-owner"},
    )
    assert response.status_code == 403


def test_integration_validation_requires_credential_source_scope() -> None:
    service = cast(AnalysisAPIService, _ValidationService(integration=True))
    with TestClient(create_app(service)) as client:
        response = client.post(
            "/api/v1/validate",
            json=export_fixture().model_dump(mode="json"),
            headers=authorization(),
        )
    assert response.status_code == 403


def test_liveness_database_readiness_and_deployment_configuration_are_distinct(
    boundary_client: TestClient,
) -> None:
    assert boundary_client.get("/health/live").json() == {"status": "alive"}
    assert boundary_client.get("/health/ready").status_code == 200
    configured = boundary_client.get("/health/models")
    assert configured.status_code == 503
    assert configured.json() == {
        "deployment_configured": False,
        "worker_capability": "not_probed",
    }


def test_database_failure_uses_stable_error_without_echoing_connection_details() -> None:
    class Unavailable(_ValidationService):
        def authenticate(self, raw_token: str) -> Principal:
            raise psycopg.OperationalError("connection failed: password=must-not-leak")

    service = cast(AnalysisAPIService, Unavailable())
    with TestClient(create_app(service)) as client:
        response = client.post(
            "/api/v1/validate",
            json=export_fixture().model_dump(mode="json"),
            headers=authorization(),
        )
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "infrastructure_failure"
    assert "must-not-leak" not in response.text
    assert response.headers["Cache-Control"] == "private, no-store"


@pytest.mark.parametrize(
    ("headers", "status"),
    [
        ({"Content-Length": str(16 * 1024**2 + 1)}, 413),
        ({"Content-Length": "-1"}, 400),
        ({"Content-Length": "not-a-number"}, 400),
        ({"Content-Length": "4"}, 400),
        ({"Content-Encoding": "gzip"}, 415),
    ],
)
def test_intake_rejects_oversize_invalid_lengths_and_encoded_bodies(
    boundary_client: TestClient, headers: dict[str, str], status: int
) -> None:
    response = boundary_client.post("/api/v1/snapshots", content=b"{}", headers=headers)
    assert response.status_code == status


def test_chunked_limit_is_enforced_before_forwarding_or_reading_remaining_chunks() -> None:
    async def exercise() -> None:
        forwarded = False
        messages: list[Message] = [
            {"type": "http.request", "body": b"1234", "more_body": True},
            {"type": "http.request", "body": b"56789", "more_body": True},
            {"type": "http.request", "body": b"must-not-be-read", "more_body": False},
        ]
        sent: list[Message] = []

        async def receive() -> Message:
            return messages.pop(0)

        async def send(message: Message) -> None:
            sent.append(message)

        async def application(_scope: Scope, _receive: object, _send: object) -> None:
            nonlocal forwarded
            forwarded = True

        middleware = BoundedJSONBody(application, max_bytes=8)
        await middleware(
            {"type": "http", "path": "/api/v1/snapshots", "method": "POST", "headers": []},
            receive,
            send,
        )
        assert not forwarded
        assert len(messages) == 1
        assert sent[0]["status"] == 413

    asyncio.run(exercise())


def test_openapi_describes_real_json_requests_and_resolves_component_references(
    boundary_client: TestClient,
) -> None:
    schema = boundary_client.get("/openapi.json").json()
    assert schema["paths"]["/api/v1/snapshots"]["post"]["requestBody"]["required"] is True
    assert schema["paths"]["/api/v1/analyses"]["post"]["security"] == [{"HTTPBearer": []}]
    components = schema["components"]["schemas"]

    def check(value: object) -> None:
        if isinstance(value, dict):
            if "$ref" in value:
                reference = value["$ref"]
                assert isinstance(reference, str) and reference.startswith("#/components/schemas/")
                assert reference.rsplit("/", 1)[1] in components
            for child in value.values():
                check(child)
        elif isinstance(value, list):
            for child in value:
                check(child)

    check(schema)


def test_api_import_and_openapi_do_not_import_optional_science() -> None:
    script = """
import importlib.abc
import sys
from typing import cast
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        blocked = {'numpy','sklearn','torch','threadpoolctl','matplotlib','plotly'}
        if fullname.split('.')[0] in blocked:
            raise AssertionError('Scientific import at API startup: ' + fullname)
sys.meta_path.insert(0, Block())
from regimex.api import AnalysisAPIService, create_app
create_app(cast(AnalysisAPIService, object())).openapi()
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)


@dataclass
class PersistentAPI:
    repository: DatabaseRepository
    service: AnalysisService
    client: TestClient
    tokens: dict[str, str] = field(repr=False)

    def headers(self, owner: str = "alice") -> dict[str, str]:
        return authorization(self.tokens[owner])


@pytest.fixture
def persistent_api(tmp_path: Path) -> Iterator[PersistentAPI]:
    dsn = os.environ.get("REGIMEX_TEST_DATABASE_DSN")
    if not dsn:
        pytest.skip("set REGIMEX_TEST_DATABASE_DSN to a disposable PostgreSQL database")
    schema = "regimex_api_" + uuid4().hex
    repository = DatabaseRepository(dsn, schema=schema)
    try:
        repository.migrate()
        tokens = {
            "alice": repository.create_token("alice", "user"),
            "bob": repository.create_token("bob", "user"),
            "admin": repository.create_token("administrator", "admin"),
            "integration": repository.create_token(
                "source-account", "integration", source_namespace="test-simulator"
            ),
        }
        service = AnalysisService(
            repository,
            LocalObjectStore(
                tmp_path / "objects", max_object_bytes=1024**2, max_total_bytes=16 * 1024**2
            ),
            ArchitectureRegistry((baseline_descriptor(),)),
            tmp_path / "work",
        )
        with repository.queue_app.open(), TestClient(create_app(service)) as client:
            yield PersistentAPI(repository, service, client, tokens)
    finally:
        with psycopg.connect(dsn, autocommit=True) as administration:
            administration.execute(
                psycopg.sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                    psycopg.sql.Identifier(schema)
                )
            )


def register_metadata(api: PersistentAPI, version: str = "http-fixture") -> ModelRecord:
    # These deliberately non-executable records exercise HTTP persistence only.
    # Actual baseline restoration/computation is tested by the service workflow.
    record = ModelRecord(
        model_version_id=version,
        architecture_id="baseline",
        bundle_sha256=hashlib.sha256(version.encode()).hexdigest(),
        manifest=ObjectInfo(
            key=f"metadata-fixtures/{version}/manifest.json",
            size_bytes=2,
            sha256=hashlib.sha256(b"{}").hexdigest(),
            version_id=version,
        ),
        members={},
        inference_runtime=InferenceRuntime(),
        inference_policy_sha256="b" * 64,
        created_at=datetime.now(UTC),
    )
    admin = api.repository.authenticate(api.tokens["admin"])
    api.repository.register_model(admin, record)
    return record


def report_metadata_fixture(
    api: PersistentAPI,
    tmp_path: Path,
    *,
    analysis_id: UUID,
    snapshot_id: UUID,
    operation: Operation,
    result: dict[str, JsonValue],
) -> AnalysisReportRecord:
    """Store report metadata for persistence tests; workflow tests exercise rendering."""
    assert operation.model is not None
    export = export_fixture()
    report_path = tmp_path / f"{analysis_id}.html"
    report_path.write_bytes(b"<!doctype html><title>Committed report fixture</title>")
    report_object = api.service.store.put(
        f"reports/{analysis_id}.html",
        report_path,
        expected_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),
    )
    return AnalysisReportRecord(
        analysis_id=analysis_id,
        owner_id=operation.owner_id,
        snapshot_id=snapshot_id,
        snapshot_sha256=export.content_sha256,
        source=export.source,
        model=operation.model,
        result_sha256=hashlib.sha256(json_bytes(result)).hexdigest(),
        plotly_version="persistence-fixture",
        object=report_object,
        created_at=datetime.now(UTC),
    )


def upload(api: PersistentAPI, *, owner: str = "alice", suffix: str = "") -> UUID:
    export = export_fixture()
    if suffix:
        value = export.model_dump(mode="json", exclude={"content_sha256"})
        value["source"]["run_id"] += suffix
        value["source"]["snapshot_id"] += suffix
        export = seal_export(ExportPayloadV2.model_validate_json(json.dumps(value)))
    response = api.client.post(
        "/api/v1/snapshots", json=export.model_dump(mode="json"), headers=api.headers(owner)
    )
    assert response.status_code == 201, response.text
    assert "object" not in response.json()
    assert response.headers["Location"].endswith(response.json()["snapshot_id"])
    return UUID(response.json()["snapshot_id"])


def test_real_tokens_revocation_upload_download_and_owner_isolation(
    persistent_api: PersistentAPI,
) -> None:
    api = persistent_api
    snapshot_id = upload(api)
    metadata_url = f"/api/v1/snapshots/{snapshot_id}"
    export_url = f"{metadata_url}/export"
    assert api.client.get(metadata_url, headers=api.headers()).status_code == 200
    downloaded = api.client.get(export_url, headers=api.headers())
    assert downloaded.status_code == 200
    assert ExportV2.model_validate_json(downloaded.content) == export_fixture()
    assert "attachment" in downloaded.headers["Content-Disposition"]
    for owner in ("bob", "admin", "integration"):
        assert api.client.get(metadata_url, headers=api.headers(owner)).status_code == 404
        assert api.client.get(export_url, headers=api.headers(owner)).status_code == 404
    identity = api.repository.authenticate(api.tokens["alice"])
    api.repository.revoke_token(identity.token_id)
    assert api.client.get(metadata_url, headers=api.headers()).status_code == 401


def test_snapshot_identity_is_immutable_and_integration_upload_source_is_bound(
    persistent_api: PersistentAPI,
) -> None:
    api = persistent_api
    snapshot_id = upload(api)
    assert upload(api) == snapshot_id
    assert upload(api, owner="integration") != snapshot_id
    value = export_fixture().model_dump(mode="json", exclude={"content_sha256"})
    value["producer"]["version"] = "changed"
    changed = seal_export(ExportPayloadV2.model_validate_json(json.dumps(value)))
    response = api.client.post(
        "/api/v1/snapshots", json=changed.model_dump(mode="json"), headers=api.headers()
    )
    assert response.status_code == 409
    value["source"]["source_id"] = "foreign-simulator"
    foreign = seal_export(ExportPayloadV2.model_validate_json(json.dumps(value)))
    response = api.client.post(
        "/api/v1/snapshots",
        json=foreign.model_dump(mode="json"),
        headers=api.headers("integration"),
    )
    assert response.status_code == 403


@pytest.mark.parametrize("damage", ["missing", "changed"])
def test_snapshot_download_rejects_lost_or_changed_committed_bytes_without_paths(
    persistent_api: PersistentAPI, damage: str
) -> None:
    api = persistent_api
    snapshot_id = upload(api)
    principal = api.repository.authenticate(api.tokens["alice"])
    snapshot = api.repository.get_snapshot(principal, snapshot_id)
    store = api.service.store
    assert isinstance(store, LocalObjectStore)
    path = store.root / snapshot.object.key
    # Simulate disk loss/corruption outside the adapter, not authorized deletion.
    if damage == "missing":
        path.unlink()
    else:
        path.chmod(0o600)
        path.write_bytes(b"damaged private-file-content")
    response = api.client.get(f"/api/v1/snapshots/{snapshot_id}/export", headers=api.headers())
    assert response.status_code == (503 if damage == "missing" else 409)
    assert response.json()["error"]["code"] == (
        "infrastructure_failure" if damage == "missing" else "incompatible_artifact"
    )
    assert str(store.root) not in response.text
    assert "private-file-content" not in response.text


def test_integration_source_scope_applies_when_credentials_share_an_owner(
    persistent_api: PersistentAPI,
) -> None:
    api = persistent_api
    snapshot_id = upload(api, owner="integration")
    other_token = api.repository.create_token(
        "source-account", "integration", source_namespace="another-simulator"
    )
    model = register_metadata(api)
    api.repository.activate_model(
        api.repository.authenticate(api.tokens["admin"]),
        model.model_version_id,
        expected_previous=None,
    )
    response = api.client.post(
        "/api/v1/analyses",
        json={"snapshot_id": str(snapshot_id)},
        headers={**api.headers("integration"), "Idempotency-Key": "source-request"},
    )
    assert response.status_code == 202
    analysis = response.json()
    for route in (
        f"/api/v1/snapshots/{snapshot_id}",
        f"/api/v1/snapshots/{snapshot_id}/export",
        analysis["polling_url"],
        analysis["job"]["polling_url"],
        analysis["result_url"],
    ):
        assert api.client.get(route, headers=authorization(other_token)).status_code == 404
    assert (
        api.client.get("/api/v1/analyses", headers=authorization(other_token)).json()["items"] == []
    )
    assert (
        api.client.post(
            analysis["job"]["polling_url"] + "/cancel", headers=authorization(other_token)
        ).status_code
        == 404
    )
    # Matching owner/key alone cannot replay an operation from a different source.
    replay = api.client.post(
        "/api/v1/analyses",
        json={"snapshot_id": str(snapshot_id)},
        headers={**authorization(other_token), "Idempotency-Key": "source-request"},
    )
    assert replay.status_code == 404


def test_models_and_architectures_enforce_role_and_publication_visibility(
    persistent_api: PersistentAPI,
) -> None:
    api = persistent_api
    record = register_metadata(api)
    assert api.client.get("/api/v1/models", headers=api.headers()).json()["items"] == []
    assert (
        api.client.get(
            f"/api/v1/models/{record.model_version_id}", headers=api.headers()
        ).status_code
        == 404
    )
    assert (
        api.client.get("/api/v1/architectures", headers=api.headers("integration")).status_code
        == 403
    )
    architectures = api.client.get("/api/v1/architectures", headers=api.headers("admin"))
    assert architectures.status_code == 200
    assert architectures.json()["items"][0]["executable"] is None
    admin = api.repository.authenticate(api.tokens["admin"])
    api.repository.activate_model(admin, record.model_version_id, expected_previous=None)
    response = api.client.get("/api/v1/models", headers=api.headers())
    assert response.json()["items"][0]["quality_status"] == "validation_only"
    assert "manifest" not in response.text
    assert "members" not in response.text
    assert api.client.get("/api/v1/models?limit=0", headers=api.headers()).status_code == 422
    assert api.client.get("/api/v1/models?offset=1", headers=api.headers()).json()["items"] == []


def test_analysis_intake_replay_conflicts_pagination_and_cancellation(
    persistent_api: PersistentAPI,
) -> None:
    api = persistent_api
    snapshot_id = upload(api)
    first = register_metadata(api, "first")
    second = register_metadata(api, "second")
    admin = api.repository.authenticate(api.tokens["admin"])
    api.repository.activate_model(admin, first.model_version_id, expected_previous=None)
    headers = {**api.headers(), "Idempotency-Key": "same-request"}
    body = {"snapshot_id": str(snapshot_id)}
    response = api.client.post("/api/v1/analyses", json=body, headers=headers)
    assert response.status_code == 202, response.text
    submitted = response.json()
    analysis_url = submitted["polling_url"]
    job_url = submitted["job"]["polling_url"]
    assert submitted["job"]["model"]["model_version_id"] == "first"
    assert "attempt_token" not in response.text
    assert "idempotency_key" not in response.text
    api.repository.activate_model(admin, second.model_version_id, expected_previous="first")
    replay = api.client.post("/api/v1/analyses", json=body, headers=headers)
    assert replay.status_code == 202
    assert replay.json() == submitted
    changed = api.client.post(
        "/api/v1/analyses", json={**body, "model_version_id": "second"}, headers=headers
    )
    assert changed.status_code == 409
    assert api.client.post("/api/v1/analyses", json=body, headers=api.headers()).status_code == 422
    assert api.client.get("/api/v1/analyses", headers=api.headers()).json()["items"] == [submitted]
    assert api.client.get("/api/v1/analyses?offset=1", headers=api.headers()).json()["items"] == []
    assert api.client.get("/api/v1/analyses", headers=api.headers("bob")).json()["items"] == []
    for path in (analysis_url, job_url, submitted["result_url"]):
        assert api.client.get(path, headers=api.headers("bob")).status_code == 404
    assert api.client.get(submitted["result_url"], headers=api.headers()).status_code == 409
    assert api.client.post(f"{job_url}/cancel", headers=api.headers("bob")).status_code == 404
    cancelled = api.client.post(f"{job_url}/cancel", headers=api.headers())
    assert cancelled.status_code == 200
    assert cancelled.json()["state"] == "cancelled"
    assert api.client.get(analysis_url, headers=api.headers()).json()["job"]["state"] == "cancelled"


def test_same_idempotency_key_is_independent_between_authenticated_owners(
    persistent_api: PersistentAPI,
) -> None:
    api = persistent_api
    model = register_metadata(api)
    api.repository.activate_model(
        api.repository.authenticate(api.tokens["admin"]),
        model.model_version_id,
        expected_previous=None,
    )
    identities = []
    for owner in ("alice", "bob"):
        snapshot_id = upload(api, owner=owner)
        response = api.client.post(
            "/api/v1/analyses",
            json={"snapshot_id": str(snapshot_id)},
            headers={**api.headers(owner), "Idempotency-Key": "same-key"},
        )
        assert response.status_code == 202
        identities.append(response.json()["analysis_id"])
    assert identities[0] != identities[1]


def test_committed_result_survives_application_recreation_and_download_is_private(
    persistent_api: PersistentAPI, tmp_path: Path
) -> None:
    api = persistent_api
    snapshot_id = upload(api)
    model = register_metadata(api)
    api.repository.activate_model(
        api.repository.authenticate(api.tokens["admin"]),
        model.model_version_id,
        expected_previous=None,
    )
    accepted = api.client.post(
        "/api/v1/analyses",
        json={"snapshot_id": str(snapshot_id)},
        headers={**api.headers(), "Idempotency-Key": "result-fixture"},
    ).json()
    operation_id = UUID(accepted["job"]["operation_id"])
    claimed = api.repository.claim_operation(operation_id)
    assert claimed is not None and claimed.attempt_token is not None and claimed.model is not None
    export = export_fixture()
    result = AnalysisResult(
        analysis_id=UUID(accepted["analysis_id"]),
        snapshot_id=snapshot_id,
        snapshot_sha256=export.content_sha256,
        source=export.source,
        model=claimed.model,
        architecture_id="baseline",
        implementation_version="0.1.0",
        decoder_version="duration-viterbi/1.0",
        analysis_profile_id="http-result-fixture",
        analysis=AnalysisConfig(),
        inference_runtime=InferenceRuntime(),
        class_names=("sideways", "growth", "decline"),
        time_grid=export.time_grid,
        instruments=export.instruments,
        sequences=(
            PredictionSequence(
                instrument_id="A/Q",
                probabilities=((1.0, 0.0, 0.0),) * export.time_grid.step_count,
                intervals=(
                    DirectionInterval(
                        instrument_id="A/Q",
                        start=0,
                        stop=export.time_grid.step_count,
                        direction=None,
                        status="insufficient_data",
                        statistics=summarize_interval(
                            export.observations, 0, export.time_grid.step_count
                        ),
                    ),
                ),
            ),
        ),
    )
    result_payload = result.model_dump(mode="json")
    assert result_payload["result_version"] == "2.0"
    for previous_shape in (
        {**result_payload, "result_version": "1.0"},
        {
            **result_payload,
            "sequences": [{**sequence, "events": []} for sequence in result_payload["sequences"]],
        },
    ):
        with pytest.raises(ValueError):
            AnalysisResult.model_validate_json(json.dumps(previous_shape))
    report = report_metadata_fixture(
        api,
        tmp_path,
        analysis_id=result.analysis_id,
        snapshot_id=snapshot_id,
        operation=claimed,
        result=result_payload,
    )
    api.repository.complete_analysis(
        operation_id, claimed.attempt_token, result_payload, report=report
    )
    with TestClient(create_app(api.service)) as restarted:
        response = restarted.get(accepted["result_url"], headers=api.headers())
        assert response.status_code == 200
        assert AnalysisResult.model_validate_json(response.content) == result
        assert response.headers["Cache-Control"] == "private, no-store"
        assert "attachment" in response.headers["Content-Disposition"]
        assert restarted.get(accepted["result_url"], headers=api.headers("bob")).status_code == 404
        replay = restarted.post(
            "/api/v1/analyses",
            json={"snapshot_id": str(snapshot_id)},
            headers={**api.headers(), "Idempotency-Key": "result-fixture"},
        )
        status = restarted.get(accepted["polling_url"], headers=api.headers())
        assert replay.status_code == 202 and status.status_code == 200
        assert replay.json() == status.json()
        assert replay.json()["result_available"] is True
        assert replay.json()["report_available"] is True


def test_status_and_lists_read_metadata_without_loading_large_committed_results(
    persistent_api: PersistentAPI, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    api = persistent_api
    snapshot_id = upload(api, owner="integration")
    model = register_metadata(api)
    api.repository.activate_model(
        api.repository.authenticate(api.tokens["admin"]),
        model.model_version_id,
        expected_previous=None,
    )
    response = api.client.post(
        "/api/v1/analyses",
        json={"snapshot_id": str(snapshot_id)},
        headers={**api.headers("integration"), "Idempotency-Key": "large-result-metadata"},
    )
    assert response.status_code == 202
    accepted = response.json()
    operation_id = UUID(accepted["job"]["operation_id"])
    claimed = api.repository.claim_operation(operation_id)
    assert claimed is not None and claimed.attempt_token is not None and claimed.model is not None
    # This opaque JSON fixture exercises metadata allocation, not inference or
    # result-schema validity. The result-download tests validate that separately.
    export = export_fixture()
    payload: dict[str, JsonValue] = {
        "analysis_id": accepted["analysis_id"],
        "snapshot_id": str(snapshot_id),
        "snapshot_sha256": export.content_sha256,
        "source": export.source.model_dump(mode="json"),
        "model": claimed.model.model_dump(mode="json"),
        "metadata-allocation-fixture": "x" * (2 * 1024**2),
    }
    report = report_metadata_fixture(
        api,
        tmp_path,
        analysis_id=UUID(accepted["analysis_id"]),
        snapshot_id=snapshot_id,
        operation=claimed,
        result=payload,
    )
    api.repository.complete_analysis(operation_id, claimed.attempt_token, payload, report=report)

    def full_result_read(*_args: object, **_kwargs: object) -> Never:
        raise AssertionError("A metadata route attempted to load the full JSON result")

    monkeypatch.setattr(api.repository, "_analysis", full_result_read)
    monkeypatch.setattr(api.repository, "get_analysis_internal", full_result_read)
    monkeypatch.setattr(api.repository, "get_analysis", full_result_read)
    for route in (accepted["polling_url"], accepted["job"]["polling_url"]):
        status = api.client.get(route, headers=api.headers("integration"))
        assert status.status_code == 200
        assert len(status.content) < 4096
        assert "metadata-allocation-fixture" not in status.text
    assert (
        api.client.get(accepted["polling_url"], headers=api.headers("integration")).json()[
            "result_available"
        ]
        is True
    )
    listing = api.client.get("/api/v1/analyses", headers=api.headers("integration"))
    assert listing.status_code == 200
    assert len(listing.content) < 4096
    assert len(listing.json()["items"]) == 1
    assert listing.json()["items"][0]["result_available"] is True
    assert "metadata-allocation-fixture" not in listing.text


def test_unsupported_stored_operation_is_private_explicit_and_unchanged(
    persistent_api: PersistentAPI,
) -> None:
    api = persistent_api
    snapshot_id = upload(api, owner="integration")
    model = register_metadata(api)
    api.repository.activate_model(
        api.repository.authenticate(api.tokens["admin"]),
        model.model_version_id,
        expected_previous=None,
    )
    submitted = api.client.post(
        "/api/v1/analyses",
        json={"snapshot_id": str(snapshot_id)},
        headers={**api.headers("integration"), "Idempotency-Key": "unsupported-operation"},
    )
    assert submitted.status_code == 202
    payload = submitted.json()
    operation_id = UUID(payload["job"]["operation_id"])
    with api.repository.connection() as connection:
        row = connection.execute(
            "SELECT document FROM regimex_operations WHERE operation_id=%s", (operation_id,)
        ).fetchone()
        assert row is not None
        historical = {**cast(dict[str, JsonValue], row[0]), "version": "1.0"}
        connection.execute(
            "UPDATE regimex_operations SET document=%s WHERE operation_id=%s",
            (psycopg.types.json.Json(historical), operation_id),
        )
    other_source = api.repository.create_token(
        "source-account", "integration", source_namespace="another-simulator"
    )
    other_run = api.repository.authenticate(api.tokens["integration"]).model_copy(
        update={"run_scope": "other-run"}
    )
    with pytest.raises(NotFound):
        api.repository.get_operation(other_run, operation_id)
    for path in (payload["polling_url"], payload["job"]["polling_url"]):
        response = api.client.get(path, headers=api.headers("integration"))
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "unsupported_contract"
        assert response.headers["Cache-Control"] == "private, no-store"
        for headers in (api.headers("alice"), authorization(other_source)):
            assert api.client.get(path, headers=headers).status_code == 404
    cancel = api.client.post(
        payload["job"]["polling_url"] + "/cancel", headers=api.headers("integration")
    )
    assert cancel.status_code == 409
    with api.repository.connection() as connection:
        row = connection.execute(
            "SELECT document FROM regimex_operations WHERE operation_id=%s", (operation_id,)
        ).fetchone()
        assert row == (historical,)


@pytest.mark.parametrize("resource", ["result", "report"])
@pytest.mark.parametrize("version", ["previous", "missing"])
def test_unsupported_saved_result_or_report_is_private_and_unchanged(
    persistent_api: PersistentAPI,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    resource: str,
    version: str,
) -> None:
    api = persistent_api
    snapshot_id = upload(api, owner="integration")
    model = register_metadata(api)
    api.repository.activate_model(
        api.repository.authenticate(api.tokens["admin"]),
        model.model_version_id,
        expected_previous=None,
    )
    response = api.client.post(
        "/api/v1/analyses",
        json={"snapshot_id": str(snapshot_id)},
        headers={**api.headers("integration"), "Idempotency-Key": "unsupported-saved-artifact"},
    )
    assert response.status_code == 202
    accepted = response.json()
    analysis_id = UUID(accepted["analysis_id"])
    operation_id = UUID(accepted["job"]["operation_id"])
    claimed = api.repository.claim_operation(operation_id)
    assert claimed is not None and claimed.attempt_token is not None and claimed.model is not None
    export = export_fixture()
    # The version must be rejected before interpreting a historical payload;
    # only the identity fields needed for the atomic persistence fixture remain.
    payload: dict[str, JsonValue] = {
        "analysis_id": str(analysis_id),
        "snapshot_id": str(snapshot_id),
        "snapshot_sha256": export.content_sha256,
        "source": export.source.model_dump(mode="json"),
        "model": claimed.model.model_dump(mode="json"),
    }
    if version == "previous":
        payload["result_version"] = "1.0"
    report = report_metadata_fixture(
        api,
        tmp_path,
        analysis_id=analysis_id,
        snapshot_id=snapshot_id,
        operation=claimed,
        result=payload,
    )

    def encode_historical_fixture(value: object) -> Jsonb:
        if resource == "report" and isinstance(value, dict) and "renderer_version" in value:
            value = dict(value)
            if version == "previous":
                value["renderer_version"] = "analysis-html/1.0"
            else:
                value.pop("renderer_version")
        return Jsonb(value)

    # Seed the former document at insertion, as an older application would have;
    # keep the database's report immutability triggers active for the whole test.
    with monkeypatch.context() as insertion:
        insertion.setattr(database, "Jsonb", encode_historical_fixture)
        api.repository.complete_analysis(
            operation_id, claimed.attempt_token, payload, report=report
        )
    state_query = (
        "SELECT a.result::text,o.document::text,r.document::text "
        "FROM regimex_analyses a JOIN regimex_operations o ON a.operation_id=o.operation_id "
        "JOIN regimex_analysis_reports r ON a.analysis_id=r.analysis_id WHERE a.analysis_id=%s"
    )
    with api.repository.connection() as connection:
        before = connection.execute(state_query, (analysis_id,)).fetchone()
    other_source = api.repository.create_token(
        "source-account", "integration", source_namespace="another-simulator"
    )
    paths = [f"/api/v1/analyses/{analysis_id}/{resource}"]
    if resource == "report":
        paths.append(paths[0] + "/preview")
    for path in paths:
        rejected = api.client.get(path, headers=api.headers("integration"))
        assert rejected.status_code == 409
        assert rejected.json()["error"]["code"] == "unsupported_contract"
        assert rejected.headers["Cache-Control"] == "private, no-store"
        for headers in (api.headers("alice"), authorization(other_source)):
            assert api.client.get(path, headers=headers).status_code == 404
    other_run = api.repository.authenticate(api.tokens["integration"]).model_copy(
        update={"run_scope": "other-run"}
    )
    with pytest.raises(NotFound):
        if resource == "result":
            api.service.get_result(other_run, analysis_id)
        else:
            api.service.read_report(other_run, analysis_id)
    with api.repository.connection() as connection:
        assert connection.execute(state_query, (analysis_id,)).fetchone() == before
