"""Connection administration through authenticated HTTP and real PostgreSQL."""

# ruff: noqa: E402

import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import pytest

psycopg = pytest.importorskip("psycopg", reason="install the service extra")
pytest.importorskip("procrastinate", reason="install the service extra")
pytest.importorskip("httpx2", reason="install the service-test group")

from fastapi.testclient import TestClient

from regimex.api import create_app
from regimex.application.auth import AuthService
from regimex.application.service import AnalysisService
from regimex.auth_contracts import AuthPolicy
from regimex.infrastructure.auth_repository import AuthRepository
from regimex.infrastructure.database import DatabaseRepository
from regimex.infrastructure.object_store import LocalObjectStore
from regimex.model_registry import ArchitectureRegistry

ORIGIN = "https://regimex.example"
ROOT = "/api/v1/admin/integrations"


@dataclass
class Context:
    client: TestClient
    database: DatabaseRepository
    admin_token: str
    user_token: str

    def headers(self, *, admin: bool = True) -> dict[str, str]:
        return {"Authorization": "Bearer " + (self.admin_token if admin else self.user_token)}

    def create(self, name: str = "Research platform") -> str:
        response = self.client.post(
            ROOT,
            json={"owner_id": "platform", "source_namespace": "simulator", "display_name": name},
            headers=self.headers(),
        )
        assert response.status_code == 201, response.text
        assert response.headers["cache-control"] == "private, no-store"
        identity = str(response.json()["integration_id"])
        assert response.headers["location"] == f"{ROOT}/{identity}"
        return identity


@pytest.fixture
def context(tmp_path: Path) -> Iterator[Context]:
    dsn = os.environ.get("REGIMEX_TEST_DATABASE_DSN")
    if not dsn:
        pytest.skip("set REGIMEX_TEST_DATABASE_DSN to a disposable PostgreSQL database")
    database = DatabaseRepository(dsn, schema="regimex_access_api_" + uuid4().hex)
    try:
        database.migrate()
        admin = database.create_token("administrator", "admin")
        user = database.create_token("researcher", "user")
        auth = AuthService(AuthRepository(database), AuthPolicy(public_origin=ORIGIN))
        service = AnalysisService(
            database,
            LocalObjectStore(
                tmp_path / "objects", max_object_bytes=1024**2, max_total_bytes=16 * 1024**2
            ),
            ArchitectureRegistry(()),
            tmp_path / "work",
        )
        with TestClient(create_app(service, auth_service=auth), base_url=ORIGIN) as client:
            yield Context(client, database, admin, user)
    finally:
        with psycopg.connect(dsn, autocommit=True) as connection:
            connection.execute(
                psycopg.sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                    psycopg.sql.Identifier(database.schema)
                )
            )


def test_connection_routes_require_administrator(context: Context) -> None:
    identity = context.create()
    paths = [ROOT, f"{ROOT}/{identity}", f"{ROOT}/{identity}/keys", f"{ROOT}/{identity}/events"]
    for path in paths:
        assert context.client.get(path).status_code == 401
        assert context.client.get(path, headers=context.headers(admin=False)).status_code == 403
    response = context.client.post(
        ROOT,
        json={"owner_id": "other", "source_namespace": "source", "display_name": "Other"},
        headers=context.headers(admin=False),
    )
    assert response.status_code == 403
    assert (
        context.client.patch(
            f"{ROOT}/{identity}",
            json={"expected_revision": 0, "enabled": False},
            headers=context.headers(admin=False),
        ).status_code
        == 403
    )


def test_issued_secret_is_absent_from_metadata_events_and_second_read(context: Context) -> None:
    identity = context.create()
    response = context.client.post(
        f"{ROOT}/{identity}/keys",
        json={"name": "Read only", "scopes": ["read"]},
        headers=context.headers(),
    )
    assert response.status_code == 201, response.text
    issued = response.json()
    token = issued["token"]
    assert token.startswith("rgx_") and len(token) == 47
    assert response.headers["cache-control"] == "private, no-store"
    assert response.headers["pragma"] == "no-cache"
    assert issued["key"]["scopes"] == ["read"]
    assert issued["key"]["expires_at"]
    for suffix in ("", "/keys", "/events"):
        metadata = context.client.get(f"{ROOT}/{identity}{suffix}", headers=context.headers())
        assert metadata.status_code == 200
        assert token not in metadata.text
        assert '"token_sha256"' not in metadata.text
    keys = context.client.get(f"{ROOT}/{identity}/keys", headers=context.headers()).json()["items"]
    assert [key["token_id"] for key in keys] == [issued["key"]["token_id"]]
    revoked = context.client.post(
        f"{ROOT}/{identity}/keys/{issued['key']['token_id']}/revoke", headers=context.headers()
    )
    assert revoked.status_code == 204
    assert revoked.content == b""
    assert context.client.get(ROOT, headers={"Authorization": "Bearer " + token}).status_code == 401


def test_cookie_administration_requires_origin_and_session_csrf(context: Context) -> None:
    login = context.client.post(
        "/api/v1/session", json={"token": context.admin_token}, headers={"Origin": ORIGIN}
    )
    assert login.status_code == 200
    body = {"owner_id": "platform", "source_namespace": "simulator", "display_name": "Platform"}
    assert context.client.post(ROOT, json=body).status_code == 403
    assert context.client.post(ROOT, json=body, headers={"Origin": ORIGIN}).status_code == 403
    headers = {"Origin": ORIGIN, "X-CSRF-Token": login.json()["csrf_token"]}
    response = context.client.post(ROOT, json=body, headers=headers)
    assert response.status_code == 201, response.text
    assert context.client.get(ROOT).status_code == 200


def test_revision_conflict_and_bounded_metadata_pages(context: Context) -> None:
    identity = context.create()
    response = context.client.patch(
        f"{ROOT}/{identity}",
        json={"expected_revision": 0, "enabled": False},
        headers=context.headers(),
    )
    assert response.status_code == 200, response.text
    assert response.json()["enabled"] is False
    assert response.json()["revision"] == 1
    stale = context.client.patch(
        f"{ROOT}/{identity}",
        json={"expected_revision": 0, "display_name": "Stale edit"},
        headers=context.headers(),
    )
    assert stale.status_code == 409
    events = context.client.get(
        f"{ROOT}/{identity}/events?limit=1&offset=0", headers=context.headers()
    )
    assert events.status_code == 200
    assert len(events.json()["items"]) == 1
    assert events.json()["offset"] == 0
    assert events.json()["limit"] == 1
    assert context.client.get(ROOT + "?limit=101", headers=context.headers()).status_code == 422
    assert context.client.get(ROOT + "?offset=-1", headers=context.headers()).status_code == 422
    assert (
        context.client.post(
            f"{ROOT}/{identity}/keys",
            json={"name": "Too long", "lifetime_days": 366},
            headers=context.headers(),
        ).status_code
        == 422
    )


def test_key_revoke_cannot_cross_connections(context: Context) -> None:
    first = context.create()
    second_response = context.client.post(
        ROOT,
        json={"owner_id": "second", "source_namespace": "simulator", "display_name": "Second"},
        headers=context.headers(),
    )
    assert second_response.status_code == 201
    second = second_response.json()["integration_id"]
    issued = context.client.post(
        f"{ROOT}/{first}/keys", json={"name": "Private service key"}, headers=context.headers()
    ).json()
    response = context.client.post(
        f"{ROOT}/{second}/keys/{issued['key']['token_id']}/revoke", headers=context.headers()
    )
    assert response.status_code == 404
    assert context.database.authenticate(issued["token"]).integration_id is not None
