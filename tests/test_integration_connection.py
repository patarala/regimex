"""Connection self-checks authenticate actual keys without delegating a subject."""

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("psycopg", reason="install the service extra")
pytest.importorskip("httpx2", reason="install the service-test group")
pytest.importorskip("procrastinate", reason="install the queue-test group")

from fastapi.testclient import TestClient
from test_integration_admin_api import Context
from test_integration_admin_api import context as context

from regimex.api import create_app
from regimex.application.auth import AuthService
from regimex.application.service import AnalysisService
from regimex.auth_contracts import AuthPolicy
from regimex.infrastructure.auth_repository import AuthRepository
from regimex.infrastructure.object_store import LocalObjectStore
from regimex.model_registry import ArchitectureRegistry

PATH = "/api/v1/connection"


def issue(context: Context, scopes: list[str] | None = None) -> dict[str, Any]:
    identifier = context.create()
    response = context.client.post(
        f"/api/v1/admin/integrations/{identifier}/keys",
        json={"name": "Diagnostic key", "scopes": scopes or ["read", "upload", "analyze"]},
        headers=context.headers(),
    )
    assert response.status_code == 201, response.text
    return {**response.json(), "integration_id": identifier}


def test_self_check_reports_exact_source_and_permissions_without_creating_subjects(
    context: Context,
) -> None:
    issued = issue(context, ["upload"])
    response = context.client.get(PATH, headers={"Authorization": "Bearer " + issued["token"]})
    assert response.status_code == 200, response.text
    assert response.json() == {
        "analysis_protocol": "regimex-analysis/1",
        "source_namespace": "simulator",
        "scopes": ["upload"],
    }
    assert response.headers["cache-control"] == "private, no-store"
    assert issued["token"] not in response.text
    assert issued["key"]["token_id"] not in response.text
    assert issued["integration_id"] not in response.text
    with context.database.connection() as connection:
        assert connection.execute(
            "SELECT count(*) FROM regimex_integration_subjects"
        ).fetchone() == (0,)
        assert connection.execute("SELECT count(*) FROM regimex_snapshots").fetchone() == (0,)
        assert connection.execute("SELECT count(*) FROM regimex_analyses").fetchone() == (0,)
    # Self-introspection does not grant the read capability on other endpoints.
    denied = context.client.get(
        "/api/v1/models", headers={"Authorization": "Bearer " + issued["token"]}
    )
    assert denied.status_code == 403


def test_self_check_rejects_other_roles_sessions_and_browser_or_subject_headers(
    context: Context,
) -> None:
    assert context.client.get(PATH).status_code == 401
    assert context.client.get(PATH, headers=context.headers()).status_code == 403
    assert context.client.get(PATH, headers=context.headers(admin=False)).status_code == 403
    issued = issue(context)
    headers = {"Authorization": "Bearer " + issued["token"]}
    for extra in (
        {"Origin": "https://regimex.example"},
        {"Sec-Fetch-Site": "same-origin"},
        {"Cookie": "unrelated=browser"},
        {"X-Regimex-Subject": "invented-user"},
        {"X-Regimex-Run-ID": "invented-run"},
        {"X-Owner-ID": "invented-owner"},
    ):
        response = context.client.get(PATH, headers={**headers, **extra})
        assert response.status_code == 403, response.text
    login = context.client.post(
        "/api/v1/session",
        json={"token": context.admin_token},
        headers={"Origin": "https://regimex.example"},
    )
    assert login.status_code == 200
    assert context.client.get(PATH).status_code == 403


@pytest.mark.parametrize("state", ["invalid", "expired", "revoked", "disabled"])
def test_rejected_key_states_have_the_same_public_unauthorized_response(
    context: Context, state: str
) -> None:
    issued = issue(context)
    token = issued["token"]
    if state == "invalid":
        token = "rgx_" + "x" * 43
    else:
        with context.database.connection() as connection:
            if state == "expired":
                connection.execute(
                    "UPDATE regimex_tokens SET created_at=clock_timestamp()-interval '2 seconds', "
                    "expires_at=clock_timestamp()-interval '1 second' WHERE token_id=%s",
                    (issued["key"]["token_id"],),
                )
            elif state == "revoked":
                connection.execute(
                    "UPDATE regimex_tokens SET revoked_at=clock_timestamp() WHERE token_id=%s",
                    (issued["key"]["token_id"],),
                )
            else:
                connection.execute(
                    "UPDATE regimex_integrations SET enabled=false WHERE integration_id=%s",
                    (issued["integration_id"],),
                )
    response = context.client.get(PATH, headers={"Authorization": "Bearer " + token})
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"
    assert token not in response.text


def test_connection_checks_share_the_existing_account_rate_budget(
    context: Context, tmp_path: Path
) -> None:
    issued = issue(context)
    service = AnalysisService(
        context.database,
        LocalObjectStore(
            tmp_path / "rate-objects",
            max_object_bytes=1024**2,
            max_total_bytes=16 * 1024**2,
        ),
        ArchitectureRegistry(()),
        tmp_path / "rate-work",
    )
    auth = AuthService(
        AuthRepository(context.database),
        AuthPolicy(
            public_origin="https://regimex.example",
            integration_requests_per_minute=1,
            integration_subject_requests_per_minute=1,
        ),
    )
    with TestClient(
        create_app(service, auth_service=auth), base_url="https://regimex.example"
    ) as client:
        headers = {"Authorization": "Bearer " + issued["token"]}
        assert client.get(PATH, headers=headers).status_code == 200
        limited = client.get(PATH, headers=headers)
        assert limited.status_code == 429
        assert 1 <= int(limited.headers["retry-after"]) <= 60
