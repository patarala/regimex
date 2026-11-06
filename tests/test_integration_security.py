"""Capabilities, operational commands and diagnostic metadata over the real API."""

import json
import logging
import os
from pathlib import Path
from uuid import UUID

import pytest

pytest.importorskip("psycopg", reason="install the service extra")
pytest.importorskip("httpx2", reason="install the service-test group")
pytest.importorskip("procrastinate", reason="install the queue-test group")

from fastapi.testclient import TestClient
from test_integration_scope import (
    ORIGIN,
    PREFIX,
    IntegrationContext,
    export,
    upload,
)
from test_integration_scope import (
    context as context,
)
from typer.testing import CliRunner

from regimex.access_contracts import IntegrationKeyCreate
from regimex.api import create_app
from regimex.application.auth import AuthService
from regimex.auth_contracts import AuthPolicy
from regimex.cli import app
from regimex.infrastructure.integration_repository import IntegrationRepository
from regimex.service_logging import install_http_logging


def key(context: IntegrationContext, scopes: tuple[str, ...]) -> str:
    administrator = context.database.authenticate(
        context.database.create_token("access-operator", "admin")
    )
    identity = context.database.authenticate(context.gateway_token)
    assert identity.integration_id is not None
    issued = IntegrationRepository(context.database).issue_key(
        administrator,
        identity.integration_id,
        IntegrationKeyCreate.model_validate({"name": "Limited key", "scopes": scopes}),
    )
    return issued.token.get_secret_value()


def test_scopes_apply_to_delegated_and_ordinary_routes(context: IntegrationContext) -> None:
    snapshot = upload(context)
    reader = key(context, ("read",))
    headers = context.headers(token=reader)
    assert context.client.get(f"{PREFIX}/snapshots/{snapshot}", headers=headers).status_code == 200
    assert (
        context.client.post(
            f"{PREFIX}/snapshots", json=export().model_dump(mode="json"), headers=headers
        ).status_code
        == 403
    )
    assert (
        context.client.post(
            f"{PREFIX}/analyses",
            json={"snapshot_id": str(snapshot)},
            headers={**headers, "Idempotency-Key": "reader-cannot-enqueue"},
        ).status_code
        == 403
    )
    ordinary = {"Authorization": "Bearer " + reader}
    assert context.client.get("/api/v1/models", headers=ordinary).status_code == 200
    assert (
        context.client.post(
            "/api/v1/snapshots", json=export().model_dump(mode="json"), headers=ordinary
        ).status_code
        == 403
    )
    analyzer = key(context, ("analyze",))
    response = context.client.post(
        f"{PREFIX}/analyses",
        json={"snapshot_id": str(snapshot)},
        headers={**context.headers(token=analyzer), "Idempotency-Key": "analysis-only"},
    )
    assert response.status_code == 202, response.text
    assert (
        context.client.get(
            f"{PREFIX}/analyses/{response.json()['analysis_id']}",
            headers=context.headers(token=analyzer),
        ).status_code
        == 403
    )


def test_rate_limit_has_retry_after_and_server_generated_diagnostic_id(
    context: IntegrationContext, caplog: pytest.LogCaptureFixture
) -> None:
    authentication = AuthService(
        context.authentication.repository,
        AuthPolicy(
            public_origin=ORIGIN,
            integration_requests_per_minute=3,
            integration_subject_requests_per_minute=1,
        ),
    )
    application = create_app(context.service, auth_service=authentication)
    install_http_logging(application)
    with TestClient(application, base_url=ORIGIN) as client, caplog.at_level(logging.INFO):
        headers = {**context.headers(), "X-Request-ID": "untrusted-diagnostic-value"}
        first = client.get(f"{PREFIX}/analyses", headers=headers)
        assert first.status_code == 200
        second = client.get(f"{PREFIX}/analyses", headers=headers)
        assert second.status_code == 429
        assert second.json()["error"]["code"] == "rate_limit"
        assert 1 <= int(second.headers["Retry-After"]) <= 60
        request_id = second.headers["X-Request-ID"]
        assert UUID(request_id).hex == request_id
        assert request_id != first.headers["X-Request-ID"]
    records = [record for record in caplog.records if record.name == "regimex.http"]
    matching = [record for record in records if getattr(record, "request_id", None) == request_id]
    assert len(matching) == 1 and getattr(matching[0], "status", None) == 429
    principal = context.database.authenticate(context.gateway_token)
    assert getattr(matching[0], "integration_id", None) == str(principal.integration_id)
    assert getattr(matching[0], "token_id", None) == str(principal.token_id)
    assert "untrusted-diagnostic-value" not in caplog.text
    assert context.gateway_token not in caplog.text


def test_cli_manages_keys_without_printing_or_overwriting_secrets(
    context: IntegrationContext, tmp_path: Path
) -> None:
    admin = context.database.create_token("cli-operator", "admin")
    credential = tmp_path / "admin.token"
    credential.write_text(admin)
    credential.chmod(0o600)
    environment = {
        "REGIMEX_DATABASE_DSN": os.environ["REGIMEX_TEST_DATABASE_DSN"],
        "REGIMEX_DATABASE_SCHEMA": context.database.schema,
    }
    prefix = ["service", "connections", "--token-file", str(credential)]

    def invoke(arguments: list[str]) -> str:
        root = logging.getLogger()
        handlers, level = root.handlers[:], root.level
        try:
            result = CliRunner().invoke(app, prefix + arguments, env=environment)
        finally:
            root.handlers[:] = handlers
            root.setLevel(level)
        assert result.exit_code == 0, result.output
        assert admin not in result.output
        return result.stdout

    connection = json.loads(
        invoke(["create", "--owner", "cli-platform", "--source", "fixture", "--name", "Platform"])
    )
    integration_id = connection["integration_id"]
    output = tmp_path / "gateway.token"
    issued = json.loads(
        invoke(
            [
                "key-create",
                integration_id,
                "--name",
                "Reader",
                "--scope",
                "read",
                "--days",
                "2",
                "--output",
                str(output),
            ]
        )
    )
    secret = output.read_text().strip()
    assert output.stat().st_mode & 0o777 == 0o600
    assert issued["scopes"] == ["read"] and secret not in json.dumps(issued)
    assert secret not in invoke(["keys", integration_id])
    assert secret not in invoke(["events", integration_id])
    paused = json.loads(
        invoke(["update", integration_id, "--revision", str(connection["revision"]), "--disable"])
    )
    assert paused["enabled"] is False
    invoke(["key-revoke", integration_id, issued["token_id"]])
    assert json.loads(invoke(["keys", integration_id]))["revoked_at"] is not None
