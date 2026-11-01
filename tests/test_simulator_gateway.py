"""JWT, immutable export and bounded gateway failure behavior using deliberate fixtures."""

from __future__ import annotations

import base64
import hashlib
import hmac
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest
from pydantic import SecretBytes, SecretStr, ValidationError
from support.simulator_gateway.client import GatewayError, GatewaySettings, RegimexClient
from support.simulator_gateway.jwt import (
    InvalidToken,
    SimulatorJWT,
    _verified_claims,
    json_bytes,
)

from regimex.exports import ExportPayloadV2, ExportV2, parse_export_v2, seal_export

if TYPE_CHECKING:
    from fastapi.testclient import TestClient
    from support.simulator_gateway.app import FixtureCatalog

FIXTURE_KEY = b"deliberate-public-test-key/0123456789"
FIXTURE_TOKEN = "rgx_deliberate_software_fixture_only"


def signed(claims: object, *, header: object | None = None, raw_claims: bytes | None = None) -> str:
    def encode(data: bytes) -> str:
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode()

    first = encode(json_bytes({"alg": "HS256", "typ": "JWT"} if header is None else header))
    second = encode(json_bytes(claims) if raw_claims is None else raw_claims)
    signature = hmac.digest(FIXTURE_KEY, f"{first}.{second}".encode(), hashlib.sha256)
    return f"{first}.{second}.{encode(signature)}"


@pytest.fixture
def jwt() -> SimulatorJWT:
    return SimulatorJWT("fixture-issuer", "fixture-audience", SecretBytes(FIXTURE_KEY))


def test_hs256_matches_independent_published_jws_vector() -> None:
    # RFC 7515 Appendix A.1: a codec vector, not a usable fixture identity token.
    # https://www.rfc-editor.org/rfc/rfc7515.html#appendix-A.1
    key = base64.urlsafe_b64decode(
        "AyM1SysPpbyDfgZld3umj1qzKObwVMkoqQ-EstJQLr_T-1qS0gZH75aKtMN3Yj0iPS4hcgUuTwjAzZr1Z9CAow=="
    )
    token = (
        "eyJ0eXAiOiJKV1QiLA0KICJhbGciOiJIUzI1NiJ9."
        "eyJpc3MiOiJqb2UiLA0KICJleHAiOjEzMDA4MTkzODAsDQogImh0dHA6Ly9leGFt"
        "cGxlLmNvbS9pc19yb290Ijp0cnVlfQ."
        "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"
    )
    assert _verified_claims(token, SecretBytes(key))["iss"] == "joe"
    with pytest.raises(InvalidToken):
        _verified_claims(token[:-1] + "A", SecretBytes(key))


def test_fixture_jwt_verifies_signature_and_exact_expiration(jwt: SimulatorJWT) -> None:
    token = jwt.issue("alice", now=1000, lifetime=60)
    assert jwt.verify(token, now=1000) == "alice"
    assert jwt.verify(token, now=1059) == "alice"
    for now in (999, 1060):
        with pytest.raises(InvalidToken):
            jwt.verify(token, now=now)
    with pytest.raises(InvalidToken):
        jwt.verify(token[:-1] + ("A" if token[-1] != "A" else "B"), now=1000)


@pytest.mark.parametrize(
    "change",
    [
        {"iss": "other-issuer"},
        {"aud": "other-audience"},
        {"aud": ["fixture-audience"]},
        {"sub": "alice\r\nX-Owner: bob"},
        {"sub": "юзер"},
        {"sub": "a" * 129},
        {"exp": 1060.0},
        {"iat": True},
        {"nbf": 1001},
        {"exp": 5001},
        {"iat": 1001},
    ],
)
def test_fixture_jwt_rejects_untrusted_identity_and_claims(
    jwt: SimulatorJWT,
    change: dict[str, object],
) -> None:
    claims = {
        "iss": "fixture-issuer",
        "aud": "fixture-audience",
        "sub": "alice",
        "iat": 1000,
        "exp": 1060,
    }
    claims.update(change)
    with pytest.raises(InvalidToken):
        jwt.verify(signed(claims), now=1000)


@pytest.mark.parametrize(
    "header",
    [
        {"alg": "none", "typ": "JWT"},
        {"alg": "RS256", "typ": "JWT"},
        {"alg": "HS256", "typ": "JWT", "jku": "https://example.invalid/key"},
        {"alg": "HS256", "typ": "JWT", "crit": ["custom"]},
    ],
)
def test_fixture_jwt_does_not_negotiate_algorithms_or_keys(
    jwt: SimulatorJWT,
    header: object,
) -> None:
    with pytest.raises(InvalidToken):
        jwt.verify(signed({}, header=header), now=1000)


def test_fixture_jwt_rejects_ambiguous_json_without_leaking_key(jwt: SimulatorJWT) -> None:
    with pytest.raises(InvalidToken, match=r"^Invalid simulator authentication\.$"):
        jwt.verify(signed({}, raw_claims=b'{"sub":"alice","sub":"bob"}'), now=1000)
    assert FIXTURE_KEY.decode() not in repr(jwt)


@dataclass
class Upstream:
    origin: str = ""
    requests: list[tuple[str, str, str | None, str | None, bool]] = field(default_factory=list)
    responder: Callable[[BaseHTTPRequestHandler], None] | None = None


@contextmanager
def upstream_server() -> Iterator[Upstream]:
    state = Upstream()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            pass

        def do_GET(self) -> None:
            state.requests.append(
                (
                    self.command,
                    self.path,
                    self.headers.get("X-Regimex-Subject"),
                    self.headers.get("X-Regimex-Run-ID"),
                    self.headers.get("Authorization") == "Bearer " + FIXTURE_TOKEN,
                )
            )
            if state.responder is not None:
                with suppress(BrokenPipeError, ConnectionResetError):
                    state.responder(self)
            else:
                self.send_response(503)
                self.end_headers()

        do_POST = do_GET

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    thread.start()
    state.origin = f"http://127.0.0.1:{server.server_port}"
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        assert not thread.is_alive()


def configuration(origin: str, **kwargs: object) -> GatewaySettings:
    return GatewaySettings.model_validate(
        {
            "source_namespace": "test-simulator",
            "regimex_origin": origin,
            "integration_token": SecretStr(FIXTURE_TOKEN),
            "allow_loopback_http": True,
            **kwargs,
        }
    )


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308, 401, 403, 500, 503])
def test_client_never_follows_redirects_or_returns_upstream_secrets(status: int) -> None:
    def respond(handler: BaseHTTPRequestHandler) -> None:
        handler.send_response(status)
        handler.send_header("Location", server.origin + "/private-target")
        handler.end_headers()
        handler.wfile.write(b"upstream-secret-never-forwarded")

    with upstream_server() as server:
        server.responder = respond
        client = RegimexClient(configuration(server.origin))
        with pytest.raises(GatewayError) as caught:
            client.json("GET", "/api/v1/integration/snapshots", subject="alice", run_id="run-1")
        assert caught.value.status == 503 and caught.value.retryable
        assert "upstream-secret" not in str(caught.value)
        assert len(server.requests) == 1
        assert server.requests[0][2:] == ("alice", "run-1", True)


@pytest.mark.parametrize("mode", ["declared", "streamed", "truncated", "duplicate-json"])
def test_client_bounds_and_verifies_untrusted_response(mode: str) -> None:
    def respond(handler: BaseHTTPRequestHandler) -> None:
        handler.send_response(200)
        handler.send_header("Content-Type", "application/json")
        if mode == "declared":
            handler.send_header("Content-Length", "1025")
        elif mode == "truncated":
            handler.send_header("Content-Length", "100")
        handler.end_headers()
        handler.wfile.write(
            b'{"a":1,"a":2}'
            if mode == "duplicate-json"
            else b"x" * 1025
            if mode in {"declared", "streamed"}
            else b"x"
        )

    with upstream_server() as server:
        server.responder = respond
        client = RegimexClient(configuration(server.origin, response_bytes=1024))
        with pytest.raises(GatewayError) as caught:
            client.json("GET", "/api/v1/integration/snapshots", subject="alice", run_id="run-1")
        assert caught.value.status in {502, 503} and caught.value.retryable


def test_client_total_deadline_stops_slow_headers() -> None:
    def respond(handler: BaseHTTPRequestHandler) -> None:
        for byte in b"HTTP/1.0 200 OK\r\nContent-Type: application/json\r\n\r\n{}":
            handler.wfile.write(bytes((byte,)))
            handler.wfile.flush()
            time.sleep(0.03)

    with upstream_server() as server:
        server.responder = respond
        client = RegimexClient(configuration(server.origin, total_timeout=0.15, read_timeout=1.0))
        start = time.monotonic()
        with pytest.raises(GatewayError):
            client.json("GET", "/api/v1/integration/snapshots", subject="alice", run_id="run-1")
        assert time.monotonic() - start < 0.7


def test_client_connect_budget_also_bounds_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    import socket

    finished = threading.Event()

    def stalled_lookup(*args: object, **kwargs: object) -> list[object]:
        try:
            time.sleep(0.3)
            raise OSError("private resolver details")
        finally:
            finished.set()

    monkeypatch.setattr(socket, "getaddrinfo", stalled_lookup)
    client = RegimexClient(
        configuration(
            "http://127.0.0.1:1",
            connect_timeout=0.05,
            total_timeout=0.1,
        )
    )
    start = time.monotonic()
    with pytest.raises(GatewayError) as caught:
        client.json("GET", "/api/v1/integration/snapshots", subject="alice", run_id="run-1")
    assert caught.value.status == 503 and "private resolver" not in str(caught.value)
    assert time.monotonic() - start < 0.2
    assert finished.wait(timeout=1)


@pytest.mark.parametrize(
    "origin",
    [
        "http://example.com",
        "http://127.0.0.1/path",
        "https://user:secret@example.com",
        "https://example.com?token=secret",
        "https://example.com:bad",
        "https://example.com\\secret",
    ],
)
def test_invalid_gateway_configuration_does_not_echo_input(origin: str) -> None:
    with pytest.raises(ValidationError) as caught:
        configuration(origin)
    assert origin not in str(caught.value) and "secret" not in str(caught.value)


@dataclass
class GatewayContext:
    client: TestClient
    catalog: FixtureCatalog
    export: ExportV2
    jwt: SimulatorJWT
    upstream: Upstream


@pytest.fixture
def gateway(jwt: SimulatorJWT) -> Iterator[GatewayContext]:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx2")
    from fastapi.testclient import TestClient
    from support.simulator_gateway.app import FixtureCatalog, FixtureRun, create_gateway

    export = parse_export_v2((Path(__file__).parent / "fixtures/experiment-v2.json").read_bytes())
    catalog = FixtureCatalog(export.source.source_id)
    catalog.register(
        FixtureRun.completed("configuration-1", "alice", export, results={"trades": [1]})
    )
    catalog.register(FixtureRun("configuration-1", "running", "alice", b'{"trades":[]}'))
    with (
        upstream_server() as server,
        TestClient(
            create_gateway(
                configuration(server.origin),
                jwt,
                catalog,
            )
        ) as client,
    ):
        yield GatewayContext(client, catalog, export, jwt, server)


def authenticated(context: GatewayContext, subject: str = "alice") -> dict[str, str]:
    return {"Authorization": "Bearer " + context.jwt.issue(subject)}


def test_export_is_atomic_owned_and_preserves_missingness(gateway: GatewayContext) -> None:
    response = gateway.client.get(
        "/api/simulations/configuration-1/export?run_id=run-1&schema_version=2.1",
        headers=authenticated(gateway),
    )
    assert response.status_code == 200
    assert parse_export_v2(response.content) == gateway.export
    for subject, run_id, expected in (("bob", "run-1", 404), ("alice", "running", 409)):
        response = gateway.client.get(
            "/api/simulations/configuration-1/export",
            params={"run_id": run_id},
            headers=authenticated(gateway, subject),
        )
        assert response.status_code == expected
    assert gateway.upstream.requests == []


def test_completed_fixture_cannot_be_overwritten(gateway: GatewayContext) -> None:
    from support.simulator_gateway.app import FixtureRun

    changed = gateway.export.model_dump(mode="json", exclude={"content_sha256"})
    assert isinstance(changed["source"], dict)
    changed["source"]["state_revision"] = "replacement"
    export = seal_export(ExportPayloadV2.model_validate_json(json_bytes(changed)))
    with pytest.raises(ValueError, match="cannot be overwritten"):
        gateway.catalog.register(
            FixtureRun.completed("configuration-1", "alice", export, results={})
        )


@pytest.mark.parametrize("assertion", ["X-Regimex-Subject", "X-Regimex-Run-ID", "X-Owner-ID"])
def test_browser_cannot_supply_trusted_identity(gateway: GatewayContext, assertion: str) -> None:
    response = gateway.client.post(
        "/api/simulations/configuration-1/regime-analyses",
        json={"run_id": "run-1"},
        headers={**authenticated(gateway), assertion: "bob"},
    )
    assert response.status_code == 400 and gateway.upstream.requests == []


@pytest.mark.parametrize(
    "payload",
    [
        {"run_id": "run-1", "owner_id": "alice"},
        {"run_id": "run-1", "url": "http://127.0.0.1/private"},
        {"run_id": "running"},
    ],
)
def test_invalid_or_running_request_never_reaches_regimex(
    gateway: GatewayContext,
    payload: dict[str, object],
) -> None:
    response = gateway.client.post(
        "/api/simulations/configuration-1/regime-analyses",
        json=payload,
        headers=authenticated(gateway),
    )
    assert response.status_code in {409, 422} and gateway.upstream.requests == []


def test_upstream_failure_preserves_existing_simulator_results(gateway: GatewayContext) -> None:
    response = gateway.client.post(
        "/api/simulations/configuration-1/regime-analyses",
        json={"run_id": "run-1"},
        headers=authenticated(gateway),
    )
    assert response.status_code == 503 and response.json()["error"]["retryable"]
    assert FIXTURE_TOKEN not in response.text
    result = gateway.client.get(
        "/api/simulations/configuration-1/results?run_id=run-1",
        headers=authenticated(gateway),
    )
    assert result.status_code == 200 and result.json() == {"trades": [1]}


def test_gateway_rejects_analysis_bound_to_another_snapshot(gateway: GatewayContext) -> None:
    analysis_id, snapshot_id = str(uuid4()), str(uuid4())

    def respond(handler: BaseHTTPRequestHandler) -> None:
        value = (
            {
                "analysis_id": analysis_id,
                "snapshot_id": snapshot_id,
                "job": {
                    "operation_id": str(uuid4()),
                    "resource_id": analysis_id,
                    "kind": "analysis",
                    "state": "succeeded",
                },
            }
            if "/analyses/" in handler.path
            else {
                "snapshot_id": snapshot_id,
                "source_id": "test-simulator",
                "run_id": "run-1",
                "source_snapshot_id": "snapshot-1",
                "content_sha256": "0" * 64,
            }
        )
        handler.send_response(200)
        handler.send_header("Content-Type", "application/json")
        handler.end_headers()
        handler.wfile.write(json_bytes(value))

    gateway.upstream.responder = respond
    response = gateway.client.get(
        f"/api/simulations/configuration-1/regime-analyses/{analysis_id}?run_id=run-1",
        headers=authenticated(gateway),
    )
    assert response.status_code == 502 and response.json()["error"]["retryable"]
