"""Real PostgreSQL sessions and browser-visible cookie/CSRF behavior."""

import hashlib
import os
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from uuid import uuid4

import pytest

pytest.importorskip("psycopg", reason="install the service extra")
pytest.importorskip("alembic", reason="install the service extra")
pytest.importorskip("procrastinate", reason="install the service extra")
pytest.importorskip("httpx2", reason="install the service-test group")

import psycopg
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from pydantic import ValidationError

from regimex.application.auth import AuthService
from regimex.auth_api import install_session_routes, request_auth
from regimex.auth_contracts import AuthPolicy, RequestAuth
from regimex.infrastructure.auth_repository import AuthRepository
from regimex.infrastructure.database import DatabaseRepository
from regimex.service_contracts import (
    CapacityExceeded,
    Forbidden,
    Principal,
    ServiceError,
    Unauthorized,
)

ORIGIN = "https://regimex.example"


@dataclass
class AuthContext:
    database: DatabaseRepository
    repository: AuthRepository
    service: AuthService
    principal: Principal
    token: str


@pytest.fixture
def context() -> Iterator[AuthContext]:
    dsn = os.environ.get("REGIMEX_TEST_DATABASE_DSN")
    if not dsn:
        pytest.skip("set REGIMEX_TEST_DATABASE_DSN to a disposable PostgreSQL database")
    database = DatabaseRepository(dsn, schema="regimex_auth_" + uuid4().hex)
    try:
        database.migrate()
        principal, token = database.issue_token("browser-owner", "user")
        repository = AuthRepository(database)
        yield AuthContext(
            database,
            repository,
            AuthService(repository, AuthPolicy(public_origin=ORIGIN)),
            principal,
            token,
        )
    finally:
        with psycopg.connect(dsn, autocommit=True) as connection:
            connection.execute(
                psycopg.sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                    psycopg.sql.Identifier(database.schema)
                )
            )


def request(method: str = "POST", **headers: str) -> RequestAuth:
    return RequestAuth(
        method=method,
        scheme="https",
        authority="regimex.example",
        headers=tuple(
            (key.replace("_", "-").encode(), value.encode("latin-1"))
            for key, value in headers.items()
        ),
    )


def app(context: AuthContext, service: AuthService | None = None) -> FastAPI:
    authentication = service or context.service
    application = FastAPI()
    install_session_routes(application, authentication)

    @application.exception_handler(ServiceError)
    def error(_request: Request, failure: ServiceError) -> JSONResponse:
        return JSONResponse(
            {"error": {"code": failure.code, "message": failure.message}},
            status_code=failure.status_code,
            headers={"Cache-Control": "private, no-store"},
        )

    @application.api_route("/protected", methods=["GET", "POST"])
    def protected(incoming: Request) -> dict[str, str]:
        principal = authentication.resolve(request_auth(incoming))
        return {"owner_id": principal.owner_id, "role": principal.role}

    return application


def test_secure_cookie_exchange_reload_csrf_logout_and_parent_revocation(
    context: AuthContext,
) -> None:
    with TestClient(app(context), base_url=ORIGIN) as client:
        login = client.post(
            "/api/v1/session", json={"token": context.token}, headers={"Origin": ORIGIN}
        )
        assert login.status_code == 200
        assert login.headers["cache-control"] == "private, no-store"
        cookie_header = login.headers["set-cookie"]
        assert cookie_header.startswith("__Host-regimex_session=")
        assert all(
            item in cookie_header.lower()
            for item in ("httponly", "secure", "samesite=strict", "path=/")
        )
        assert "domain=" not in cookie_header.lower()
        view = login.json()
        assert set(view) == {"principal", "csrf_token", "expires_at"}
        assert view["principal"] == {"owner_id": context.principal.owner_id, "role": "user"}
        raw_session = client.cookies.get(context.service.cookie_name)
        assert raw_session and raw_session not in login.text and context.token not in login.text
        with context.database.connection() as connection:
            row = connection.execute(
                "SELECT row_to_json(s)::text FROM regimex_sessions s"
            ).fetchone()
            assert (
                row is not None
                and raw_session not in str(row[0])
                and context.token not in str(row[0])
            )
            assert hashlib.sha256(raw_session.encode()).hexdigest() in str(row[0])
        assert client.get("/api/v1/session").json() == view
        assert client.get("/protected").status_code == 200
        assert client.post("/protected", headers={"Origin": ORIGIN}).status_code == 403
        assert (
            client.post(
                "/protected",
                headers={"Origin": "https://evil.example", "X-CSRF-Token": view["csrf_token"]},
            ).status_code
            == 403
        )
        assert (
            client.post(
                "/protected", headers={"Origin": ORIGIN, "X-CSRF-Token": view["csrf_token"]}
            ).status_code
            == 200
        )
        assert client.delete("/api/v1/session", headers={"Origin": ORIGIN}).status_code == 403
        logout = client.delete(
            "/api/v1/session", headers={"Origin": ORIGIN, "X-CSRF-Token": view["csrf_token"]}
        )
        assert logout.status_code == 204 and "Max-Age=0" in logout.headers["set-cookie"]
        assert client.get("/protected").status_code == 401
        with context.database.connection() as connection:
            assert connection.execute("SELECT count(*) FROM regimex_sessions").fetchone() == (0,)
        assert (
            client.post(
                "/api/v1/session", json={"token": context.token}, headers={"Origin": ORIGIN}
            ).status_code
            == 200
        )
        context.database.revoke_token(context.principal.token_id)
        assert client.get("/protected").status_code == 401
        invalid = client.get("/api/v1/session")
        assert invalid.status_code == 401 and "Max-Age=0" in invalid.headers["set-cookie"]
        assert client.cookies.get(context.service.cookie_name) is None


@pytest.mark.parametrize(
    "origin", [None, "null", "https://regimex.example.evil", "http://regimex.example"]
)
def test_exchange_requires_exact_origin_before_issuing_any_session(
    context: AuthContext, origin: str | None
) -> None:
    with TestClient(app(context), base_url=ORIGIN) as client:
        response = client.post(
            "/api/v1/session",
            json={"token": context.token},
            headers={} if origin is None else {"Origin": origin},
        )
        assert response.status_code == 403
    with context.database.connection() as connection:
        assert connection.execute("SELECT count(*) FROM regimex_sessions").fetchone() == (0,)


def test_session_absolute_expiry_is_enforced_and_expired_rows_are_reclaimed(
    context: AuthContext,
) -> None:
    issued = context.repository.issue_session(context.token, maximum_sessions=1)
    with context.database.connection() as connection:
        connection.execute(
            "UPDATE regimex_sessions SET created_at=clock_timestamp()-interval '2 minutes', "
            "expires_at=clock_timestamp()-interval '1 minute' WHERE session_id=%s",
            (issued.record.session_id,),
        )
    with pytest.raises(Unauthorized):
        context.repository.authenticate_session(issued.cookie_value.get_secret_value())
    replacement = context.repository.issue_session(context.token, maximum_sessions=1)
    assert replacement.record.session_id != issued.record.session_id
    with context.database.connection() as connection:
        assert connection.execute("SELECT count(*) FROM regimex_sessions").fetchone() == (1,)


def test_concurrent_session_issue_cannot_exceed_parent_bound(context: AuthContext) -> None:
    def issue(_index: int) -> bool:
        try:
            context.repository.issue_session(context.token, maximum_sessions=1)
        except CapacityExceeded:
            return False
        return True

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert sum(pool.map(issue, range(4))) == 1
    with context.database.connection() as connection:
        assert connection.execute("SELECT count(*) FROM regimex_sessions").fetchone() == (1,)


def test_bearer_cookie_duplicate_credentials_and_exchange_ambiguity_are_rejected(
    context: AuthContext,
) -> None:
    issued = context.service.login(context.token, request(origin=ORIGIN))
    raw_session = issued.cookie_value.get_secret_value()
    with TestClient(app(context), base_url=ORIGIN) as client:
        both = {
            "Authorization": "Bearer " + context.token,
            "Cookie": context.service.cookie_name + "=" + raw_session,
        }
        assert client.get("/protected", headers=both).status_code == 400
        duplicated = [
            ("Authorization", "Bearer " + context.token),
            ("Authorization", "Bearer " + context.token),
        ]
        assert client.get("/protected", headers=duplicated).status_code == 400
        for header, value in (("Origin", ORIGIN), ("X-CSRF-Token", "a" * 64)):
            repeated = [
                ("Authorization", "Bearer " + context.token),
                (header, value),
                (header, value),
            ]
            assert client.get("/protected", headers=repeated).status_code == 400
        cookies = f"__Host-regimex_session={raw_session}; regimex_session={raw_session}"
        assert client.get("/protected", headers={"Cookie": cookies}).status_code == 400
        assert (
            client.post(
                "/api/v1/session", json={"token": context.token}, headers={"Origin": ORIGIN, **both}
            ).status_code
            == 400
        )
        body = ('{"token":"' + context.token + '","token":"' + context.token + '"}').encode()
        assert (
            client.post(
                "/api/v1/session",
                content=body,
                headers={"Origin": ORIGIN, "Content-Type": "application/json"},
            ).status_code
            == 400
        )
        assert (
            client.get(
                "/protected", headers={"Authorization": "Bearer " + context.token}
            ).status_code
            == 200
        )


def test_csrf_cannot_be_reused_across_sessions_and_unicode_header_fails_cleanly(
    context: AuthContext,
) -> None:
    first = context.service.login(context.token, request(origin=ORIGIN))
    second = context.service.login(context.token, request(origin=ORIGIN))
    assert first.view.csrf_token != second.view.csrf_token
    for csrf in (first.view.csrf_token, "é" * 64):
        with pytest.raises(Forbidden):
            context.service.resolve(
                request(
                    cookie=context.service.cookie_name
                    + "="
                    + second.cookie_value.get_secret_value(),
                    origin=ORIGIN,
                    x_csrf_token=csrf,
                )
            )


def test_integration_token_cannot_be_exchanged_for_browser_session(context: AuthContext) -> None:
    _principal, token = context.database.issue_token(
        "gateway", "integration", source_namespace="simulator"
    )
    with TestClient(app(context), base_url=ORIGIN) as client:
        assert (
            client.post(
                "/api/v1/session", json={"token": token}, headers={"Origin": ORIGIN}
            ).status_code
            == 403
        )


def test_https_is_default_and_loopback_http_requires_explicit_allowance(
    context: AuthContext,
) -> None:
    for origin in ("http://regimex.example", "http://127.0.0.1:8000", "http://localhost:8000"):
        with pytest.raises(ValidationError):
            AuthPolicy(public_origin=origin)
    with pytest.raises(ValidationError):
        AuthPolicy(public_origin="http://localhost:8000", allow_loopback_http=True)
    local_origin = "http://127.0.0.1:8000"
    policy = AuthPolicy(public_origin=local_origin, allow_loopback_http=True)
    local = AuthService(context.repository, policy)
    with TestClient(app(context, local), base_url=local_origin) as client:
        response = client.post(
            "/api/v1/session", json={"token": context.token}, headers={"Origin": local_origin}
        )
        assert response.status_code == 200
        assert response.headers["set-cookie"].startswith("regimex_session=")
        assert "; secure" not in response.headers["set-cookie"].lower()
    with TestClient(app(context), base_url="http://regimex.example") as client:
        assert (
            client.post(
                "/api/v1/session", json={"token": context.token}, headers={"Origin": ORIGIN}
            ).status_code
            == 403
        )
