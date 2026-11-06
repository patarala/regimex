"""Persistent integration budgets under concurrent requests and queue intake."""

from __future__ import annotations

import os
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest
from pydantic import SecretStr, ValidationError

from regimex.auth_contracts import AuthPolicy
from regimex.infrastructure.object_store import ObjectInfo
from regimex.service_contracts import CapacityExceeded, Forbidden, Principal, SnapshotRecord
from regimex.service_settings import ServiceSettings

if TYPE_CHECKING:
    from regimex.infrastructure.database import DatabaseRepository
    from regimex.infrastructure.integration_limits import IntegrationLimitsRepository


@pytest.fixture
def database() -> Iterator[DatabaseRepository]:
    dsn = os.environ.get("REGIMEX_TEST_DATABASE_DSN")
    if not dsn:
        pytest.skip("set REGIMEX_TEST_DATABASE_DSN to a disposable PostgreSQL database")
    psycopg = pytest.importorskip("psycopg", reason="install the service extra")
    pytest.importorskip("alembic", reason="install the service extra")
    pytest.importorskip("procrastinate", reason="install the service extra")
    from regimex.infrastructure.database import DatabaseRepository

    repository = DatabaseRepository(dsn, schema="regimex_limits_" + uuid4().hex)
    try:
        repository.migrate()
        yield repository
    finally:
        with psycopg.connect(dsn, autocommit=True) as connection:
            connection.execute(
                psycopg.sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                    psycopg.sql.Identifier(repository.schema)
                )
            )


def _identity(
    database: DatabaseRepository,
    subject: str = "alice",
    *,
    owner: str = "simulator",
    source: str = "simulator",
    token: str | None = None,
    run: str = "run-1",
) -> tuple[Principal, str]:
    from regimex.infrastructure.auth_repository import AuthRepository

    token = token or database.create_token(owner, "integration", source_namespace=source)
    return AuthRepository(database).bind_subject(token, subject, run), token


def _limits(
    database: DatabaseRepository, total: int = 10, subject: int = 5
) -> IntegrationLimitsRepository:
    from regimex.infrastructure.integration_limits import IntegrationLimitsRepository

    return IntegrationLimitsRepository(
        database, requests_per_minute=total, subject_requests_per_minute=subject
    )


def _snapshot(database: DatabaseRepository, principal: Principal) -> SnapshotRecord:
    assert principal.source_namespace is not None
    return database.add_snapshot(
        SnapshotRecord(
            snapshot_id=uuid4(),
            owner_id=principal.owner_id,
            source_id=principal.source_namespace,
            run_id=principal.run_scope or "run-1",
            source_snapshot_id="snapshot-1",
            content_sha256="a" * 64,
            object=ObjectInfo(
                key="snapshots/" + principal.owner_id,
                sha256="a" * 64,
                size_bytes=1,
                version_id="fixture",
            ),
            created_at=datetime.now(UTC),
        )
    )


def _publish(database: DatabaseRepository) -> None:
    from test_service_database import _model

    model = _model(database)
    admin = database.authenticate(database.create_token("administrator", "admin"))
    database.activate_model(admin, model.model_version_id, expected_previous=None)


def test_request_budgets_are_shared_across_keys_and_isolated_between_connections(
    database: DatabaseRepository,
) -> None:
    from regimex.infrastructure.integration_limits import IntegrationRateLimited

    alice, token = _identity(database)
    bob, _ = _identity(database, "bob", token=token)
    limits = _limits(database, total=3, subject=2)
    limits.consume(alice)
    limits.consume(alice)
    with pytest.raises(IntegrationRateLimited, match="user") as failure:
        limits.consume(alice)
    assert 1 <= failure.value.retry_after_seconds <= 60
    limits.consume(bob)
    rotated, _ = _identity(database)
    assert rotated.owner_id == alice.owner_id
    assert rotated.integration_id == alice.integration_id
    with pytest.raises(IntegrationRateLimited, match="connection"):
        limits.consume(rotated)
    independent, _ = _identity(database, owner="second-platform")
    limits.consume(independent)


def test_request_windows_reset_without_growing_rows(database: DatabaseRepository) -> None:
    from regimex.infrastructure.integration_limits import IntegrationRateLimited

    principal, _ = _identity(database)
    limits = _limits(database, total=1, subject=1)
    limits.consume(principal)
    with pytest.raises(IntegrationRateLimited):
        limits.consume(principal)
    with database.connection() as connection:
        connection.execute(
            "UPDATE regimex_integration_request_windows "
            "SET started_at=clock_timestamp()-interval '61 seconds'"
        )
    limits.consume(principal)
    with database.connection() as connection:
        assert connection.execute(
            "SELECT count(*),sum(request_count) FROM regimex_integration_request_windows"
        ).fetchone() == (2, 2)


def test_concurrent_requests_cannot_overspend_shared_budget(database: DatabaseRepository) -> None:
    from regimex.infrastructure.integration_limits import IntegrationRateLimited

    first, token = _identity(database)
    identities = [first] + [
        _identity(database, f"subject-{index}", token=token)[0] for index in range(11)
    ]
    limits = _limits(database, total=4, subject=2)

    def request(principal: Principal) -> bool:
        try:
            limits.consume(principal)
            return True
        except IntegrationRateLimited:
            return False

    with ThreadPoolExecutor(max_workers=6) as executor:
        accepted = list(executor.map(request, identities))
    assert sum(accepted) == 4
    with database.connection() as connection:
        assert connection.execute(
            "SELECT count(*),sum(request_count) FROM regimex_integration_request_windows"
        ).fetchone() == (5, 8)


def test_rejected_concurrent_delegations_do_not_allocate_subjects(
    database: DatabaseRepository,
) -> None:
    from regimex.infrastructure.auth_repository import AuthRepository
    from regimex.infrastructure.integration_limits import IntegrationRateLimited

    token = database.create_token("simulator", "integration", source_namespace="simulator")
    limits = _limits(database, total=3, subject=2)

    def bind(index: int) -> bool:
        try:
            AuthRepository(database).bind_subject(
                token, f"subject-{index}", "run-1", required_scope="read", limits=limits
            )
            return True
        except IntegrationRateLimited:
            return False

    with ThreadPoolExecutor(max_workers=6) as executor:
        accepted = list(executor.map(bind, range(12)))
    assert sum(accepted) == 3
    with database.connection() as connection:
        assert connection.execute(
            "SELECT count(*) FROM regimex_integration_subjects"
        ).fetchone() == (3,)
        assert connection.execute(
            "SELECT count(*),sum(request_count) FROM regimex_integration_request_windows"
        ).fetchone() == (4, 6)


def test_scope_denial_cannot_allocate_a_subject_or_consume_request_budget(
    database: DatabaseRepository,
) -> None:
    from regimex.infrastructure.auth_repository import AuthRepository

    token = database.create_token("simulator", "integration", source_namespace="simulator")
    principal = database.authenticate(token)
    with database.connection() as connection:
        connection.execute(
            "UPDATE regimex_tokens SET scopes=ARRAY['read'] WHERE token_id=%s",
            (principal.token_id,),
        )
    with pytest.raises(Forbidden):
        AuthRepository(database).bind_subject(
            token, "new-subject", "run-1", required_scope="upload", limits=_limits(database)
        )
    with database.connection() as connection:
        assert connection.execute(
            "SELECT count(*) FROM regimex_integration_subjects"
        ).fetchone() == (0,)
        assert connection.execute(
            "SELECT count(*) FROM regimex_integration_request_windows"
        ).fetchone() == (0,)


def test_delegation_authentication_and_key_rotation_share_a_safe_lock_order(
    database: DatabaseRepository,
) -> None:
    from regimex.access_contracts import IntegrationKeyCreate
    from regimex.infrastructure.auth_repository import AuthRepository
    from regimex.infrastructure.integration_repository import IntegrationRepository

    token = database.create_token("simulator", "integration", source_namespace="simulator")
    principal = database.authenticate(token)
    assert principal.integration_id is not None
    integration_id = principal.integration_id
    admin = database.authenticate(database.create_token("admin", "admin"))
    limits = _limits(database, total=100, subject=100)

    def bind() -> None:
        for index in range(8):
            AuthRepository(database).bind_subject(
                token, f"subject-{index}", "run-1", required_scope="read", limits=limits
            )

    def authenticate() -> None:
        for _ in range(8):
            limits.consume(database.authenticate(token))

    def rotate() -> None:
        repository = IntegrationRepository(database)
        for index in range(8):
            issued = repository.issue_key(
                admin, integration_id, IntegrationKeyCreate(name=f"rotated-{index}")
            )
            repository.revoke_key(admin, integration_id, issued.key.token_id)

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(action) for action in (bind, authenticate, rotate)]
        for future in futures:
            future.result(timeout=20)
    with database.connection() as connection:
        assert connection.execute(
            "SELECT sum(request_count) FROM regimex_integration_request_windows"
        ).fetchone() == (32,)


def test_unbound_keys_share_connection_budget_and_reject_fabricated_owners(
    database: DatabaseRepository,
) -> None:
    from regimex.infrastructure.integration_limits import IntegrationRateLimited

    delegated, token = _identity(database)
    unbound = database.authenticate(token)
    limits = _limits(database, total=2, subject=1)
    limits.consume(delegated)
    limits.consume(unbound)
    with pytest.raises(IntegrationRateLimited):
        limits.consume(unbound)
    forged = delegated.model_copy(update={"owner_id": "unregistered-owner"})
    with pytest.raises(Forbidden):
        limits.consume(forged)
    with database.connection() as connection:
        assert connection.execute(
            "SELECT count(*) FROM regimex_integration_request_windows"
        ).fetchone() == (3,)


@pytest.mark.parametrize("change", ["revoked", "expired", "disabled"])
def test_rate_budget_rechecks_previously_resolved_credentials(
    database: DatabaseRepository, change: str
) -> None:
    from regimex.service_contracts import Unauthorized

    principal, _ = _identity(database)
    with database.connection() as connection:
        if change == "disabled":
            connection.execute(
                "UPDATE regimex_integrations SET enabled=false WHERE integration_id=%s",
                (principal.integration_id,),
            )
        elif change == "expired":
            connection.execute(
                "UPDATE regimex_tokens SET created_at=clock_timestamp()-interval '2 days', "
                "expires_at=clock_timestamp()-interval '1 day' WHERE token_id=%s",
                (principal.token_id,),
            )
        else:
            database.revoke_token(principal.token_id)
    with pytest.raises(Unauthorized):
        _limits(database).consume(principal)


def test_pending_limit_groups_subjects_and_preserves_idempotent_replays(
    database: DatabaseRepository,
) -> None:
    _publish(database)
    alice, token = _identity(database)
    bob, _ = _identity(database, "bob", token=token)
    first_snapshot = _snapshot(database, alice)
    second_snapshot = _snapshot(database, bob)
    first = database.submit_analysis(
        alice,
        first_snapshot.snapshot_id,
        idempotency_key="first",
        request_sha256="a" * 64,
        integration_max_pending=1,
    )
    with pytest.raises(CapacityExceeded, match="integration connection"):
        database.submit_analysis(
            bob,
            second_snapshot.snapshot_id,
            idempotency_key="second",
            request_sha256="b" * 64,
            integration_max_pending=1,
        )
    rotated, _ = _identity(database)
    replay = database.submit_analysis(
        rotated,
        first_snapshot.snapshot_id,
        idempotency_key="first",
        request_sha256="a" * 64,
        integration_max_pending=1,
    )
    assert replay.analysis_id == first.analysis_id
    database.cancel_operation(alice, first.operation.operation_id)
    accepted = database.submit_analysis(
        bob,
        second_snapshot.snapshot_id,
        idempotency_key="second",
        request_sha256="b" * 64,
        integration_max_pending=1,
    )
    assert accepted.operation.state == "queued"


def test_pending_limit_is_atomic_across_subjects(database: DatabaseRepository) -> None:
    _publish(database)
    _, token = _identity(database)
    inputs = [_identity(database, f"subject-{index}", token=token)[0] for index in range(10)]
    snapshots = {item.owner_id: _snapshot(database, item) for item in inputs}

    def submit(principal: Principal) -> bool:
        try:
            database.submit_analysis(
                principal,
                snapshots[principal.owner_id].snapshot_id,
                idempotency_key="first",
                request_sha256="a" * 64,
                integration_max_pending=3,
            )
            return True
        except CapacityExceeded:
            return False

    with ThreadPoolExecutor(max_workers=5) as executor:
        accepted = list(executor.map(submit, inputs))
    assert sum(accepted) == 3
    with database.connection() as connection:
        assert connection.execute("SELECT count(*) FROM regimex_operations").fetchone() == (3,)
        assert connection.execute("SELECT count(*) FROM regimex_deliveries").fetchone() == (3,)


def test_pending_limit_counts_unbound_owner_and_keeps_other_source_independent(
    database: DatabaseRepository,
) -> None:
    _publish(database)
    delegated, token = _identity(database)
    unbound = database.authenticate(token)
    first = _snapshot(database, unbound)
    second = _snapshot(database, delegated)
    database.submit_analysis(
        unbound,
        first.snapshot_id,
        idempotency_key="first",
        request_sha256="a" * 64,
        integration_max_pending=1,
    )
    with pytest.raises(CapacityExceeded, match="integration connection"):
        database.submit_analysis(
            delegated,
            second.snapshot_id,
            idempotency_key="second",
            request_sha256="b" * 64,
            integration_max_pending=1,
        )
    other_source, _ = _identity(database, source="other-source")
    third = _snapshot(database, other_source)
    assert (
        database.submit_analysis(
            other_source,
            third.snapshot_id,
            idempotency_key="third",
            request_sha256="c" * 64,
            integration_max_pending=1,
        ).operation.state
        == "queued"
    )


def test_pending_intake_rechecks_permissions_and_revocation(database: DatabaseRepository) -> None:
    from regimex.service_contracts import Unauthorized

    _publish(database)
    principal, _ = _identity(database)
    snapshot = _snapshot(database, principal)
    with database.connection() as connection:
        connection.execute(
            "UPDATE regimex_tokens SET scopes=ARRAY['read'] WHERE token_id=%s",
            (principal.token_id,),
        )
    with pytest.raises(Forbidden):
        database.submit_analysis(
            principal, snapshot.snapshot_id, idempotency_key="denied", request_sha256="a" * 64
        )
    database.revoke_token(principal.token_id)
    with pytest.raises(Unauthorized):
        database.submit_analysis(
            principal, snapshot.snapshot_id, idempotency_key="denied", request_sha256="a" * 64
        )
    with database.connection() as connection:
        assert connection.execute("SELECT count(*) FROM regimex_deliveries").fetchone() == (0,)


def test_limit_configuration_propagates_and_rejects_ineffective_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("REGIMEX_DATABASE_DSN", "postgresql://example/test")
    monkeypatch.setenv("REGIMEX_INTEGRATION_REQUESTS_PER_MINUTE", "90")
    monkeypatch.setenv("REGIMEX_INTEGRATION_SUBJECT_REQUESTS_PER_MINUTE", "15")
    monkeypatch.setenv("REGIMEX_INTEGRATION_MAX_PENDING", "12")
    settings = ServiceSettings.from_env()
    assert settings.integration_max_pending == 12
    assert settings.auth_policy.integration_requests_per_minute == 90
    assert settings.auth_policy.integration_subject_requests_per_minute == 15
    with pytest.raises(ValidationError, match="cannot exceed"):
        AuthPolicy(
            public_origin="https://example.test",
            integration_requests_per_minute=10,
            integration_subject_requests_per_minute=11,
        )
    with pytest.raises(ValidationError):
        ServiceSettings(database_dsn=SecretStr("unused"), integration_max_pending=0)
