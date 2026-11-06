"""Integration identity and credential lifecycle over real PostgreSQL transactions."""

import hashlib
import os
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import timedelta
from uuid import UUID, uuid4

import pytest

pytest.importorskip("psycopg", reason="install the service extra")
pytest.importorskip("alembic", reason="install the service extra")
pytest.importorskip("procrastinate", reason="install the service extra")

import psycopg
from alembic import command
from alembic.config import Config
from pydantic import ValidationError

from regimex.access_contracts import IntegrationCreate, IntegrationKeyCreate, IntegrationUpdate
from regimex.data_catalog import content_hash
from regimex.infrastructure import identity_namespaces, integration_repository
from regimex.infrastructure.auth_repository import AuthRepository
from regimex.infrastructure.database import DatabaseRepository
from regimex.infrastructure.integration_limits import (
    IntegrationLimitsRepository,
    IntegrationRateLimited,
)
from regimex.infrastructure.integration_repository import IntegrationRepository, require_integration
from regimex.service_contracts import (
    CapacityExceeded,
    Conflict,
    Forbidden,
    NotFound,
    Principal,
    Unauthorized,
)


@dataclass
class AccessContext:
    database: DatabaseRepository
    repository: IntegrationRepository
    admin: Principal
    user: Principal
    integration_id: UUID


@pytest.fixture
def database() -> Iterator[DatabaseRepository]:
    dsn = os.environ.get("REGIMEX_TEST_DATABASE_DSN")
    if not dsn:
        pytest.skip("set REGIMEX_TEST_DATABASE_DSN to a disposable PostgreSQL database")
    repository = DatabaseRepository(dsn, schema="regimex_access_" + uuid4().hex)
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


@pytest.fixture
def access(database: DatabaseRepository) -> AccessContext:
    admin, _ = database.issue_token("operator", "admin")
    user, _ = database.issue_token("researcher", "user")
    repository = IntegrationRepository(database)
    integration = repository.create(
        admin,
        IntegrationCreate(
            owner_id="platform", source_namespace="simulator", display_name="Research platform"
        ),
    )
    return AccessContext(database, repository, admin, user, integration.integration_id)


def test_managed_keys_are_bounded_hashed_and_explicitly_scoped(access: AccessContext) -> None:
    issued = access.repository.issue_key(
        access.admin,
        access.integration_id,
        IntegrationKeyCreate(name="Results reader", lifetime_days=7, scopes=("read",)),
    )
    raw = issued.token.get_secret_value()
    assert raw not in repr(issued)
    principal = access.database.authenticate(raw)
    assert principal.integration_id == access.integration_id
    assert principal.integration_scopes == ("read",)
    assert issued.key.expires_at is not None
    assert issued.key.expires_at - issued.key.created_at == timedelta(days=7)
    views = access.repository.list_keys(access.admin, access.integration_id)
    assert len(views) == 1 and views[0].last_used_at is not None
    assert raw not in views[0].model_dump_json()
    with access.database.connection() as connection:
        assert connection.execute(
            "SELECT token_sha256 FROM regimex_tokens WHERE token_id=%s", (principal.token_id,)
        ).fetchone() == (hashlib.sha256(raw.encode("ascii")).hexdigest(),)
    events = access.repository.events(access.admin, access.integration_id)
    assert [event.action for event in events] == ["key.created", "integration.created"]
    assert all(raw not in event.model_dump_json() for event in events)
    assert events[0].actor_token_id == access.admin.token_id


def test_rotation_pause_and_resume_preserve_delegated_owners(access: AccessContext) -> None:
    auth = AuthRepository(access.database)
    first = access.repository.issue_key(
        access.admin, access.integration_id, IntegrationKeyCreate(name="Current")
    )
    second = access.repository.issue_key(
        access.admin, access.integration_id, IntegrationKeyCreate(name="Replacement")
    )
    identity = auth.bind_subject(first.token.get_secret_value(), "alice", "run-1")
    replacement = auth.bind_subject(second.token.get_secret_value(), "alice", "run-1")
    assert replacement.owner_id == identity.owner_id
    assert replacement.integration_id == identity.integration_id == access.integration_id
    access.repository.revoke_key(access.admin, access.integration_id, first.key.token_id)
    with pytest.raises(Unauthorized):
        auth.bind_subject(first.token.get_secret_value(), "alice", "run-1")
    assert auth.bind_subject(second.token.get_secret_value(), "alice", "run-1") == replacement
    paused = access.repository.update(
        access.admin, access.integration_id, IntegrationUpdate(expected_revision=0, enabled=False)
    )
    assert not paused.enabled
    paused_event = access.repository.events(access.admin, access.integration_id)[0]
    assert paused_event.enabled is False and paused_event.display_name == "Research platform"
    with pytest.raises(Unauthorized):
        access.database.authenticate(second.token.get_secret_value())
    with pytest.raises(Unauthorized):
        auth.bind_subject(second.token.get_secret_value(), "alice", "run-1")
    with pytest.raises(Conflict, match="Enable"):
        access.repository.issue_key(
            access.admin, access.integration_id, IntegrationKeyCreate(name="Forbidden while paused")
        )
    with pytest.raises(Conflict, match="Enable"):
        access.database.create_token("platform", "integration", source_namespace="simulator")
    access.repository.update(
        access.admin,
        access.integration_id,
        IntegrationUpdate(expected_revision=paused.revision, enabled=True),
    )
    assert auth.bind_subject(second.token.get_secret_value(), "alice", "run-1") == replacement
    assert access.repository.events(access.admin, access.integration_id)[0].enabled is True


def test_expired_keys_fail_both_bearer_and_subject_authentication(access: AccessContext) -> None:
    issued = access.repository.issue_key(
        access.admin, access.integration_id, IntegrationKeyCreate(name="Expiring")
    )
    with access.database.connection() as connection:
        connection.execute(
            "UPDATE regimex_tokens SET created_at=clock_timestamp()-interval '2 days',"
            "expires_at=clock_timestamp()-interval '1 day' WHERE token_id=%s",
            (issued.key.token_id,),
        )
    with pytest.raises(Unauthorized):
        access.database.authenticate(issued.token.get_secret_value())
    with pytest.raises(Unauthorized):
        AuthRepository(access.database).bind_subject(
            issued.token.get_secret_value(), "alice", "run-1"
        )


def test_intake_revalidates_key_scope_and_owner_binding(access: AccessContext) -> None:
    issued = access.repository.issue_key(
        access.admin,
        access.integration_id,
        IntegrationKeyCreate(name="Read only", scopes=("read",)),
    )
    principal = AuthRepository(access.database).bind_subject(
        issued.token.get_secret_value(), "alice", "run-1"
    )
    with access.database.connection() as connection:
        require_integration(connection, principal, "read")
        with pytest.raises(Forbidden):
            require_integration(connection, principal, "upload")
        with pytest.raises(Forbidden):
            require_integration(
                connection, principal.model_copy(update={"owner_id": "somebody-else"}), "read"
            )
    access.repository.revoke_key(access.admin, access.integration_id, issued.key.token_id)
    with access.database.connection() as connection, pytest.raises(Unauthorized):
        require_integration(connection, principal, "read")


def test_management_checks_current_administrator_credential(access: AccessContext) -> None:
    forged = access.user.model_copy(update={"role": "admin"})
    with pytest.raises(Forbidden):
        access.repository.issue_key(
            forged, access.integration_id, IntegrationKeyCreate(name="Escalated")
        )
    with pytest.raises(Forbidden):
        access.repository.get(access.user, uuid4())
    with pytest.raises(Forbidden):
        access.repository.list(access.user)
    access.database.revoke_token(access.admin.token_id)
    with pytest.raises(Unauthorized):
        access.repository.update(
            access.admin,
            access.integration_id,
            IntegrationUpdate(expected_revision=0, enabled=False),
        )
    with access.database.connection() as connection:
        assert connection.execute(
            "SELECT enabled,revision FROM regimex_integrations WHERE integration_id=%s",
            (access.integration_id,),
        ).fetchone() == (True, 0)


def test_key_revocation_checks_integration_and_is_idempotent(access: AccessContext) -> None:
    other = access.repository.create(
        access.admin,
        IntegrationCreate(
            owner_id="other-platform", source_namespace="other", display_name="Other"
        ),
    )
    issued = access.repository.issue_key(
        access.admin, access.integration_id, IntegrationKeyCreate(name="Private")
    )
    with pytest.raises(NotFound):
        access.repository.revoke_key(access.admin, other.integration_id, issued.key.token_id)
    access.database.authenticate(issued.token.get_secret_value())
    access.repository.revoke_key(access.admin, access.integration_id, issued.key.token_id)
    access.repository.revoke_key(access.admin, access.integration_id, issued.key.token_id)
    assert (
        sum(
            event.action == "key.revoked"
            for event in access.repository.events(access.admin, access.integration_id)
        )
        == 1
    )


def test_concurrent_issuance_respects_active_key_bound(
    access: AccessContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(integration_repository, "MAX_ACTIVE_KEYS", 2)

    def issue(index: int) -> bool:
        try:
            access.repository.issue_key(
                access.admin, access.integration_id, IntegrationKeyCreate(name=f"Key {index}")
            )
            return True
        except CapacityExceeded:
            return False

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = tuple(executor.map(issue, range(4)))
    assert sum(results) == 2
    assert len(access.repository.list_keys(access.admin, access.integration_id)) == 2


def test_local_provisioning_keeps_one_integration_and_expiring_keys(
    access: AccessContext,
) -> None:
    def provision(_index: int) -> str:
        return access.database.create_token(
            "new-platform", "integration", source_namespace="source"
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        tokens = tuple(executor.map(provision, range(2)))
    identities = tuple(access.database.authenticate(token) for token in tokens)
    assert identities[0].integration_id == identities[1].integration_id
    integration_id = identities[0].integration_id
    assert integration_id is not None
    keys = access.repository.list_keys(access.admin, integration_id)
    assert len(keys) == 2 and all(key.expires_at is not None for key in keys)
    access.database.revoke_token(identities[0].token_id)
    assert access.repository.events(access.admin, integration_id)[0].action == "key.revoked"


def test_concurrent_updates_reject_stale_revision(access: AccessContext) -> None:
    def update(index: int) -> bool:
        try:
            access.repository.update(
                access.admin,
                access.integration_id,
                IntegrationUpdate(expected_revision=0, display_name=f"Platform {index}"),
            )
            return True
        except Conflict:
            return False

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(update, range(2)))
    assert sum(results) == 1
    assert access.repository.get(access.admin, access.integration_id).revision == 1


def test_management_history_has_a_bounded_retention_window(
    access: AccessContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(integration_repository, "MAX_ACCESS_EVENTS", 3)
    for revision in range(5):
        access.repository.update(
            access.admin,
            access.integration_id,
            IntegrationUpdate(expected_revision=revision, display_name=f"Platform {revision}"),
        )
    events = access.repository.events(access.admin, access.integration_id)
    assert len(events) == 3 and all(event.action == "integration.updated" for event in events)
    assert [event.display_name for event in events] == ["Platform 4", "Platform 3", "Platform 2"]
    assert (
        len(access.repository.events(access.admin, access.integration_id, limit=1, offset=1)) == 1
    )


def test_integration_accounts_cannot_impersonate_local_or_delegated_owners(
    access: AccessContext,
) -> None:
    for owner in (access.user.owner_id, access.admin.owner_id):
        with pytest.raises(Conflict, match="reserved"):
            access.repository.create(
                access.admin,
                IntegrationCreate(owner_id=owner, source_namespace="private", display_name="Fake"),
            )
    access.database.revoke_token(access.user.token_id)
    with pytest.raises(Conflict, match="reserved"):
        access.database.create_token(
            access.user.owner_id, "integration", source_namespace="private"
        )
    key = access.repository.issue_key(
        access.admin, access.integration_id, IntegrationKeyCreate(name="Backend")
    )
    delegated = AuthRepository(access.database).bind_subject(
        key.token.get_secret_value(), "alice", "run-1"
    )
    with pytest.raises(Conflict, match="reserved"):
        access.repository.create(
            access.admin,
            IntegrationCreate(
                owner_id=delegated.owner_id, source_namespace="simulator", display_name="Fake"
            ),
        )
    for owner in ("platform", delegated.owner_id):
        with pytest.raises(Conflict, match="reserved"):
            access.database.create_token(owner, "user")
    second_source = access.repository.create(
        access.admin,
        IntegrationCreate(owner_id="platform", source_namespace="second", display_name="Second"),
    )
    assert second_source.owner_id == "platform"
    assert second_source.integration_id != access.integration_id


def test_retained_resource_owners_cannot_be_claimed_as_integration_accounts(
    access: AccessContext,
) -> None:
    from test_service_database import _snapshot

    _snapshot(access.database, owner="retained-owner")
    with pytest.raises(Conflict, match="retained resources"):
        access.repository.create(
            access.admin,
            IntegrationCreate(
                owner_id="retained-owner", source_namespace="local-upload", display_name="Fake"
            ),
        )


def test_concurrent_local_and_integration_provisioning_reserves_one_namespace(
    access: AccessContext,
) -> None:
    def create(integration: bool) -> bool:
        try:
            if integration:
                access.repository.create(
                    access.admin,
                    IntegrationCreate(
                        owner_id="contested-owner", source_namespace="simulator", display_name="New"
                    ),
                )
            else:
                access.database.create_token("contested-owner", "user")
            return True
        except Conflict:
            return False

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(create, (True, False)))
    assert sum(results) == 1


def test_new_subject_identity_skips_existing_account_names(
    access: AccessContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collision, fresh = uuid4(), uuid4()
    access.database.create_token("external_" + collision.hex, "user")
    candidates = iter((collision, fresh))
    monkeypatch.setattr(identity_namespaces, "uuid4", lambda: next(candidates))
    key = access.repository.issue_key(
        access.admin, access.integration_id, IntegrationKeyCreate(name="Backend")
    )
    principal = AuthRepository(access.database).bind_subject(
        key.token.get_secret_value(), "alice", "run-1"
    )
    assert principal.owner_id == "external_" + fresh.hex


def test_delegated_denials_attribute_only_verified_credential_ids(access: AccessContext) -> None:
    key = access.repository.issue_key(
        access.admin,
        access.integration_id,
        IntegrationKeyCreate(name="Reader", scopes=("read",)),
    )
    auth = AuthRepository(access.database)
    with pytest.raises(Forbidden) as denied:
        auth.bind_subject(key.token.get_secret_value(), "alice", "run-1", required_scope="upload")
    assert denied.value.integration_id == access.integration_id
    assert denied.value.token_id == key.key.token_id
    limits = IntegrationLimitsRepository(
        access.database, requests_per_minute=1, subject_requests_per_minute=1
    )
    auth.bind_subject(key.token.get_secret_value(), "alice", "run-1", limits=limits)
    with pytest.raises(IntegrationRateLimited) as limited:
        auth.bind_subject(key.token.get_secret_value(), "alice", "run-1", limits=limits)
    assert limited.value.integration_id == access.integration_id
    assert limited.value.token_id == key.key.token_id
    with pytest.raises(Unauthorized) as unknown:
        auth.bind_subject("rgx_" + "a" * 43, "alice", "run-1")
    assert unknown.value.integration_id is None and unknown.value.token_id is None


@pytest.mark.parametrize(
    "changes",
    [
        {"lifetime_days": 0},
        {"lifetime_days": 366},
        {"scopes": ()},
        {"scopes": ("read", "read")},
        {"scopes": ("admin",)},
        {"name": "\n"},
    ],
)
def test_invalid_key_policies_are_rejected(changes: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        IntegrationKeyCreate.model_validate({"name": "Key", **changes})


@pytest.mark.parametrize("collision", [None, "local", "delegated"])
def test_upgrade_preserves_bindings_and_rejects_ambiguous_owner_namespaces(
    monkeypatch: pytest.MonkeyPatch,
    collision: str | None,
) -> None:
    dsn = os.environ.get("REGIMEX_TEST_DATABASE_DSN")
    if not dsn:
        pytest.skip("set REGIMEX_TEST_DATABASE_DSN to a disposable PostgreSQL database")
    database = DatabaseRepository(dsn, schema="regimex_access_upgrade_" + uuid4().hex)
    original_upgrade = command.upgrade

    def before_access(config: Config, revision: str) -> None:
        original_upgrade(config, "0003_experiment_names" if revision == "head" else revision)

    raw_token = "rgx_" + "a" * 43
    token_id = uuid4()
    try:
        with monkeypatch.context() as patch:
            patch.setattr(command, "upgrade", before_access)
            database.migrate()
        with database.connection() as connection:
            connection.execute(
                "INSERT INTO regimex_tokens "
                "(token_id,token_sha256,owner_id,role,source_namespace,created_at) "
                "VALUES (%s,%s,'platform','integration','simulator',clock_timestamp())",
                (token_id, hashlib.sha256(raw_token.encode("ascii")).hexdigest()),
            )
            connection.execute(
                "INSERT INTO regimex_integration_subjects "
                "(integration_owner_id,source_namespace,subject_sha256,effective_owner_id) "
                "VALUES ('platform','simulator',%s,'external_original')",
                (content_hash({"version": "external-subject/1.0", "subject": "alice"}),),
            )
            if collision is not None:
                connection.execute(
                    "INSERT INTO regimex_tokens "
                    "(token_id,token_sha256,owner_id,role,created_at) "
                    "VALUES (%s,%s,%s,'user',clock_timestamp())",
                    (
                        uuid4(),
                        "b" * 64,
                        "platform" if collision == "local" else "external_original",
                    ),
                )
        if collision is not None:
            with pytest.raises(RuntimeError, match="account identities overlap"):
                database.migrate()
            with database.connection() as connection:
                assert connection.execute("SELECT version_num FROM alembic_version").fetchone() == (
                    "0003_experiment_names",
                )
                assert connection.execute(
                    "SELECT effective_owner_id FROM regimex_integration_subjects"
                ).fetchone() == ("external_original",)
            return
        database.migrate()
        assert database.readiness()
        principal = AuthRepository(database).bind_subject(raw_token, "alice", "run-1")
        assert principal.owner_id == "external_original"
        assert principal.integration_id is not None
        assert principal.integration_scopes == ("read", "upload", "analyze")
        assert principal.token_id == token_id
        with database.connection() as connection:
            assert connection.execute(
                "SELECT expires_at FROM regimex_tokens WHERE token_id=%s", (token_id,)
            ).fetchone() == (None,)
    finally:
        with psycopg.connect(dsn, autocommit=True) as connection:
            connection.execute(
                psycopg.sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                    psycopg.sql.Identifier(database.schema)
                )
            )
