"""Transactional integration identities, credentials and access-management history."""

import hashlib
import re
import secrets
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, cast
from uuid import UUID, uuid4

import psycopg
from pydantic import SecretStr

from regimex.access_contracts import (
    AccessAction,
    AccessEventView,
    IntegrationCreate,
    IntegrationKeyCreate,
    IntegrationKeyView,
    IntegrationScope,
    IntegrationUpdate,
    IntegrationView,
    IssuedIntegrationKey,
)
from regimex.infrastructure.identity_namespaces import reserve_integration_owner
from regimex.service_contracts import (
    CapacityExceeded,
    Conflict,
    Forbidden,
    NotFound,
    Principal,
    Role,
    Unauthorized,
)

if TYPE_CHECKING:
    from regimex.infrastructure.database import DatabaseRepository

Connection = psycopg.Connection[tuple[object, ...]]
_TOKEN = re.compile(r"rgx_[A-Za-z0-9_-]{43}\Z")
_INTEGRATION_FIELDS = (
    "integration_id,owner_id,source_namespace,display_name,enabled,revision,created_at,updated_at"
)
_KEY_FIELDS = "token_id,integration_id,name,scopes,created_at,expires_at,last_used_at,revoked_at"
_EVENT_FIELDS = (
    "event_id,integration_id,created_at,actor_owner_id,actor_token_id,action,target_token_id,"
    "enabled,display_name"
)
MAX_ACTIVE_KEYS = 32
MAX_ACCESS_EVENTS = 10_000


def _integration(row: tuple[object, ...]) -> IntegrationView:
    return IntegrationView.model_validate(
        dict(zip(_INTEGRATION_FIELDS.split(","), row, strict=True))
    )


def _key(row: tuple[object, ...]) -> IntegrationKeyView:
    values = dict(zip(_KEY_FIELDS.split(","), row, strict=True))
    values["scopes"] = tuple(cast(list[str], values["scopes"]))
    return IntegrationKeyView.model_validate(values)


def _active_token(
    connection: Connection, clause: str, value: str | UUID, *, lock: bool = False
) -> Principal:
    # Callers select one of the two fixed predicates; no caller-supplied SQL enters here.
    if clause not in {"t.token_sha256=%s", "t.token_id=%s"}:
        raise ValueError("Unsupported credential lookup.")
    row = connection.execute(
        "SELECT t.token_id,t.owner_id,t.role,t.source_namespace,t.integration_id,t.scopes "
        "FROM regimex_tokens t LEFT JOIN regimex_integrations i "
        "ON t.integration_id=i.integration_id WHERE " + clause + " AND t.revoked_at IS NULL "
        "AND (t.expires_at IS NULL OR t.expires_at>clock_timestamp()) "
        "AND (t.role<>'integration' OR (i.enabled AND i.owner_id=t.owner_id "
        "AND i.source_namespace=t.source_namespace))" + (" FOR SHARE OF t" if lock else ""),
        (value,),
    ).fetchone()
    if row is None:
        raise Unauthorized()
    return Principal(
        token_id=cast(UUID, row[0]),
        owner_id=cast(str, row[1]),
        role=cast(Role, row[2]),
        source_namespace=cast(str | None, row[3]),
        integration_id=cast(UUID | None, row[4]),
        integration_scopes=tuple(cast(list[IntegrationScope], row[5])),
    )


def authenticate_token(
    connection: Connection, raw_token: str, *, lock: bool = False, touch: bool = True
) -> Principal:
    """Resolve the current token and integration state in the caller's transaction."""
    if _TOKEN.fullmatch(raw_token) is None:
        raise Unauthorized()
    principal = _active_token(
        connection, "t.token_sha256=%s", hashlib.sha256(raw_token.encode("ascii")).hexdigest()
    )
    if principal.role == "integration" and (touch or lock):
        connection.execute(
            "SELECT integration_id FROM regimex_integrations WHERE integration_id=%s FOR SHARE",
            (principal.integration_id,),
        ).fetchone()
    if (principal.role == "integration" and touch) or lock:
        # Session issuance needs exclusive parent locking for its concurrent-session bound.
        row = connection.execute(
            "SELECT token_id FROM regimex_tokens WHERE token_id=%s "
            "AND revoked_at IS NULL AND (expires_at IS NULL OR expires_at>clock_timestamp()) "
            "FOR UPDATE",
            (principal.token_id,),
        ).fetchone()
        if row is None:
            raise Unauthorized()
        principal = _active_token(connection, "t.token_id=%s", principal.token_id)
    if principal.role == "integration" and touch:
        connection.execute(
            "UPDATE regimex_tokens SET last_used_at=clock_timestamp() WHERE token_id=%s",
            (principal.token_id,),
        )
    return principal


def require_integration(
    connection: Connection, principal: Principal, scope: IntegrationScope | None = None
) -> None:
    """Recheck delegated permissions while intake holds its integration lock."""
    current = _active_token(connection, "t.token_id=%s", principal.token_id)
    if (
        principal.role != "integration"
        or current.role != "integration"
        or principal.integration_id is None
        or current.integration_id != principal.integration_id
        or current.source_namespace != principal.source_namespace
    ):
        raise Forbidden()
    if scope is not None and scope not in current.integration_scopes:
        raise Forbidden("The integration key does not permit this action.")
    if principal.run_scope is None:
        if principal.owner_id != current.owner_id:
            raise Forbidden()
    elif (
        connection.execute(
            "SELECT 1 FROM regimex_integration_subjects WHERE integration_owner_id=%s "
            "AND source_namespace=%s AND effective_owner_id=%s",
            (current.owner_id, current.source_namespace, principal.owner_id),
        ).fetchone()
        is None
    ):
        raise Forbidden()


def _admin(connection: Connection, actor: Principal, *, lock: bool = False) -> None:
    if actor.role != "admin":
        raise Forbidden()
    current = _active_token(connection, "t.token_id=%s", actor.token_id, lock=lock)
    if current.role != "admin" or current.owner_id != actor.owner_id:
        raise Forbidden()


def _get(connection: Connection, integration_id: UUID, *, lock: bool = False) -> IntegrationView:
    row = connection.execute(
        "SELECT "
        + _INTEGRATION_FIELDS
        + " FROM regimex_integrations WHERE integration_id=%s"
        + (" FOR UPDATE" if lock else ""),
        (integration_id,),
    ).fetchone()
    if row is None:
        raise NotFound("Integration was not found.")
    return _integration(row)


def _event(
    connection: Connection,
    integration_id: UUID,
    action: AccessAction,
    actor: Principal | None,
    *,
    token_id: UUID | None = None,
) -> None:
    """Store fixed public identifiers only; retain at most 10,000 events per connection."""
    connection.execute(
        "INSERT INTO regimex_access_events "
        "(event_id,integration_id,actor_owner_id,actor_token_id,action,target_token_id,"
        "enabled,display_name) SELECT %s,%s,%s,%s,%s,%s,enabled,display_name "
        "FROM regimex_integrations WHERE integration_id=%s",
        (
            uuid4(),
            integration_id,
            None if actor is None else actor.owner_id,
            None if actor is None else actor.token_id,
            action,
            token_id,
            integration_id,
        ),
    )
    connection.execute(
        "DELETE FROM regimex_access_events WHERE event_id IN ("
        "SELECT event_id FROM regimex_access_events WHERE integration_id=%s "
        "ORDER BY created_at DESC,event_id DESC OFFSET %s)",
        (integration_id, MAX_ACCESS_EVENTS),
    )


def _issue_key(
    connection: Connection,
    integration: IntegrationView,
    request: IntegrationKeyCreate,
    actor: Principal | None,
) -> IssuedIntegrationKey:
    if not integration.enabled:
        raise Conflict("Enable the integration before issuing a key.")
    count = connection.execute(
        "SELECT count(*) FROM regimex_tokens WHERE integration_id=%s AND revoked_at IS NULL "
        "AND (expires_at IS NULL OR expires_at>clock_timestamp())",
        (integration.integration_id,),
    ).fetchone()
    assert count is not None
    if cast(int, count[0]) >= MAX_ACTIVE_KEYS:
        raise CapacityExceeded("The integration has reached its active key limit.")
    now_row = connection.execute("SELECT clock_timestamp()").fetchone()
    assert now_row is not None
    now = cast(datetime, now_row[0])
    token_id = uuid4()
    raw_token = "rgx_" + secrets.token_urlsafe(32)
    row = connection.execute(
        "INSERT INTO regimex_tokens "
        "(token_id,token_sha256,owner_id,role,source_namespace,created_at,"
        "integration_id,name,expires_at,scopes) "
        "VALUES (%s,%s,%s,'integration',%s,%s,%s,%s,%s,%s) RETURNING " + _KEY_FIELDS,
        (
            token_id,
            hashlib.sha256(raw_token.encode("ascii")).hexdigest(),
            integration.owner_id,
            integration.source_namespace,
            now,
            integration.integration_id,
            request.name,
            now + timedelta(days=request.lifetime_days),
            list(request.scopes),
        ),
    ).fetchone()
    assert row is not None
    _event(connection, integration.integration_id, "key.created", actor, token_id=token_id)
    return IssuedIntegrationKey(key=_key(row), token=SecretStr(raw_token))


def provision_integration_token(
    database: DatabaseRepository, owner_id: str, source_namespace: str
) -> str:
    """Local operator provisioning shares the same state and checks as managed keys."""
    request = IntegrationCreate(
        owner_id=owner_id, source_namespace=source_namespace, display_name=source_namespace[:120]
    )
    with database.connection() as connection:
        reserve_integration_owner(connection, owner_id)
        database._lock(connection, "integration-provision:" + owner_id + ":" + source_namespace)
        row = connection.execute(
            "SELECT integration_id FROM regimex_integrations WHERE owner_id=%s "
            "AND source_namespace=%s FOR UPDATE",
            (owner_id, source_namespace),
        ).fetchone()
        if row is None:
            integration_id = uuid4()
            inserted = connection.execute(
                "INSERT INTO regimex_integrations "
                "(integration_id,owner_id,source_namespace,display_name) VALUES (%s,%s,%s,%s) "
                "ON CONFLICT (owner_id,source_namespace) DO NOTHING RETURNING integration_id",
                (integration_id, request.owner_id, request.source_namespace, request.display_name),
            ).fetchone()
            if inserted is not None:
                _event(connection, integration_id, "integration.created", None)
            else:
                existing = connection.execute(
                    "SELECT integration_id FROM regimex_integrations WHERE owner_id=%s "
                    "AND source_namespace=%s FOR UPDATE",
                    (owner_id, source_namespace),
                ).fetchone()
                assert existing is not None
                integration_id = cast(UUID, existing[0])
        else:
            integration_id = cast(UUID, row[0])
        issued = _issue_key(
            connection,
            _get(connection, integration_id, lock=True),
            IntegrationKeyCreate(name="Provisioned key"),
            None,
        )
        return issued.token.get_secret_value()


def revoke_token(database: DatabaseRepository, token_id: UUID) -> None:
    """Local operator revocation also records integration-key changes."""
    with database.connection() as connection:
        row = connection.execute(
            "SELECT integration_id FROM regimex_tokens WHERE token_id=%s", (token_id,)
        ).fetchone()
        if row is None:
            raise NotFound()
        integration_id = cast(UUID | None, row[0])
        if integration_id is not None:
            _get(connection, integration_id, lock=True)
        changed = connection.execute(
            "UPDATE regimex_tokens SET revoked_at=clock_timestamp() "
            "WHERE token_id=%s AND revoked_at IS NULL RETURNING token_id",
            (token_id,),
        ).fetchone()
        if changed is not None and integration_id is not None:
            _event(connection, integration_id, "key.revoked", None, token_id=token_id)


class IntegrationRepository:
    def __init__(self, database: DatabaseRepository) -> None:
        self.database = database

    def create(self, actor: Principal, request: IntegrationCreate) -> IntegrationView:
        with self.database.connection() as connection:
            _admin(connection, actor, lock=True)
            reserve_integration_owner(connection, request.owner_id)
            row = connection.execute(
                "INSERT INTO regimex_integrations "
                "(integration_id,owner_id,source_namespace,display_name) VALUES (%s,%s,%s,%s) "
                "ON CONFLICT (owner_id,source_namespace) DO NOTHING RETURNING "
                + _INTEGRATION_FIELDS,
                (uuid4(), request.owner_id, request.source_namespace, request.display_name),
            ).fetchone()
            if row is None:
                raise Conflict("An integration already exists for this account and source.")
            result = _integration(row)
            _event(connection, result.integration_id, "integration.created", actor)
            return result

    def list(
        self, actor: Principal, *, limit: int = 50, offset: int = 0
    ) -> tuple[IntegrationView, ...]:
        self.database._page(limit, offset)
        with self.database.connection() as connection:
            _admin(connection, actor)
            rows = connection.execute(
                "SELECT " + _INTEGRATION_FIELDS + " FROM regimex_integrations "
                "ORDER BY created_at DESC,integration_id DESC LIMIT %s OFFSET %s",
                (limit, offset),
            ).fetchall()
            return tuple(_integration(row) for row in rows)

    def get(self, actor: Principal, integration_id: UUID) -> IntegrationView:
        with self.database.connection() as connection:
            _admin(connection, actor)
            return _get(connection, integration_id)

    def update(
        self, actor: Principal, integration_id: UUID, request: IntegrationUpdate
    ) -> IntegrationView:
        with self.database.connection() as connection:
            _admin(connection, actor, lock=True)
            current = _get(connection, integration_id, lock=True)
            if current.revision != request.expected_revision:
                raise Conflict("The integration changed. Reload it before editing.")
            row = connection.execute(
                "UPDATE regimex_integrations SET display_name=%s,enabled=%s,revision=revision+1,"
                "updated_at=clock_timestamp() WHERE integration_id=%s RETURNING "
                + _INTEGRATION_FIELDS,
                (
                    current.display_name if request.display_name is None else request.display_name,
                    current.enabled if request.enabled is None else request.enabled,
                    integration_id,
                ),
            ).fetchone()
            assert row is not None
            _event(connection, integration_id, "integration.updated", actor)
            return _integration(row)

    def issue_key(
        self, actor: Principal, integration_id: UUID, request: IntegrationKeyCreate
    ) -> IssuedIntegrationKey:
        with self.database.connection() as connection:
            _admin(connection, actor, lock=True)
            integration = _get(connection, integration_id, lock=True)
            return _issue_key(connection, integration, request, actor)

    def list_keys(
        self, actor: Principal, integration_id: UUID, *, limit: int = 50, offset: int = 0
    ) -> tuple[IntegrationKeyView, ...]:
        self.database._page(limit, offset)
        with self.database.connection() as connection:
            _admin(connection, actor)
            _get(connection, integration_id)
            rows = connection.execute(
                "SELECT " + _KEY_FIELDS + " FROM regimex_tokens WHERE integration_id=%s "
                "ORDER BY created_at DESC,token_id DESC LIMIT %s OFFSET %s",
                (integration_id, limit, offset),
            ).fetchall()
            return tuple(_key(row) for row in rows)

    def revoke_key(self, actor: Principal, integration_id: UUID, token_id: UUID) -> None:
        with self.database.connection() as connection:
            _admin(connection, actor, lock=True)
            _get(connection, integration_id, lock=True)
            row = connection.execute(
                "SELECT revoked_at FROM regimex_tokens WHERE integration_id=%s "
                "AND token_id=%s FOR UPDATE",
                (integration_id, token_id),
            ).fetchone()
            if row is None:
                raise NotFound("Integration key was not found.")
            if row[0] is None:
                connection.execute(
                    "UPDATE regimex_tokens SET revoked_at=clock_timestamp() WHERE token_id=%s",
                    (token_id,),
                )
                _event(connection, integration_id, "key.revoked", actor, token_id=token_id)

    def events(
        self, actor: Principal, integration_id: UUID, *, limit: int = 50, offset: int = 0
    ) -> tuple[AccessEventView, ...]:
        self.database._page(limit, offset)
        with self.database.connection() as connection:
            _admin(connection, actor)
            _get(connection, integration_id)
            rows = connection.execute(
                "SELECT " + _EVENT_FIELDS + " FROM regimex_access_events WHERE integration_id=%s "
                "ORDER BY created_at DESC,event_id DESC LIMIT %s OFFSET %s",
                (integration_id, limit, offset),
            ).fetchall()
            return tuple(
                AccessEventView.model_validate(
                    dict(zip(_EVENT_FIELDS.split(","), row, strict=True))
                )
                for row in rows
            )
