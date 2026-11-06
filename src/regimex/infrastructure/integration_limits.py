"""Database-clock request budgets and atomic integration intake limits."""

import math
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, cast

from regimex.infrastructure.integration_repository import require_integration
from regimex.service_contracts import CapacityExceeded, Principal, ServiceError, Unauthorized

if TYPE_CHECKING:
    import psycopg

    from regimex.infrastructure.database import DatabaseRepository


class IntegrationRateLimited(ServiceError):
    def __init__(self, retry_after_seconds: int, *, subject: bool) -> None:
        label = "user" if subject else "connection"
        super().__init__(
            "rate_limit", f"This integration {label} has reached its request limit.", 429
        )
        self.retry_after_seconds = max(1, retry_after_seconds)


def lock_integration(
    database: DatabaseRepository,
    connection: psycopg.Connection[tuple[object, ...]],
    principal: Principal,
    *,
    token_update: bool = False,
) -> None:
    """Serialize this connection's budgets and fence administrative revocation."""
    if principal.integration_id is None:
        raise Unauthorized()
    database._lock(connection, f"integration-capacity:{principal.integration_id}")
    connection.execute(
        "SELECT integration_id FROM regimex_integrations WHERE integration_id=%s FOR SHARE",
        (principal.integration_id,),
    ).fetchone()
    connection.execute(
        "SELECT token_id FROM regimex_tokens WHERE token_id=%s "
        + ("FOR UPDATE" if token_update else "FOR SHARE"),
        (principal.token_id,),
    ).fetchone()


class IntegrationLimitsRepository:
    """One reusable minute window per integration and authenticated owner.

    Rejected requests spend neither budget. Windows start on the first accepted
    request and expire after sixty seconds; keys never identify budget rows.
    """

    def __init__(
        self,
        database: DatabaseRepository,
        *,
        requests_per_minute: int = 600,
        subject_requests_per_minute: int = 120,
    ) -> None:
        for value in (requests_per_minute, subject_requests_per_minute):
            if type(value) is not int or not 1 <= value <= 1_000_000:
                raise ValueError("Integration request limits must be between one and one million.")
        if subject_requests_per_minute > requests_per_minute:
            raise ValueError("The subject request limit cannot exceed the integration limit.")
        self.database = database
        self.requests_per_minute = requests_per_minute
        self.subject_requests_per_minute = subject_requests_per_minute

    def consume(self, principal: Principal) -> None:
        if principal.role != "integration":
            return
        with self.database.connection() as connection:
            self.consume_in_transaction(connection, principal)

    def consume_in_transaction(
        self, connection: psycopg.Connection[tuple[object, ...]], principal: Principal
    ) -> None:
        """Admit a delegated request atomically with creation of its owner binding."""
        lock_integration(self.database, connection, principal)
        require_integration(connection, principal)
        row = connection.execute("SELECT clock_timestamp()").fetchone()
        assert row is not None
        now = cast(datetime, row[0])
        windows: list[tuple[str, datetime, int]] = []
        for owner, maximum in (
            ("", self.requests_per_minute),
            (principal.owner_id, self.subject_requests_per_minute),
        ):
            row = connection.execute(
                "SELECT started_at,request_count FROM regimex_integration_request_windows "
                "WHERE integration_id=%s AND owner_id=%s",
                (principal.integration_id, owner),
            ).fetchone()
            started, count = now, 0
            if row is not None and now < cast(datetime, row[0]) + timedelta(seconds=60):
                started, count = cast(datetime, row[0]), cast(int, row[1])
            if count >= maximum:
                remaining = math.ceil((started + timedelta(seconds=60) - now).total_seconds())
                raise IntegrationRateLimited(remaining, subject=bool(owner))
            windows.append((owner, started, count + 1))
        for owner, started, count in windows:
            connection.execute(
                "INSERT INTO regimex_integration_request_windows "
                "(integration_id,owner_id,started_at,request_count) VALUES (%s,%s,%s,%s) "
                "ON CONFLICT (integration_id,owner_id) DO UPDATE SET "
                "started_at=EXCLUDED.started_at,request_count=EXCLUDED.request_count",
                (principal.integration_id, owner, started, count),
            )


def check_pending(
    connection: psycopg.Connection[tuple[object, ...]], principal: Principal, maximum: int
) -> None:
    """Count all subjects under the integration lock held by analysis intake."""
    row = connection.execute(
        "WITH integration AS (SELECT owner_id,source_namespace FROM regimex_integrations "
        "WHERE integration_id=%s), owners AS (SELECT owner_id FROM integration UNION "
        "SELECT subject.effective_owner_id FROM regimex_integration_subjects subject "
        "JOIN integration i ON subject.integration_owner_id=i.owner_id "
        "AND subject.source_namespace=i.source_namespace) "
        "SELECT count(*) FROM (SELECT 1 FROM owners "
        "JOIN regimex_operations o ON o.owner_id=owners.owner_id "
        "JOIN regimex_analyses a ON a.operation_id=o.operation_id "
        "JOIN regimex_snapshots snapshot ON snapshot.snapshot_id=a.snapshot_id "
        "WHERE o.kind='analysis' AND o.state IN ('queued','running','cancel_requested') "
        "AND snapshot.source_id=(SELECT source_namespace FROM integration) LIMIT %s) pending",
        (principal.integration_id, maximum),
    ).fetchone()
    assert row is not None
    if cast(int, row[0]) >= maximum:
        raise CapacityExceeded(
            "This integration connection has reached its pending analysis limit."
        )
