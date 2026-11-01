"""Shared offline maintenance assertions, coordination and PostgreSQL snapshots.

SQL locks stop database commits, not object uploads or independent filesystem
access. Callers still require stopped services and exclusive store access. A
retention caller must commit durable version fences before external deletion.
"""

import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal, cast

import psycopg
from psycopg import sql
from pydantic import AwareDatetime, Field

from regimex.contracts import StrictModel
from regimex.service_contracts import Conflict

Connection = psycopg.Connection[tuple[object, ...]]
_SCHEMA = re.compile(r"[a-z_][a-z0-9_]{0,62}\Z")


class ServicesStoppedAttestation(StrictModel):
    stopped_at: AwareDatetime
    statement: str = Field(min_length=1, max_length=4096)
    api_stopped: Literal[True]
    analysis_workers_stopped: Literal[True]
    administrative_workers_stopped: Literal[True]
    maintenance_stopped: Literal[True]
    garbage_collection_stopped: Literal[True]
    exclusive_object_access: Literal[True]


@dataclass(frozen=True, slots=True)
class QuiescenceLimits:
    lock_timeout_seconds: int = 5
    statement_timeout_seconds: int = 30
    max_tables: int = 256

    def __post_init__(self) -> None:
        if any(
            type(value) is not int or value < 1
            for value in (
                self.lock_timeout_seconds,
                self.statement_timeout_seconds,
                self.max_tables,
            )
        ):
            raise ValueError("Quiescence limits must be positive integers.")


DEFAULT_QUIESCENCE_LIMITS = QuiescenceLimits()


def _schema(schema: str) -> str:
    if _SCHEMA.fullmatch(schema) is None:
        raise ValueError("Maintenance requires a bounded lowercase PostgreSQL schema identifier.")
    return schema


def maintenance_lock(connection: Connection, schema: str) -> None:
    """Hold a session lock until connection close or explicit maintenance_unlock."""
    result = connection.execute(
        "SELECT pg_try_advisory_lock(hashtextextended(%s,0))",
        (_schema(schema) + ":regimex-offline-maintenance",),
    ).fetchone()
    if result != (True,):
        raise Conflict("Another recovery or garbage-collection process owns maintenance.")


def maintenance_unlock(connection: Connection, schema: str) -> None:
    connection.execute(
        "SELECT pg_advisory_unlock(hashtextextended(%s,0))",
        (_schema(schema) + ":regimex-offline-maintenance",),
    )


@contextmanager
def quiescent_transaction(
    connection: Connection,
    *,
    maximum_idle_seconds: int = 600,
    schema: str | None = None,
    limits: QuiescenceLimits = DEFAULT_QUIESCENCE_LIMITS,
) -> Iterator[None]:
    """Lock before taking the repeatable-read snapshot; bound database waits.

    An explicit zero idle timeout supports long offline backups whose caller
    enforces an overall deadline while copying objects and running pg_dump.
    The connection must already hold the shared maintenance session lock.
    """
    if not connection.autocommit:
        raise ValueError("Maintenance snapshots require an autocommit connection.")
    if type(maximum_idle_seconds) is not int or maximum_idle_seconds < 0:
        raise ValueError("The maintenance idle timeout must be a nonnegative integer.")
    if schema is None:
        row = connection.execute("SELECT current_schema()").fetchone()
        if row is None:
            raise Conflict("The maintenance schema is unavailable.")
        schema = cast(str, row[0])
    schema = _schema(schema)
    # Table discovery happens outside the snapshot. PostgreSQL requires LOCK
    # before the first SELECT at repeatable read to see commits it waited for.
    tables = connection.execute(
        "SELECT tablename FROM pg_tables WHERE schemaname=%s ORDER BY tablename LIMIT %s",
        (schema, limits.max_tables + 1),
    ).fetchall()
    if not tables or len(tables) > limits.max_tables:
        raise Conflict("The initialized maintenance schema exceeds its table bound.")
    with connection.transaction():
        connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
        connection.execute(sql.SQL("SET LOCAL search_path TO {}").format(sql.Identifier(schema)))
        connection.execute(
            sql.SQL("SET LOCAL lock_timeout TO {}").format(
                sql.Literal(str(limits.lock_timeout_seconds * 1000))
            )
        )
        connection.execute(
            sql.SQL("SET LOCAL statement_timeout TO {}").format(
                sql.Literal(str(limits.statement_timeout_seconds * 1000))
            )
        )
        connection.execute(
            sql.SQL("SET LOCAL idle_in_transaction_session_timeout TO {}").format(
                sql.Literal(str(maximum_idle_seconds * 1000))
            )
        )
        connection.execute(
            sql.SQL("LOCK TABLE {} IN SHARE MODE").format(
                sql.SQL(",").join(sql.Identifier(schema, cast(str, row[0])) for row in tables)
            )
        )
        if connection.execute(
            "SELECT 1 FROM regimex_operations "
            "WHERE state IN ('queued','running','cancel_requested') LIMIT 1"
        ).fetchone():
            raise Conflict("Maintenance requires zero queued, running or cancelling operations.")
        if connection.execute(
            "SELECT 1 FROM procrastinate_jobs WHERE status='doing' LIMIT 1"
        ).fetchone():
            raise Conflict("Maintenance requires every active queue delivery to stop.")
        if connection.execute(
            "SELECT 1 FROM pg_stat_activity WHERE datname=current_database() "
            "AND pid<>pg_backend_pid() AND application_name='regimex' LIMIT 1"
        ).fetchone():
            raise Conflict("Regimex database sessions remain; stop every service process.")
        yield
