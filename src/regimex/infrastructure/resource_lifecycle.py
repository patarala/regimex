"""Serialize an owner's payload writes with reviewed resource removal."""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING
from uuid import UUID

from regimex.service_contracts import Conflict

if TYPE_CHECKING:
    import psycopg

    from regimex.infrastructure.database import DatabaseRepository


@contextmanager
def owner_payload_lock(
    database: DatabaseRepository, owner_id: str
) -> Iterator[psycopg.Connection[tuple[object, ...]]]:
    # Session ownership spans short database transactions and external object
    # I/O. Process death closes the connection and releases the lock.
    with database.connection() as connection:
        from psycopg.errors import LockNotAvailable, QueryCanceled

        connection.autocommit = True
        try:
            connection.execute(
                "SELECT pg_advisory_lock(hashtextextended(%s,0))",
                (f"regimex:{database.schema}:owner-payload:{owner_id}",),
            )
        except (LockNotAvailable, QueryCanceled) as exc:
            raise Conflict("Another storage change is in progress; retry shortly.") from exc
        yield connection


def require_snapshot_identity(
    connection: psycopg.Connection[tuple[object, ...]],
    owner_id: str,
    source_id: str,
    run_id: str,
    source_snapshot_id: str,
) -> None:
    if connection.execute(
        "SELECT 1 FROM regimex_deleted_snapshots WHERE owner_id=%s AND source_id=%s "
        "AND run_id=%s AND source_snapshot_id=%s",
        (owner_id, source_id, run_id, source_snapshot_id),
    ).fetchone():
        raise Conflict("This experiment was deleted; its original identity cannot be reused.")


def require_object_key(connection: psycopg.Connection[tuple[object, ...]], key: str) -> None:
    if connection.execute(
        "SELECT 1 FROM regimex_deleted_object_keys WHERE object_key=%s", (key,)
    ).fetchone():
        raise Conflict("This object's resource was deleted; its storage key cannot be reused.")
    parts = key.split("/")
    if len(parts) == 3 and parts[0] == "reports":
        try:
            analysis_id = UUID(parts[1])
        except ValueError:
            return
        if connection.execute(
            "SELECT 1 FROM regimex_deleted_analyses WHERE analysis_id=%s", (analysis_id,)
        ).fetchone():
            raise Conflict("The analysis was deleted; another report cannot be uploaded.")
