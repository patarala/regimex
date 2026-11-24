"""Cross-host S3 admission and fencing backed by the service's PostgreSQL database."""

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import cast
from uuid import UUID, uuid4

import psycopg

from regimex.infrastructure.database import DatabaseRepository
from regimex.infrastructure.maintenance import Connection
from regimex.infrastructure.object_store import (
    ObjectCapacityError,
    ObjectConflictError,
    ObjectIntegrityError,
)


@dataclass(frozen=True)
class PendingWrite:
    key: str
    token: UUID
    digest: str
    size: int


def _lock(
    connection: Connection, name: str, *, shared: bool, checkpoint: Callable[[], None] | None
) -> None:
    query = (
        "SELECT pg_try_advisory_lock_shared(hashtextextended(%s,0))"
        if shared
        else "SELECT pg_try_advisory_lock(hashtextextended(%s,0))"
    )
    deadline = time.monotonic() + 5.0
    while True:
        if checkpoint is not None:
            checkpoint()
        if connection.execute(query, (name,)).fetchone() == (True,):
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("S3 coordination lock wait exceeded 5 seconds")
        time.sleep(min(remaining, 0.05))


class S3Coordinator:
    def __init__(self, database: DatabaseRepository) -> None:
        self.database = database

    @contextmanager
    def session(
        self,
        scope: str,
        limits: tuple[int, int, int],
        *,
        shared: bool,
        key: str | None = None,
        checkpoint: Callable[[], None] | None = None,
        verify_namespace: Callable[[], object] | None = None,
    ) -> Iterator[Connection]:
        try:
            with self.database.connection() as db:
                # Session locks survive the commit that makes a reservation durable.
                db.autocommit = True
                db.execute("SET synchronous_commit=on")
                _lock(db, scope, shared=shared, checkpoint=checkpoint)
                if key is not None:
                    _lock(db, scope + ":" + key, shared=False, checkpoint=checkpoint)
                self._bind(db, scope, limits, verify_namespace=verify_namespace)
                yield db
                # A completed read or write may be returned only by a live session.
                db.execute("SELECT 1")
        except psycopg.errors.LockNotAvailable:
            raise TimeoutError("S3 capacity reservation lock wait exceeded its deadline") from None
        except psycopg.Error:
            raise OSError("S3 storage coordination database is unavailable") from None

    @staticmethod
    def _bind(
        connection: Connection,
        scope: str,
        limits: tuple[int, int, int],
        *,
        verify_namespace: Callable[[], object] | None,
    ) -> None:
        query = (
            "SELECT n.scope_id,n.object_limit,n.byte_limit,n.entry_limit "
            "FROM regimex_s3_namespaces n JOIN regimex_maintenance_identity i "
            "USING (installation_id) WHERE i.singleton"
        )
        row = connection.execute(query).fetchone()
        if row is None:
            if verify_namespace is None:
                return
            # A typo in an unreachable endpoint must not permanently bind the database.
            verify_namespace()
            connection.execute(
                "INSERT INTO regimex_s3_namespaces(installation_id,scope_id,object_limit,"
                "byte_limit,entry_limit) SELECT installation_id,%s,%s,%s,%s "
                "FROM regimex_maintenance_identity WHERE singleton ON CONFLICT DO NOTHING",
                (scope, *limits),
            )
            row = connection.execute(query).fetchone()
        if row != (scope, *limits):
            raise ObjectIntegrityError(
                "S3 nodes have different provider namespaces or capacity settings"
            )

    @staticmethod
    def pending(db: Connection, scope: str, key: str) -> PendingWrite | None:
        row = db.execute(
            "SELECT write_token,sha256,size_bytes FROM regimex_s3_pending_writes "
            "WHERE scope_id=%s AND object_key=%s",
            (scope, key),
        ).fetchone()
        if row is None:
            return None
        token, checksum, size = cast(tuple[UUID, str, int], row)
        return PendingWrite(key, token, checksum, size)

    @staticmethod
    def pending_usage(db: Connection, scope: str) -> tuple[int, int]:
        row = db.execute(
            "SELECT coalesce(sum(size_bytes),0)::bigint,count(*) "
            "FROM regimex_s3_pending_writes WHERE scope_id=%s",
            (scope,),
        ).fetchone()
        assert row is not None
        return cast(tuple[int, int], row)

    @staticmethod
    def reserve(
        db: Connection,
        scope: str,
        key: str,
        checksum: str,
        size: int,
        usage: Callable[[], tuple[int, int]],
        *,
        byte_limit: int,
        entry_limit: int,
        checkpoint: Callable[[], None] | None,
    ) -> PendingWrite:
        # Scanning provider pages can outlast the database's idle transaction
        # timeout. A session lock serializes admission without an open transaction.
        admission = "s3-capacity:" + scope
        _lock(db, admission, shared=False, checkpoint=checkpoint)
        try:
            # Sample reservations first. A PUT committed during the following inventory
            # is charged at least once, even if another session finishes it meanwhile.
            pending_bytes, pending_count = S3Coordinator.pending_usage(db, scope)
            used_bytes, used_count = usage()
            if used_count + pending_count >= entry_limit:
                raise ObjectCapacityError("S3 inventory and pending writes reached the entry limit")
            if used_bytes + pending_bytes + size > byte_limit:
                raise ObjectCapacityError("S3 storage and pending writes exceed capacity")
            write = PendingWrite(key, uuid4(), checksum, size)
            reservation = db.execute(
                "INSERT INTO regimex_s3_pending_writes(installation_id,scope_id,object_key,"
                "write_token,sha256,size_bytes) SELECT installation_id,%s,%s,%s,%s,%s "
                "FROM regimex_maintenance_identity WHERE singleton",
                (scope, key, write.token, checksum, size),
            )
            if reservation.rowcount != 1:
                raise ObjectIntegrityError("S3 reservation requires an active installation")
            return write
        finally:
            db.execute("SELECT pg_advisory_unlock(hashtextextended(%s,0))", (admission,))

    @staticmethod
    def finish(db: Connection, scope: str, write: PendingWrite) -> None:
        if (
            db.execute(
                "DELETE FROM regimex_s3_pending_writes WHERE scope_id=%s AND object_key=%s "
                "AND write_token=%s",
                (scope, write.key, write.token),
            ).rowcount
            != 1
        ):
            raise ObjectConflictError("S3 write reservation has changed")
