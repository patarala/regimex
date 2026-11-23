"""Persist owner-scoped terminal activity receipts with idempotent completion."""

from datetime import UTC, datetime
from uuid import UUID

from psycopg import Cursor
from psycopg.rows import class_row

from regimex.infrastructure.database import DatabaseRepository
from regimex.service_contracts import Conflict, Forbidden, NotFound, Principal
from regimex.terminal_activity import (
    ActivityFinish,
    ActivityStart,
    TerminalActivity,
    TerminalActivitySummary,
)

_COLUMNS = "activity_id,title,note,status,started_at,finished_at,exit_code"
_OWNER = " FROM regimex_terminal_activities WHERE owner_id=%s"


class TerminalActivityRepository:
    def __init__(self, database: DatabaseRepository) -> None:
        self.database = database

    @staticmethod
    def _owner(principal: Principal) -> str:
        if principal.role == "integration":
            raise Forbidden()
        return principal.owner_id

    @staticmethod
    def _record(
        cursor: Cursor[TerminalActivity], owner: str, activity_id: UUID, *, lock: bool = False
    ) -> TerminalActivity:
        row = cursor.execute(
            "SELECT "
            + _COLUMNS
            + ",result_text"
            + _OWNER
            + " AND activity_id=%s"
            + (" FOR UPDATE" if lock else ""),
            (owner, activity_id),
        ).fetchone()
        if row is None:
            raise NotFound()
        return row

    def list(
        self, principal: Principal, *, limit: int, offset: int
    ) -> tuple[TerminalActivitySummary, ...]:
        owner = self._owner(principal)
        self.database._page(limit, offset)
        with (
            self.database.connection() as conn,
            conn.cursor(row_factory=class_row(TerminalActivitySummary)) as cursor,
        ):
            return tuple(
                cursor.execute(
                    "SELECT "
                    + _COLUMNS
                    + _OWNER
                    + " ORDER BY started_at DESC,activity_id DESC LIMIT %s OFFSET %s",
                    (owner, limit, offset),
                ).fetchall()
            )

    def get(self, principal: Principal, activity_id: UUID) -> TerminalActivity:
        owner = self._owner(principal)
        with (
            self.database.connection() as conn,
            conn.cursor(row_factory=class_row(TerminalActivity)) as cursor,
        ):
            return self._record(cursor, owner, activity_id)

    def start(self, principal: Principal, request: ActivityStart) -> TerminalActivity:
        owner = self._owner(principal)
        with (
            self.database.connection() as conn,
            conn.cursor(row_factory=class_row(TerminalActivity)) as cursor,
        ):
            cursor.execute(
                "INSERT INTO regimex_terminal_activities "
                "(owner_id,activity_id,title,note,status,started_at) "
                "VALUES (%s,%s,%s,%s,'running',COALESCE(%s,statement_timestamp())) "
                "ON CONFLICT (owner_id,activity_id) DO NOTHING",
                (owner, request.activity_id, request.title, request.note, request.started_at),
            )
            record = self._record(cursor, owner, request.activity_id)
            if (record.title, record.note) != (request.title, request.note) or (
                request.started_at is not None and record.started_at != request.started_at
            ):
                raise Conflict("The activity ID already records a different start.")
            return record

    def finish(
        self, principal: Principal, activity_id: UUID, request: ActivityFinish
    ) -> TerminalActivity:
        owner = self._owner(principal)
        with (
            self.database.connection() as conn,
            conn.cursor(row_factory=class_row(TerminalActivity)) as cursor,
        ):
            record = self._record(cursor, owner, activity_id, lock=True)
            if record.status != "running":
                if (record.status, record.exit_code, record.result_text) != (
                    request.status,
                    request.exit_code,
                    request.result_text,
                ) or (
                    request.finished_at is not None and request.finished_at != record.finished_at
                ):
                    raise Conflict("The activity already records a different outcome.")
                return record
            finished_at = request.finished_at or datetime.now(UTC)
            if finished_at < record.started_at:
                raise Conflict("The activity cannot finish before its recorded start.")
            cursor.execute(
                "UPDATE regimex_terminal_activities SET status=%s,exit_code=%s,result_text=%s,"
                "finished_at=%s WHERE owner_id=%s AND activity_id=%s",
                (
                    request.status,
                    request.exit_code,
                    request.result_text,
                    finished_at,
                    owner,
                    activity_id,
                ),
            )
            return record.model_copy(
                update={
                    "status": request.status,
                    "exit_code": request.exit_code,
                    "finished_at": finished_at,
                    "result_text": request.result_text,
                }
            )
