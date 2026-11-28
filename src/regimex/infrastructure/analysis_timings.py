"""Read retained Procrastinate events for specific load-test submissions."""

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from psycopg.rows import class_row

from regimex.infrastructure.database import DatabaseRepository


@dataclass(frozen=True)
class QueueActivity:
    analysis_id: UUID
    operation_id: UUID
    state: str
    attempts: int
    deliveries: int
    events: list[str]
    deferred: datetime | None
    started: datetime | None
    succeeded: datetime | None


def queue_activity(repository: DatabaseRepository, ids: list[UUID]) -> dict[UUID, QueueActivity]:
    with (
        repository.connection() as connection,
        connection.cursor(row_factory=class_row(QueueActivity)) as cursor,
    ):
        cursor.execute(
            """
            SELECT a.analysis_id, o.operation_id, o.state,
                   (o.document->>'attempt')::int AS attempts,
                   count(DISTINCT d.job_id) AS deliveries,
                   array_remove(array_agg(e.type::text ORDER BY d.job_id, e.id), NULL) AS events,
                   min(e.at) FILTER (WHERE e.type = 'deferred') AS deferred,
                   min(e.at) FILTER (WHERE e.type = 'started') AS started,
                   min(e.at) FILTER (WHERE e.type = 'succeeded') AS succeeded
            FROM regimex_analyses a
            JOIN regimex_operations o ON o.operation_id = a.operation_id
            LEFT JOIN regimex_deliveries d ON d.operation_id = o.operation_id
            LEFT JOIN procrastinate_events e ON e.job_id = d.job_id
            WHERE a.analysis_id = ANY(%s)
            GROUP BY a.analysis_id, o.operation_id
            """,
            (ids,),
        )
        return {row.analysis_id: row for row in cursor}
