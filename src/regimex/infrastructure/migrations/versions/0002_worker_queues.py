"""Associate registered queue workers with their actual queue and slot capacity."""

from alembic import op

revision = "0002_worker_queues"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """CREATE TABLE regimex_worker_queues (
            worker_id bigint PRIMARY KEY REFERENCES procrastinate_workers(id) ON DELETE CASCADE,
            queue_name text NOT NULL CHECK (queue_name IN ('analysis','admin','maintenance')),
            slots integer NOT NULL CHECK (slots >= 1))"""
    )
    op.execute("CREATE INDEX regimex_worker_queues_queue ON regimex_worker_queues(queue_name)")


def downgrade() -> None:
    raise RuntimeError("Database downgrades are not supported.")
