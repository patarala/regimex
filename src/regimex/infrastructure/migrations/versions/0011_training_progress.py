"""Keep incremental chart samples while a training attempt is active."""

from alembic import op

revision = "0011_training_progress"
down_revision = "0010_evaluation_reuse"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """CREATE TABLE regimex_training_progress (
            operation_id uuid NOT NULL REFERENCES regimex_training_runs(operation_id),
            attempt integer NOT NULL CHECK (attempt > 0),
            position integer NOT NULL CHECK (position BETWEEN 1 AND 64000),
            measurement jsonb NOT NULL,
            PRIMARY KEY (operation_id, attempt, position))"""
    )


def downgrade() -> None:
    raise RuntimeError("Database downgrades are not supported.")
