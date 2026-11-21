"""Track when the owner moved a completed training run out of the working list."""

from alembic import op

revision = "0013_training_archive"
down_revision = "0012_training_metrics"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE regimex_training_runs ADD COLUMN archived_at timestamptz")
    op.execute(
        "CREATE INDEX regimex_training_archived ON regimex_training_runs (operation_id) "
        "WHERE archived_at IS NOT NULL"
    )


def downgrade() -> None:
    raise RuntimeError("Database downgrades are not supported.")
