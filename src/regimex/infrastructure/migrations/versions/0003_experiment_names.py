"""Store editable history names separately from immutable snapshot documents."""

from alembic import op

revision = "0003_experiment_names"
down_revision = "0002_worker_queues"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE regimex_snapshots ADD COLUMN display_name text "
        "CHECK (display_name IS NULL OR char_length(display_name) BETWEEN 1 AND 120), "
        "ADD COLUMN display_name_search text "
        "CHECK ((display_name IS NULL) = (display_name_search IS NULL)), "
        "ADD COLUMN name_revision bigint NOT NULL DEFAULT 0 CHECK (name_revision >= 0)"
    )


def downgrade() -> None:
    raise RuntimeError("Database downgrades are not supported.")
