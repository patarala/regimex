"""Keep training run names separate from immutable training requests and results."""

from alembic import op

revision = "0009_training_run_names"
down_revision = "0008_analysis_metadata"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE regimex_training_runs ADD COLUMN display_name text "
        "CHECK (display_name IS NULL OR char_length(display_name) BETWEEN 1 AND 120), "
        "ADD COLUMN name_revision bigint NOT NULL DEFAULT 0 CHECK (name_revision >= 0)"
    )


def downgrade() -> None:
    raise RuntimeError("Database downgrades are not supported.")
