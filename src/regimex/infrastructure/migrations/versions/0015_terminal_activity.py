"""Track explicitly recorded terminal work independently of worker jobs."""

from alembic import op

revision = "0015_terminal_activity"
down_revision = "0014_external_training"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """CREATE TABLE regimex_terminal_activities (
            owner_id text NOT NULL, activity_id uuid NOT NULL,
            title text NOT NULL CHECK (length(title) BETWEEN 1 AND 120),
            note text CHECK (length(note) <= 500),
            status text NOT NULL CHECK (status IN ('running','succeeded','failed','interrupted')),
            started_at timestamptz NOT NULL, finished_at timestamptz,
            exit_code integer, result_text text CHECK (length(result_text) <= 16000),
            PRIMARY KEY (owner_id,activity_id),
            CHECK ((status='running') = (finished_at IS NULL)),
            CHECK (status!='running' OR (exit_code IS NULL AND result_text IS NULL)),
            CHECK ((status!='succeeded' OR exit_code=0) AND (status!='failed' OR exit_code!=0)),
            CHECK (finished_at >= started_at))"""
    )
    op.execute(
        "CREATE INDEX regimex_terminal_activity_owner_time ON regimex_terminal_activities "
        "(owner_id,started_at DESC,activity_id DESC)"
    )


def downgrade() -> None:
    raise RuntimeError("Database downgrades are not supported.")
