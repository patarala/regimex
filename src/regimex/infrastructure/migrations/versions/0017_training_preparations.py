"""Track each training data preparation and its current administrative job."""

from alembic import op

revision = "0017_training_preparations"
down_revision = "0016_s3_coordination"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """CREATE TABLE regimex_training_preparations (
            owner_id text NOT NULL,
            preparation_id uuid NOT NULL,
            created_at timestamptz NOT NULL DEFAULT now(),
            plan jsonb NOT NULL,
            stage text NOT NULL CHECK (
                stage IN ('development','final','train','validation','test')),
            operation_id uuid NOT NULL UNIQUE REFERENCES regimex_admin_runs(operation_id),
            selection_id text,
            outputs jsonb NOT NULL DEFAULT '{}',
            PRIMARY KEY(owner_id, preparation_id))"""
    )
    op.execute(
        "CREATE INDEX regimex_preparations_history ON regimex_training_preparations "
        "(owner_id,created_at DESC,preparation_id DESC)"
    )
    op.execute(
        "CREATE INDEX regimex_preparations_unfinished ON regimex_training_preparations "
        "(created_at,preparation_id) WHERE NOT (outputs ? stage)"
    )


def downgrade() -> None:
    raise RuntimeError("Database downgrades are not supported.")
