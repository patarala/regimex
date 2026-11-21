"""Keep training results available after an unused candidate's weights are discarded."""

from alembic import op

revision = "0012_training_metrics"
down_revision = "0011_training_progress"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """CREATE TABLE regimex_training_metrics (
            operation_id uuid PRIMARY KEY REFERENCES regimex_training_runs(operation_id),
            document json NOT NULL CHECK (json_typeof(document)='object'))"""
    )
    op.execute(
        "ALTER TABLE regimex_deletion_plans ALTER COLUMN source_id DROP NOT NULL, "
        "DROP CONSTRAINT regimex_deletion_plans_target_kind_check, "
        "ADD CONSTRAINT regimex_deletion_plans_target_kind_check "
        "CHECK (target_kind IN ('analysis','experiment','model_weights')), "
        "ADD CONSTRAINT regimex_deletion_plans_source_check "
        "CHECK ((target_kind='model_weights') = (source_id IS NULL))"
    )
    op.execute(
        "ALTER TABLE regimex_models ADD COLUMN weights_deletion_id uuid UNIQUE "
        "REFERENCES regimex_deletion_plans(plan_id)"
    )


def downgrade() -> None:
    raise RuntimeError("Database downgrades are not supported.")
