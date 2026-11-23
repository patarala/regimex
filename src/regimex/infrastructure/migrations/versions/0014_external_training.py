"""Record verified external training without creating local dataset or reference entries."""

from alembic import op

revision = "0014_external_training"
down_revision = "0013_training_archive"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE regimex_training_runs "
        "ADD COLUMN external boolean NOT NULL DEFAULT false, "
        "ADD COLUMN external_evaluations json NOT NULL DEFAULT '[]'::json "
        "CHECK (json_typeof(external_evaluations)='array'), "
        "ADD COLUMN external_latency json CHECK (json_typeof(external_latency)='object'), "
        "ALTER COLUMN dataset_id DROP NOT NULL, "
        "ALTER COLUMN labels_reference_set_id DROP NOT NULL, "
        "ALTER COLUMN validation_reference_set_id DROP NOT NULL, "
        "ADD CONSTRAINT regimex_training_input_origin CHECK ("
        "(external AND committed_model_version_id IS NOT NULL "
        "AND model_version_id=committed_model_version_id "
        "AND num_nonnulls(dataset_id,labels_reference_set_id,validation_reference_set_id)=0) OR "
        "(NOT external AND num_nonnulls(dataset_id,labels_reference_set_id,"
        "validation_reference_set_id)=3))"
    )


def downgrade() -> None:
    raise RuntimeError("Database downgrades are not supported.")
