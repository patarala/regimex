"""Bind additional validation request keys to the evaluation they reused."""

from alembic import op

revision = "0010_evaluation_reuse"
down_revision = "0009_training_run_names"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """CREATE TABLE regimex_evaluation_keys (
            owner_id text NOT NULL,
            idempotency_key text NOT NULL CHECK (length(idempotency_key) BETWEEN 1 AND 128),
            request_sha256 text NOT NULL CHECK (request_sha256 ~ '^[0-9a-f]{64}$'),
            operation_id uuid NOT NULL REFERENCES regimex_evaluation_requests(operation_id),
            PRIMARY KEY (owner_id,idempotency_key))"""
    )
    op.execute(
        """CREATE TRIGGER regimex_evaluation_key_immutable
            BEFORE UPDATE OR DELETE ON regimex_evaluation_keys
            FOR EACH ROW EXECUTE FUNCTION regimex_reject_intake_rewrite()"""
    )
    op.execute(
        "CREATE INDEX regimex_evaluation_inputs ON regimex_evaluation_requests "
        "(model_version_id,dataset_id,reference_set_id)"
    )


def downgrade() -> None:
    raise RuntimeError("Database downgrades are not supported.")
