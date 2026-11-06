"""Persist bounded request windows shared by all keys of an integration."""

from alembic import op

revision = "0005_integration_limits"
down_revision = "0004_integration_access"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "CREATE TABLE regimex_integration_request_windows ("
        "integration_id uuid NOT NULL REFERENCES regimex_integrations(integration_id), "
        "owner_id text NOT NULL CHECK (char_length(owner_id)<=128), "
        "started_at timestamptz NOT NULL, "
        "request_count integer NOT NULL CHECK (request_count BETWEEN 1 AND 1000000), "
        "PRIMARY KEY(integration_id,owner_id))"
    )


def downgrade() -> None:
    raise RuntimeError("Database downgrades are not supported.")
