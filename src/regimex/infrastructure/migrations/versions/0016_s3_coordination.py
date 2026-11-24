"""Reserve provider capacity before a conditional S3 write can leave the process."""

from alembic import op

revision = "0016_s3_coordination"
down_revision = "0015_terminal_activity"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """CREATE TABLE regimex_s3_namespaces (
            installation_id uuid PRIMARY KEY,
            scope_id text NOT NULL,
            object_limit bigint NOT NULL CHECK (object_limit >= 0),
            byte_limit bigint NOT NULL CHECK (byte_limit >= object_limit),
            entry_limit integer NOT NULL CHECK (entry_limit > 0),
            UNIQUE (installation_id, scope_id))"""
    )
    op.execute(
        """CREATE TABLE regimex_s3_pending_writes (
            installation_id uuid NOT NULL,
            scope_id text NOT NULL,
            object_key text NOT NULL,
            write_token uuid NOT NULL UNIQUE,
            sha256 text NOT NULL CHECK (sha256 ~ '^[0-9a-f]{64}$'),
            size_bytes bigint NOT NULL CHECK (size_bytes >= 0),
            PRIMARY KEY(scope_id, object_key),
            FOREIGN KEY(installation_id, scope_id) REFERENCES
                regimex_s3_namespaces(installation_id, scope_id))"""
    )


def downgrade() -> None:
    raise RuntimeError("Database downgrades are not supported.")
