"""Account for immutable owner payloads before uploads and across retries."""

from alembic import op

revision = "0006_storage_quotas"
down_revision = "0005_integration_limits"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "CREATE TABLE regimex_storage_reservations ("
        "object_key text PRIMARY KEY CHECK(octet_length(object_key) BETWEEN 1 AND 1024), "
        "owner_id text NOT NULL CHECK(char_length(owner_id) BETWEEN 1 AND 128), "
        "source_id text NOT NULL CHECK(char_length(source_id) BETWEEN 1 AND 128), "
        "integration_id uuid REFERENCES regimex_integrations(integration_id), "
        "sha256 text NOT NULL CHECK(sha256 ~ '^[0-9a-f]{64}$'), "
        "size_bytes bigint NOT NULL CHECK(size_bytes>=0), "
        "created_at timestamptz NOT NULL DEFAULT clock_timestamp())"
    )
    op.execute("CREATE INDEX regimex_storage_owner ON regimex_storage_reservations(owner_id)")
    op.execute(
        "CREATE INDEX regimex_storage_integration "
        "ON regimex_storage_reservations(integration_id) WHERE integration_id IS NOT NULL"
    )
    op.execute(
        "WITH owned AS ("
        "SELECT object_key,owner_id,source_id FROM regimex_snapshots UNION "
        "SELECT r.object_key,r.owner_id,s.source_id FROM regimex_analysis_reports r "
        "JOIN regimex_snapshots s ON s.snapshot_id=r.snapshot_id) "
        "INSERT INTO regimex_storage_reservations "
        "(object_key,owner_id,source_id,integration_id,sha256,size_bytes) "
        "SELECT owned.object_key,owned.owner_id,owned.source_id,"
        "i.integration_id,o.sha256,o.size_bytes "
        "FROM owned JOIN regimex_objects o USING(object_key) "
        "LEFT JOIN regimex_integrations i ON i.source_namespace=owned.source_id "
        "AND (i.owner_id=owned.owner_id OR EXISTS ("
        "SELECT 1 FROM regimex_integration_subjects s WHERE s.effective_owner_id=owned.owner_id "
        "AND s.integration_owner_id=i.owner_id AND s.source_namespace=i.source_namespace))"
    )


def downgrade() -> None:
    raise RuntimeError("Database downgrades are not supported.")
