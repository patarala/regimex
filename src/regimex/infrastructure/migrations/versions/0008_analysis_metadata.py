"""Record analysis retry lineage and backfill Unicode history search fields."""

from alembic import op
from sqlalchemy import text

from regimex.infrastructure.history_repository import search_text

revision = "0008_analysis_metadata"
down_revision = "0007_resource_deletion"
branch_labels = None
depends_on = None


def upgrade() -> None:
    connection = op.get_bind()
    op.execute("ALTER TABLE regimex_snapshots ADD COLUMN identifiers_search text")
    op.execute("ALTER TABLE regimex_analyses ADD COLUMN model_version_search text")
    op.execute("ALTER TABLE regimex_analyses ADD COLUMN retry_of uuid")
    snapshots = text(
        "SELECT snapshot_id,source_id,run_id,source_snapshot_id,document->'instrument_ids' "
        "FROM regimex_snapshots"
    ).execution_options(yield_per=500)
    with connection.execute(snapshots) as rows:
        for batch in rows.partitions(500):
            connection.execute(
                text(
                    "UPDATE regimex_snapshots SET identifiers_search=:value WHERE snapshot_id=:id"
                ),
                [
                    {"id": row[0], "value": search_text(row[1], row[2], row[3], *(row[4] or []))}
                    for row in batch
                ],
            )
    analyses = text("SELECT analysis_id,model_version_id FROM regimex_analyses").execution_options(
        yield_per=500
    )
    with connection.execute(analyses) as rows:
        for batch in rows.partitions(500):
            connection.execute(
                text(
                    "UPDATE regimex_analyses SET model_version_search=:value WHERE analysis_id=:id"
                ),
                [{"id": row[0], "value": search_text(row[1])} for row in batch],
            )
    op.execute("ALTER TABLE regimex_snapshots ALTER COLUMN identifiers_search SET NOT NULL")
    op.execute("ALTER TABLE regimex_analyses ALTER COLUMN model_version_search SET NOT NULL")


def downgrade() -> None:
    raise RuntimeError("Database downgrades are not supported.")
