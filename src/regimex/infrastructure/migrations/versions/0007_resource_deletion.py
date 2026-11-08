"""Keep deletion receipts and prevent replay from restoring removed resources."""

from alembic import op

revision = "0007_resource_deletion"
down_revision = "0006_storage_quotas"
branch_labels = None
depends_on = None

STATEMENTS = (
    """ALTER TABLE regimex_tokens DROP CONSTRAINT regimex_token_scopes,
        ADD CONSTRAINT regimex_token_scopes CHECK (
            array_position(scopes,NULL) IS NULL
            AND scopes <@ ARRAY['read','upload','analyze','delete']::text[]
            AND ((role='integration' AND cardinality(scopes) BETWEEN 1 AND 4)
                 OR (role<>'integration' AND cardinality(scopes)=0)))""",
    """CREATE TABLE regimex_deletion_plans (
        plan_id uuid PRIMARY KEY, owner_id text NOT NULL, source_id text NOT NULL,
        run_id text NOT NULL, target_kind text NOT NULL
            CHECK (target_kind IN ('analysis','experiment')), target_id uuid NOT NULL,
        sha256 text NOT NULL CHECK (sha256 ~ '^[0-9a-f]{64}$'),
        document jsonb NOT NULL, created_at timestamptz NOT NULL,
        expires_at timestamptz NOT NULL, applied_at timestamptz,
        completed_at timestamptz, CHECK (completed_at IS NULL OR applied_at IS NOT NULL))""",
    """CREATE INDEX regimex_deletion_plans_owner
        ON regimex_deletion_plans(owner_id,created_at DESC,plan_id)""",
    """CREATE TABLE regimex_deleted_snapshots (
        snapshot_id uuid PRIMARY KEY, owner_id text NOT NULL, source_id text NOT NULL,
        run_id text NOT NULL, source_snapshot_id text NOT NULL,
        deletion_id uuid NOT NULL REFERENCES regimex_deletion_plans(plan_id),
        UNIQUE(owner_id,source_id,run_id,source_snapshot_id))""",
    """CREATE TABLE regimex_deleted_analyses (
        analysis_id uuid PRIMARY KEY,
        deletion_id uuid NOT NULL REFERENCES regimex_deletion_plans(plan_id))""",
    """CREATE TABLE regimex_deleted_object_keys (
        object_key text PRIMARY KEY,
        deletion_id uuid NOT NULL REFERENCES regimex_deletion_plans(plan_id))""",
    """CREATE TABLE regimex_deletion_objects (
        deletion_id uuid NOT NULL REFERENCES regimex_deletion_plans(plan_id),
        object_key text NOT NULL REFERENCES regimex_deleted_object_keys(object_key),
        document jsonb NOT NULL,
        state text NOT NULL DEFAULT 'pending' CHECK(state IN ('pending','deleted')),
        failure text, PRIMARY KEY(deletion_id,object_key))""",
    """CREATE FUNCTION regimex_preserve_resource_tombstone() RETURNS trigger
        LANGUAGE plpgsql AS $$ BEGIN
        RAISE EXCEPTION 'resource deletion identities are permanent';
        END $$""",
    """CREATE TRIGGER regimex_deleted_snapshot_immutable BEFORE UPDATE OR DELETE
        ON regimex_deleted_snapshots FOR EACH ROW
        EXECUTE FUNCTION regimex_preserve_resource_tombstone()""",
    """CREATE TRIGGER regimex_deleted_analysis_immutable BEFORE UPDATE OR DELETE
        ON regimex_deleted_analyses FOR EACH ROW
        EXECUTE FUNCTION regimex_preserve_resource_tombstone()""",
    """CREATE TRIGGER regimex_deleted_object_key_immutable BEFORE UPDATE OR DELETE
        ON regimex_deleted_object_keys FOR EACH ROW
        EXECUTE FUNCTION regimex_preserve_resource_tombstone()""",
    """CREATE FUNCTION regimex_guard_deleted_resource() RETURNS trigger
        LANGUAGE plpgsql AS $$ BEGIN
        IF TG_TABLE_NAME='regimex_snapshots' THEN
            IF EXISTS (SELECT 1 FROM regimex_deleted_snapshots d
                WHERE d.snapshot_id=NEW.snapshot_id OR
                (d.owner_id,d.source_id,d.run_id,d.source_snapshot_id)=
                (NEW.owner_id,NEW.source_id,NEW.run_id,NEW.source_snapshot_id)) THEN
                RAISE EXCEPTION 'snapshot identity was permanently removed';
            END IF;
        ELSIF TG_TABLE_NAME='regimex_analyses' THEN
            IF EXISTS (SELECT 1 FROM regimex_deleted_analyses d
                WHERE d.analysis_id=NEW.analysis_id) THEN
                RAISE EXCEPTION 'analysis identity was permanently removed';
            END IF;
        ELSE
            IF EXISTS (SELECT 1 FROM regimex_deleted_object_keys d
                WHERE d.object_key=NEW.object_key) THEN
                RAISE EXCEPTION 'object key was permanently removed';
            END IF;
        END IF;
        RETURN NEW;
        END $$""",
    """CREATE TRIGGER regimex_snapshot_deletion_guard BEFORE INSERT OR UPDATE
        ON regimex_snapshots FOR EACH ROW EXECUTE FUNCTION regimex_guard_deleted_resource()""",
    """CREATE TRIGGER regimex_analysis_deletion_guard BEFORE INSERT OR UPDATE
        ON regimex_analyses FOR EACH ROW EXECUTE FUNCTION regimex_guard_deleted_resource()""",
    """CREATE TRIGGER regimex_object_key_deletion_guard BEFORE INSERT OR UPDATE
        ON regimex_objects FOR EACH ROW EXECUTE FUNCTION regimex_guard_deleted_resource()""",
    """CREATE OR REPLACE FUNCTION regimex_reject_report_rewrite() RETURNS trigger
        LANGUAGE plpgsql AS $$ BEGIN
        IF TG_OP='DELETE' AND EXISTS(SELECT 1 FROM regimex_deleted_analyses d
            WHERE d.analysis_id=OLD.analysis_id) THEN RETURN OLD; END IF;
        IF TG_OP='UPDATE' AND
            (to_jsonb(NEW)-'document')=(to_jsonb(OLD)-'document') AND
            (NEW.document #- '{object,version_id}')=(OLD.document #- '{object,version_id}')
            THEN RETURN NEW; END IF;
        RAISE EXCEPTION 'analysis reports are immutable';
        END $$""",
)


def upgrade() -> None:
    for statement in STATEMENTS:
        op.execute(statement)


def downgrade() -> None:
    raise RuntimeError("Database downgrades are not supported.")
