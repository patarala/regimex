"""Create the service schema, transactional catalog and immutable resource guards."""

from alembic import op

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None

STATEMENTS = (
    """CREATE TABLE regimex_queue_schema (
        singleton boolean PRIMARY KEY CHECK (singleton), schema_sha256 text NOT NULL)""",
    """CREATE TABLE regimex_tokens (
        token_id uuid PRIMARY KEY, token_sha256 text NOT NULL UNIQUE,
        owner_id text NOT NULL,
        role text NOT NULL CHECK (role IN ('user','admin','integration')),
        source_namespace text, created_at timestamptz NOT NULL,
        revoked_at timestamptz,
        CHECK ((role = 'integration') = (source_namespace IS NOT NULL)))""",
    """CREATE TABLE regimex_objects (
        object_key text PRIMARY KEY, sha256 text NOT NULL,
        size_bytes bigint NOT NULL CHECK (size_bytes >= 0), version_id text NOT NULL,
        document jsonb NOT NULL)""",
    """CREATE TABLE regimex_snapshots (
        snapshot_id uuid PRIMARY KEY, owner_id text NOT NULL,
        source_id text NOT NULL, run_id text NOT NULL, source_snapshot_id text NOT NULL,
        content_sha256 text NOT NULL,
        object_key text NOT NULL REFERENCES regimex_objects(object_key),
        document jsonb NOT NULL,
        UNIQUE (owner_id, source_id, run_id, source_snapshot_id))""",
    """CREATE TABLE regimex_models (
        model_version_id text PRIMARY KEY, architecture_id text NOT NULL,
        bundle_sha256 text NOT NULL,
        manifest_key text NOT NULL REFERENCES regimex_objects(object_key),
        document jsonb NOT NULL)""",
    """CREATE TABLE regimex_model_members (
        model_version_id text NOT NULL REFERENCES regimex_models(model_version_id),
        member_path text NOT NULL,
        object_key text NOT NULL REFERENCES regimex_objects(object_key),
        PRIMARY KEY (model_version_id, member_path))""",
    """CREATE TABLE regimex_deployment (
        singleton boolean PRIMARY KEY CHECK (singleton),
        model_version_id text NOT NULL REFERENCES regimex_models(model_version_id))""",
    """CREATE TABLE regimex_operations (
        operation_id uuid PRIMARY KEY, owner_id text NOT NULL,
        kind text NOT NULL CHECK (kind IN (
            'generation','reference','training','evaluation','analysis','publication')),
        idempotency_key text NOT NULL, request_sha256 text NOT NULL,
        state text NOT NULL CHECK (state IN (
            'queued','running','cancel_requested','cancelled','succeeded','failed')),
        heartbeat_at timestamptz, document jsonb NOT NULL,
        UNIQUE (owner_id, kind, idempotency_key))""",
    """CREATE INDEX regimex_operations_owner_state
        ON regimex_operations (owner_id, state)""",
    """CREATE INDEX regimex_operations_recovery
        ON regimex_operations (heartbeat_at) WHERE state IN ('running','cancel_requested')""",
    """CREATE TABLE regimex_deliveries (
        operation_id uuid NOT NULL REFERENCES regimex_operations(operation_id),
        job_id bigint PRIMARY KEY)""",
    """CREATE INDEX regimex_deliveries_operation ON regimex_deliveries (operation_id)""",
    """CREATE TABLE regimex_analyses (
        analysis_id uuid PRIMARY KEY,
        operation_id uuid NOT NULL UNIQUE REFERENCES regimex_operations(operation_id),
        snapshot_id uuid NOT NULL REFERENCES regimex_snapshots(snapshot_id),
        model_version_id text NOT NULL REFERENCES regimex_models(model_version_id),
        result json CHECK (result IS NULL OR json_typeof(result) = 'object'))""",
    """CREATE TABLE regimex_catalog_bindings (
        snapshot_id uuid PRIMARY KEY REFERENCES regimex_snapshots(snapshot_id),
        source_namespace text NOT NULL, group_namespace text NOT NULL,
        group_id text NOT NULL, seed bigint, document jsonb NOT NULL)""",
    """CREATE TABLE regimex_datasets (
        dataset_id text PRIMARY KEY, owner_id text NOT NULL,
        content_sha256 text NOT NULL, split_sha256 text NOT NULL,
        generation_provenance_key text REFERENCES regimex_objects(object_key),
        document json NOT NULL, UNIQUE (owner_id,dataset_id))""",
    """CREATE TABLE regimex_dataset_members (
        dataset_id text NOT NULL REFERENCES regimex_datasets(dataset_id),
        position integer NOT NULL CHECK (position >= 0),
        snapshot_id uuid NOT NULL REFERENCES regimex_catalog_bindings(snapshot_id),
        partition text NOT NULL CHECK (partition IN ('train','validation','test')),
        PRIMARY KEY (dataset_id, position), UNIQUE (dataset_id, snapshot_id))""",
    """CREATE TABLE regimex_reference_sets (
        reference_set_id text PRIMARY KEY, owner_id text NOT NULL,
        dataset_id text NOT NULL REFERENCES regimex_datasets(dataset_id),
        partition text NOT NULL CHECK (partition IN ('train','validation','test')),
        content_sha256 text NOT NULL, document json NOT NULL,
        UNIQUE (owner_id,reference_set_id))""",
    """CREATE TABLE regimex_reference_objects (
        reference_set_id text NOT NULL REFERENCES regimex_reference_sets(reference_set_id),
        position integer NOT NULL CHECK (position >= 0),
        source_namespace text NOT NULL, source_snapshot_id text NOT NULL,
        object_key text NOT NULL REFERENCES regimex_objects(object_key),
        PRIMARY KEY (reference_set_id, position),
        UNIQUE (reference_set_id,source_namespace,source_snapshot_id))""",
    """CREATE TABLE regimex_admin_runs (
        operation_id uuid PRIMARY KEY REFERENCES regimex_operations(operation_id),
        kind text NOT NULL CHECK (kind IN (
            'generation','reference','training','evaluation','publication')),
        request json NOT NULL, output json
            CHECK (output IS NULL OR json_typeof(output) = 'object'))""",
    """CREATE TABLE regimex_generation_targets (
        dataset_id text PRIMARY KEY,
        operation_id uuid NOT NULL UNIQUE REFERENCES regimex_admin_runs(operation_id))""",
    """CREATE TABLE regimex_reference_targets (
        reference_set_id text PRIMARY KEY,
        operation_id uuid NOT NULL UNIQUE REFERENCES regimex_admin_runs(operation_id))""",
    """CREATE TABLE regimex_training_runs (
        operation_id uuid PRIMARY KEY REFERENCES regimex_admin_runs(operation_id),
        dataset_id text NOT NULL REFERENCES regimex_datasets(dataset_id),
        labels_reference_set_id text NOT NULL
            REFERENCES regimex_reference_sets(reference_set_id),
        validation_reference_set_id text NOT NULL
            REFERENCES regimex_reference_sets(reference_set_id),
        model_version_id text NOT NULL UNIQUE,
        committed_model_version_id text UNIQUE REFERENCES regimex_models(model_version_id))""",
    """CREATE TABLE regimex_evaluation_requests (
        operation_id uuid PRIMARY KEY REFERENCES regimex_admin_runs(operation_id),
        model_version_id text NOT NULL REFERENCES regimex_models(model_version_id),
        dataset_id text NOT NULL REFERENCES regimex_datasets(dataset_id),
        reference_set_id text NOT NULL REFERENCES regimex_reference_sets(reference_set_id))""",
    """CREATE TABLE regimex_evaluation_records (
        run_id uuid PRIMARY KEY REFERENCES regimex_admin_runs(operation_id),
        owner_id text NOT NULL,
        model_version_id text NOT NULL REFERENCES regimex_models(model_version_id),
        dataset_id text NOT NULL REFERENCES regimex_datasets(dataset_id),
        reference_set_id text NOT NULL REFERENCES regimex_reference_sets(reference_set_id),
        report_key text NOT NULL REFERENCES regimex_objects(object_key),
        document json NOT NULL)""",
    """CREATE TABLE regimex_publication_requests (
        operation_id uuid PRIMARY KEY REFERENCES regimex_admin_runs(operation_id),
        model_version_id text NOT NULL REFERENCES regimex_models(model_version_id),
        evaluation_run_id uuid NOT NULL REFERENCES regimex_evaluation_records(run_id))""",
    """CREATE INDEX regimex_datasets_owner ON regimex_datasets (owner_id,dataset_id)""",
    """CREATE INDEX regimex_reference_sets_owner
        ON regimex_reference_sets (owner_id,reference_set_id)""",
    """CREATE TABLE regimex_publication_audit (
        publication_id uuid PRIMARY KEY, owner_id text NOT NULL,
        previous_model_version_id text REFERENCES regimex_models(model_version_id),
        model_version_id text NOT NULL REFERENCES regimex_models(model_version_id),
        published_at timestamptz NOT NULL,
        evaluation_run_id uuid REFERENCES regimex_evaluation_records(run_id),
        operation_id uuid REFERENCES regimex_admin_runs(operation_id))""",
    """CREATE TABLE regimex_benchmark_revisions (
        revision bigint PRIMARY KEY CHECK (revision >= 0),
        sha256 text NOT NULL UNIQUE,
        previous_sha256 text REFERENCES regimex_benchmark_revisions(sha256),
        document jsonb NOT NULL,
        CHECK ((revision=0) = (previous_sha256 IS NULL)))""",
    """CREATE TABLE regimex_benchmark_bindings (
        benchmark_id text PRIMARY KEY, owner_id text NOT NULL,
        dataset_id text NOT NULL REFERENCES regimex_datasets(dataset_id),
        reference_set_id text NOT NULL REFERENCES regimex_reference_sets(reference_set_id),
        document jsonb NOT NULL)""",
    """CREATE TABLE regimex_benchmark_history_datasets (
        dataset_id text PRIMARY KEY REFERENCES regimex_datasets(dataset_id),
        owner_id text NOT NULL, dataset_sha256 text NOT NULL UNIQUE,
        registered_revision bigint NOT NULL
            REFERENCES regimex_benchmark_revisions(revision))""",
    """CREATE TABLE regimex_benchmark_choices (
        benchmark_id text PRIMARY KEY REFERENCES regimex_benchmark_bindings(benchmark_id),
        model_version_id text NOT NULL REFERENCES regimex_models(model_version_id),
        document jsonb NOT NULL)""",
    """CREATE FUNCTION regimex_reject_benchmark_rewrite() RETURNS trigger
        LANGUAGE plpgsql AS $$ BEGIN
        RAISE EXCEPTION 'benchmark records are immutable';
        END $$""",
    """CREATE TRIGGER regimex_benchmark_revision_immutable
        BEFORE UPDATE OR DELETE ON regimex_benchmark_revisions
        FOR EACH ROW EXECUTE FUNCTION regimex_reject_benchmark_rewrite()""",
    """CREATE TRIGGER regimex_benchmark_binding_immutable
        BEFORE UPDATE OR DELETE ON regimex_benchmark_bindings
        FOR EACH ROW EXECUTE FUNCTION regimex_reject_benchmark_rewrite()""",
    """CREATE TRIGGER regimex_benchmark_choice_immutable
        BEFORE UPDATE OR DELETE ON regimex_benchmark_choices
        FOR EACH ROW EXECUTE FUNCTION regimex_reject_benchmark_rewrite()""",
    """CREATE TRIGGER regimex_benchmark_history_immutable
        BEFORE UPDATE OR DELETE ON regimex_benchmark_history_datasets
        FOR EACH ROW EXECUTE FUNCTION regimex_reject_benchmark_rewrite()""",
    """CREATE INDEX regimex_benchmark_bindings_owner
        ON regimex_benchmark_bindings (owner_id,benchmark_id)""",
    """CREATE TABLE regimex_intake_keys (
        owner_id text NOT NULL,
        kind text NOT NULL CHECK (kind IN
            ('dataset_import','reference_import','final_reference_import')),
        idempotency_key text NOT NULL CHECK (length(idempotency_key) BETWEEN 1 AND 128),
        request_sha256 text NOT NULL CHECK (request_sha256 ~ '^[0-9a-f]{64}$'),
        dataset_id text, reference_set_id text,
        created_at timestamptz NOT NULL DEFAULT now(),
        PRIMARY KEY (owner_id,kind,idempotency_key),
        FOREIGN KEY (owner_id,dataset_id) REFERENCES regimex_datasets(owner_id,dataset_id),
        FOREIGN KEY (owner_id,reference_set_id)
            REFERENCES regimex_reference_sets(owner_id,reference_set_id),
        CHECK ((kind='dataset_import' AND dataset_id IS NOT NULL AND reference_set_id IS NULL)
            OR (kind!='dataset_import' AND dataset_id IS NULL
                AND reference_set_id IS NOT NULL)))""",
    """CREATE FUNCTION regimex_reject_intake_rewrite() RETURNS trigger
        LANGUAGE plpgsql AS $$ BEGIN
        RAISE EXCEPTION 'intake receipts are immutable';
        END $$""",
    """CREATE TRIGGER regimex_intake_key_immutable
        BEFORE UPDATE OR DELETE ON regimex_intake_keys
        FOR EACH ROW EXECUTE FUNCTION regimex_reject_intake_rewrite()""",
    """CREATE TABLE regimex_restore_state (
        singleton boolean PRIMARY KEY CHECK (singleton),
        ready boolean NOT NULL CHECK (ready))""",
    """INSERT INTO regimex_restore_state(singleton,ready) VALUES (true,true)""",
    """CREATE TABLE regimex_maintenance_identity (
        singleton boolean PRIMARY KEY CHECK (singleton),
        installation_id uuid NOT NULL UNIQUE,
        created_at timestamptz NOT NULL DEFAULT now(),
        restored_from_installation_id uuid,
        restored_from_backup_sha256 text
            CHECK (restored_from_backup_sha256 ~ '^[0-9a-f]{64}$'),
        CHECK ((restored_from_installation_id IS NULL) =
            (restored_from_backup_sha256 IS NULL)))""",
    """INSERT INTO regimex_maintenance_identity(singleton,installation_id)
        VALUES (true,gen_random_uuid())""",
    """CREATE TABLE regimex_orphan_plans (
        plan_id uuid PRIMARY KEY, installation_id uuid NOT NULL,
        owner_id text NOT NULL, store_scope_id text NOT NULL,
        sha256 text NOT NULL CHECK (sha256 ~ '^[0-9a-f]{64}$'),
        document jsonb NOT NULL, created_at timestamptz NOT NULL DEFAULT now())""",
    """CREATE TABLE regimex_object_delete_fences (
        object_key text NOT NULL, version_id text NOT NULL,
        sha256 text NOT NULL CHECK (sha256 ~ '^[0-9a-f]{64}$'),
        size_bytes bigint NOT NULL CHECK (size_bytes >= 0),
        plan_id uuid NOT NULL REFERENCES regimex_orphan_plans(plan_id),
        store_scope_id text NOT NULL,
        created_at timestamptz NOT NULL DEFAULT now(),
        PRIMARY KEY (object_key,version_id))""",
    """CREATE TABLE regimex_object_version_claims (
        object_key text NOT NULL, version_id text NOT NULL,
        state text NOT NULL CHECK (state IN ('registered','fenced')),
        PRIMARY KEY (object_key,version_id))""",
    """CREATE TABLE regimex_orphan_cleanup_events (
        event_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        plan_id uuid NOT NULL REFERENCES regimex_orphan_plans(plan_id),
        kind text NOT NULL CHECK (kind IN ('object','multipart')),
        object_key text NOT NULL, identity text NOT NULL,
        status text NOT NULL CHECK (status IN ('deleted','already_missing','changed','failed',
            'protected','mpu_aborted','mpu_already_missing','mpu_changed','mpu_failed')),
        occurred_at timestamptz NOT NULL DEFAULT now())""",
    """CREATE FUNCTION regimex_guard_deleted_object_version() RETURNS trigger
        LANGUAGE plpgsql AS $$ DECLARE version_state text; BEGIN
        INSERT INTO regimex_object_version_claims(object_key,version_id,state)
            VALUES (NEW.object_key,NEW.version_id,'registered')
            ON CONFLICT (object_key,version_id) DO UPDATE
                SET state=regimex_object_version_claims.state
            RETURNING state INTO version_state;
        IF version_state != 'registered' THEN
            RAISE EXCEPTION 'object version was fenced by authorized maintenance';
        END IF;
        RETURN NEW;
        END $$""",
    """CREATE TRIGGER regimex_object_delete_fence
        BEFORE INSERT OR UPDATE OF object_key,version_id ON regimex_objects
        FOR EACH ROW EXECUTE FUNCTION regimex_guard_deleted_object_version()""",
    """CREATE FUNCTION regimex_preserve_object_version_claim() RETURNS trigger
        LANGUAGE plpgsql AS $$ BEGIN
        IF TG_OP='DELETE' OR NEW IS DISTINCT FROM OLD THEN
            RAISE EXCEPTION 'object version claims cannot be changed or removed';
        END IF;
        RETURN NEW;
        END $$""",
    """CREATE TRIGGER regimex_object_version_claim_immutable
        BEFORE UPDATE OR DELETE ON regimex_object_version_claims
        FOR EACH ROW EXECUTE FUNCTION regimex_preserve_object_version_claim()""",
    """CREATE FUNCTION regimex_reject_retention_rewrite() RETURNS trigger
        LANGUAGE plpgsql AS $$ BEGIN
        RAISE EXCEPTION 'retention evidence is immutable';
        END $$""",
    """CREATE TRIGGER regimex_orphan_plan_immutable
        BEFORE UPDATE OR DELETE ON regimex_orphan_plans
        FOR EACH ROW EXECUTE FUNCTION regimex_reject_retention_rewrite()""",
    """CREATE TRIGGER regimex_object_delete_fence_immutable
        BEFORE UPDATE OR DELETE ON regimex_object_delete_fences
        FOR EACH ROW EXECUTE FUNCTION regimex_reject_retention_rewrite()""",
    """CREATE TRIGGER regimex_orphan_cleanup_event_immutable
        BEFORE UPDATE OR DELETE ON regimex_orphan_cleanup_events
        FOR EACH ROW EXECUTE FUNCTION regimex_reject_retention_rewrite()""",
    """CREATE INDEX regimex_orphan_cleanup_plan
        ON regimex_orphan_cleanup_events(plan_id,event_id)""",
    """CREATE TABLE regimex_sessions (
        session_id uuid PRIMARY KEY,
        token_id uuid NOT NULL REFERENCES regimex_tokens(token_id) ON DELETE CASCADE,
        session_sha256 text NOT NULL UNIQUE CHECK (session_sha256 ~ '^[0-9a-f]{64}$'),
        created_at timestamptz NOT NULL, expires_at timestamptz NOT NULL,
        CHECK (expires_at > created_at AND expires_at <= created_at + interval '8 hours'))""",
    """CREATE INDEX regimex_sessions_parent ON regimex_sessions(token_id,expires_at)""",
    """CREATE TABLE regimex_integration_subjects (
        integration_owner_id text NOT NULL, source_namespace text NOT NULL,
        subject_sha256 text NOT NULL CHECK (subject_sha256 ~ '^[0-9a-f]{64}$'),
        effective_owner_id text NOT NULL UNIQUE,
        created_at timestamptz NOT NULL DEFAULT now(),
        PRIMARY KEY(integration_owner_id,source_namespace,subject_sha256))""",
    """CREATE FUNCTION regimex_reject_subject_rewrite() RETURNS trigger
        LANGUAGE plpgsql AS $$ BEGIN
        RAISE EXCEPTION 'integration subject bindings are immutable';
        END $$""",
    """CREATE TRIGGER regimex_integration_subject_immutable
        BEFORE UPDATE OR DELETE ON regimex_integration_subjects
        FOR EACH ROW EXECUTE FUNCTION regimex_reject_subject_rewrite()""",
    """CREATE TABLE regimex_analysis_reports (
        analysis_id uuid PRIMARY KEY REFERENCES regimex_analyses(analysis_id),
        snapshot_id uuid NOT NULL REFERENCES regimex_snapshots(snapshot_id),
        owner_id text NOT NULL,
        model_version_id text NOT NULL REFERENCES regimex_models(model_version_id),
        result_sha256 text NOT NULL CHECK(result_sha256 ~ '^[0-9a-f]{64}$'),
        object_key text NOT NULL REFERENCES regimex_objects(object_key),
        document jsonb NOT NULL)""",
    """CREATE FUNCTION regimex_reject_report_rewrite() RETURNS trigger
        LANGUAGE plpgsql AS $$ BEGIN
        IF TG_OP = 'UPDATE' AND
        (to_jsonb(NEW) - 'document') = (to_jsonb(OLD) - 'document') AND
        (NEW.document #- '{object,version_id}') = (OLD.document #- '{object,version_id}')
        THEN RETURN NEW; END IF;
        RAISE EXCEPTION 'analysis reports are immutable';
        END $$""",
    """CREATE TRIGGER regimex_analysis_report_immutable BEFORE UPDATE OR DELETE
        ON regimex_analysis_reports FOR EACH ROW
        EXECUTE FUNCTION regimex_reject_report_rewrite()""",
)


def upgrade() -> None:
    for statement in STATEMENTS:
        op.execute(statement)


def downgrade() -> None:
    raise RuntimeError("Destructive downgrades require a coordinated database/object restore.")
