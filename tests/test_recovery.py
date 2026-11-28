"""Coordinated PostgreSQL/object restoration and fail-closed partial targets."""

import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from pydantic import JsonValue, SecretStr

psycopg = pytest.importorskip("psycopg")
pytest.importorskip("alembic")
pytest.importorskip("procrastinate")

from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo
from test_object_store_contract import cleanup_owned_test_prefix, s3_configuration

from regimex.application import recovery, recovery_objects
from regimex.application.service import AnalysisService
from regimex.directions import DEFAULT_ANALYSIS_PROFILE
from regimex.exports import ExportPayloadV2, ExportV2, seal_export
from regimex.infrastructure.database import DatabaseRepository
from regimex.infrastructure.maintenance import ServicesStoppedAttestation
from regimex.infrastructure.object_store import (
    LocalObjectStore,
    ObjectCapacityError,
    ObjectInfo,
    ObjectIntegrityError,
    ObjectStore,
)
from regimex.infrastructure.s3_object_store import S3ObjectStore
from regimex.model_registry import ArchitectureRegistry
from regimex.recovery_contracts import BackupResult, PostgresTools, RestoreReceipt
from regimex.service_contracts import Conflict, ServiceError
from regimex.storage import digest, json_bytes

DatabaseFactory = Callable[[str], DatabaseRepository]


def stopped() -> ServicesStoppedAttestation:
    return ServicesStoppedAttestation(
        stopped_at=datetime.now(UTC),
        statement="This test owns the disposable database/store; every service process is stopped.",
        api_stopped=True,
        analysis_workers_stopped=True,
        administrative_workers_stopped=True,
        maintenance_stopped=True,
        garbage_collection_stopped=True,
        exclusive_object_access=True,
    )


def local(root: Path) -> LocalObjectStore:
    return LocalObjectStore(root, max_object_bytes=16 * 1024**2, max_total_bytes=128 * 1024**2)


def analysis_service(
    database: DatabaseRepository, store: ObjectStore, root: Path
) -> AnalysisService:
    return AnalysisService(database, store, ArchitectureRegistry.discover(), root)


@pytest.fixture(scope="module")
def tools() -> PostgresTools:
    try:
        return PostgresTools.discover()
    except ValueError, FileNotFoundError:
        pytest.skip(
            "offline recovery tests need matching PGDUMP_BIN/PGRESTORE_BIN or tools on PATH"
        )


@pytest.fixture(scope="module")
def databases() -> Iterator[DatabaseFactory]:
    maintenance = os.environ.get("REGIMEX_TEST_DATABASE_DSN")
    if not maintenance:
        pytest.skip(
            "set REGIMEX_TEST_DATABASE_DSN to a disposable PostgreSQL cluster with CREATEDB"
        )
    created = []

    def create(schema: str = "public") -> DatabaseRepository:
        name = "regimex_recovery_" + uuid4().hex
        with psycopg.connect(maintenance, autocommit=True) as connection:
            try:
                connection.execute(
                    sql.SQL("CREATE DATABASE {} TEMPLATE template0").format(sql.Identifier(name))
                )
            except psycopg.errors.InsufficientPrivilege:
                pytest.skip("offline restore needs CREATEDB on the disposable test cluster")
        created.append(name)
        return DatabaseRepository(make_conninfo(maintenance, dbname=name), schema=schema)

    try:
        yield create
    finally:
        with psycopg.connect(maintenance, autocommit=True) as connection:
            for name in reversed(created):
                connection.execute(
                    sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name))
                )


@dataclass
class MinimalBackup:
    source: DatabaseRepository
    service: AnalysisService
    store: LocalObjectStore
    directory: Path
    backup: BackupResult
    snapshot_id: UUID
    token: str = field(repr=False)


@pytest.fixture
def minimal(databases: DatabaseFactory, tools: PostgresTools, tmp_path: Path) -> MinimalBackup:
    database = databases("recovery_scope")
    database.migrate()
    store = local(tmp_path / "source-objects")
    service = analysis_service(database, store, tmp_path / "source-work")
    token = database.create_token("recovery-user", "user")
    export = ExportV2.model_validate_json(
        (Path(__file__).parent / "fixtures/experiment-v2.json").read_bytes()
    )
    snapshot = service.create_snapshot(database.authenticate(token), export)
    directory = tmp_path / "backup"
    result = recovery.backup_service(
        SecretStr(database._dsn),
        database.schema,
        store,
        directory,
        tools=tools,
        attestation=stopped(),
    )
    return MinimalBackup(database, service, store, directory, result, snapshot.snapshot_id, token)


def restore(
    minimal: MinimalBackup, target: DatabaseRepository, store: ObjectStore, tools: PostgresTools
) -> RestoreReceipt:
    return recovery.restore_service(
        SecretStr(target._dsn),
        store,
        minimal.directory,
        expected_manifest_sha256=minimal.backup.manifest_sha256,
        tools=tools,
        attestation=stopped(),
    )


def test_read_only_database_blocks_service_readiness_but_preserves_diagnostics(
    databases: DatabaseFactory, tmp_path: Path
) -> None:
    from regimex.service_diagnostics import diagnose
    from regimex.service_settings import ServiceSettings

    database = databases("service_readiness")
    database.migrate()
    assert database.readiness()
    with psycopg.connect(database._dsn, autocommit=True) as admin:
        admin.execute(
            sql.SQL("ALTER DATABASE {} SET default_transaction_read_only=on").format(
                sql.Identifier(admin.info.dbname)
            )
        )
    assert not database.readiness()
    with database.connection() as connection:
        assert connection.execute("SHOW transaction_read_only").fetchone() == ("on",)
    settings = ServiceSettings(
        database_dsn=SecretStr(database._dsn),
        database_schema=database.schema,
        object_root=tmp_path / "objects",
        work_root=tmp_path / "work",
    )
    checks = {check.name: check for check in diagnose(settings).checks}
    assert checks["database"].status == "failed" and "read-only" in checks["database"].message
    assert checks["schema"].status == checks["storage"].status == "passed"


def test_recovery_preserves_external_training_provenance_and_metrics(
    databases: DatabaseFactory, tools: PostgresTools, tmp_path: Path
) -> None:
    from test_baseline_backend import bundle

    from regimex.external_evaluations import ExternalEvaluation
    from regimex.infrastructure.admin_repository import AdminRepository
    from regimex.model_registry import InferenceRuntime

    source = databases("external_recovery")
    source.migrate()
    token = source.create_token("offline-training", "admin")
    principal = source.authenticate(token)
    store = local(tmp_path / "source")
    service = analysis_service(source, store, tmp_path / "work")
    activity_id = uuid4()
    # Existing installations can have terminal receipts in their backup inventory.
    with source.connection() as connection:
        receipt = connection.execute(
            "INSERT INTO regimex_terminal_activities "
            "(owner_id,activity_id,title,status,started_at,finished_at,exit_code,result_text) "
            "VALUES (%s,%s,'Fit','succeeded',now(),now(),0,'ready') RETURNING *",
            (principal.owner_id, activity_id),
        ).fetchone()
    assert receipt is not None
    _, _, _, checksum = bundle(tmp_path)
    evidence = ExternalEvaluation(
        kind="controlled_transfer",
        bundle_sha256=checksum,
        report_sha256="a" * 64,
        protocol_sha256="b" * 64,
        freeze_sha256="c" * 64,
        candidate="fixture",
        sample_count=10,
        interval_macro_f1=0.5,
        passed=False,
    )
    model = service.register_bundle(
        principal,
        tmp_path / "bundle",
        manifest_sha256=checksum,
        inference_runtime=InferenceRuntime(threads=1),
        record_training=True,
        external_evaluations=(evidence,),
    )
    original = AdminRepository(source)
    (run,) = original.list_runs(principal, kind="training")
    metrics = original.get_training_metrics(principal, run.run_id)
    directory = tmp_path / "backup"
    backup = recovery.backup_service(
        SecretStr(source._dsn), source.schema, store, directory, tools=tools, attestation=stopped()
    )
    target = databases(source.schema)
    restored_store = local(tmp_path / "restored")
    recovery.restore_service(
        SecretStr(target._dsn),
        restored_store,
        directory,
        expected_manifest_sha256=backup.manifest_sha256,
        tools=tools,
        attestation=stopped(),
    )
    restored = AdminRepository(target)
    with target.connection() as connection:
        assert (
            connection.execute(
                "SELECT * FROM regimex_terminal_activities WHERE owner_id=%s AND activity_id=%s",
                (principal.owner_id, activity_id),
            ).fetchone()
            == receipt
        )
    assert restored.get_run(target.authenticate(token), run.run_id) == run
    assert metrics is not None and restored.get_training_metrics(principal, run.run_id) == metrics
    restored_model = target.get_model(model.model_version_id)
    for path, member in restored_model.members.items():
        assert restored_store.stat(member.key) == member
        assert member.sha256 == model.members[path].sha256


def test_custom_schema_recovery_preserves_identity_and_rotates_storage(
    minimal: MinimalBackup,
    databases: DatabaseFactory,
    tools: PostgresTools,
    tmp_path: Path,
) -> None:
    target = databases(minimal.source.schema)
    store = local(tmp_path / "target-objects")
    receipt = restore(minimal, target, store, tools)
    assert target.readiness()
    principal = target.authenticate(minimal.token)
    restored = analysis_service(target, store, tmp_path / "target-work")
    assert restored.read_snapshot(principal, minimal.snapshot_id) == minimal.service.read_snapshot(
        minimal.source.authenticate(minimal.token), minimal.snapshot_id
    )
    assert receipt.changed_versions == receipt.object_count == 1
    assert receipt.new_registered_claims == 1
    assert minimal.backup.manifest.source_maintenance_identity is not None
    assert (
        receipt.new_installation_id
        != minimal.backup.manifest.source_maintenance_identity.installation_id
    )
    assert receipt.table_evidence == minimal.backup.manifest.tables
    assert all(
        path.stat().st_mode & 0o777 == 0o600
        for path in minimal.directory.rglob("*")
        if path.is_file()
    )
    assert minimal.token.encode() not in (minimal.directory / "manifest.json").read_bytes()
    assert minimal.source._dsn.encode() not in (minimal.directory / "manifest.json").read_bytes()


def test_cross_timezone_restore_through_a_service_without_object_scratch(
    minimal: MinimalBackup,
    databases: DatabaseFactory,
    tools: PostgresTools,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = databases(minimal.source.schema)
    parameters = conninfo_to_dict(target._dsn)
    name = parameters.pop("dbname")
    assert isinstance(name, str)
    with psycopg.connect(target._dsn, autocommit=True) as connection:
        connection.execute(
            sql.SQL("ALTER DATABASE {} SET timezone TO 'Europe/Moscow'").format(
                sql.Identifier(name)
            )
        )
    service_file = tmp_path / "pg_service.conf"
    service_file.write_text(f"[restore_target]\ndbname={name}\n")
    monkeypatch.setenv("PGSERVICEFILE", str(service_file))
    parameters["service"] = "restore_target"
    service_dsn = make_conninfo("", **parameters)
    store = local(tmp_path / "restore-objects")

    def unexpected_copy(*args: object, **kwargs: object) -> None:
        pytest.fail("Restored objects should be verified directly in the target store.")

    monkeypatch.setattr(store, "materialize", unexpected_copy)
    receipt = recovery.restore_service(
        SecretStr(service_dsn),
        store,
        minimal.directory,
        expected_manifest_sha256=minimal.backup.manifest_sha256,
        tools=tools,
        attestation=stopped(),
    )
    assert receipt.object_count == len(minimal.backup.manifest.objects) == 1
    assert target.readiness()


def test_recovery_uses_the_guard_endpoint_from_libpq_defaults(
    minimal: MinimalBackup,
    databases: DatabaseFactory,
    tools: PostgresTools,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = databases(minimal.source.schema)
    with psycopg.connect(minimal.source._dsn) as connection:
        info = connection.info
        for variable, value in (
            ("PGHOST", info.host),
            ("PGPORT", str(info.port)),
            ("PGUSER", info.user),
            ("PGPASSWORD", info.password),
        ):
            monkeypatch.setenv(variable, value)
        source = SecretStr(make_conninfo(dbname=info.dbname))
    directory = tmp_path / "libpq-backup"
    backup = recovery.backup_service(
        source, minimal.source.schema, minimal.store, directory, tools=tools, attestation=stopped()
    )
    with psycopg.connect(target._dsn) as connection:
        destination = SecretStr(make_conninfo(dbname=connection.info.dbname))
    store = local(tmp_path / "libpq-target")
    receipt = recovery.restore_service(
        destination,
        store,
        directory,
        expected_manifest_sha256=backup.manifest_sha256,
        tools=tools,
        attestation=stopped(),
    )
    assert target.readiness() and receipt.object_count == 1


def test_restore_object_larger_than_compose_tmpfs(
    minimal: MinimalBackup,
    databases: DatabaseFactory,
    tools: PostgresTools,
    tmp_path: Path,
) -> None:
    size = 65 * 1024**2
    source_store = LocalObjectStore(
        minimal.store.root, max_object_bytes=size, max_total_bytes=2 * size
    )
    payload = tmp_path / "weights"
    with payload.open("wb") as file:
        file.truncate(size)
    weights = source_store.put("models/large/weights", payload, expected_sha256=digest(payload))
    with minimal.source.connection() as connection:
        minimal.source._object(connection, weights)
    directory = tmp_path / "large-backup"
    saved = recovery.backup_service(
        SecretStr(minimal.source._dsn),
        minimal.source.schema,
        source_store,
        directory,
        tools=tools,
        attestation=stopped(),
    )
    target = databases(minimal.source.schema)
    target_store = LocalObjectStore(
        tmp_path / "large-target", max_object_bytes=size, max_total_bytes=2 * size
    )
    receipt = recovery.restore_service(
        SecretStr(target._dsn),
        target_store,
        directory,
        expected_manifest_sha256=saved.manifest_sha256,
        tools=tools,
        attestation=stopped(),
    )
    assert receipt.object_count == 2 and target.readiness()
    restored = target_store.stat(weights.key)
    assert (restored.size_bytes, restored.sha256) == (size, weights.sha256)


@pytest.mark.parametrize(
    "revision",
    [
        "0007_resource_deletion",
        "0008_analysis_metadata",
        "0009_training_run_names",
        "0010_evaluation_reuse",
        "0011_training_progress",
        "0012_training_metrics",
        "0013_training_archive",
        "0014_external_training",
        "0015_terminal_activity",
        "0016_s3_coordination",
    ],
)
def test_previous_schema_backup_can_be_restored_then_migrated(
    minimal: MinimalBackup,
    databases: DatabaseFactory,
    tools: PostgresTools,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    revision: str,
) -> None:
    with minimal.source.connection() as connection:
        connection.execute("DROP TABLE regimex_training_preparations")
        if revision < "0016_s3_coordination":
            connection.execute("DROP TABLE regimex_s3_pending_writes,regimex_s3_namespaces")
        if revision < "0015_terminal_activity":
            connection.execute("DROP TABLE regimex_terminal_activities")
        if revision < "0014_external_training":
            connection.execute(
                "ALTER TABLE regimex_training_runs DROP CONSTRAINT regimex_training_input_origin, "
                "DROP COLUMN external, DROP COLUMN external_evaluations, "
                "DROP COLUMN external_latency, "
                "ALTER COLUMN dataset_id SET NOT NULL, ALTER COLUMN labels_reference_set_id "
                "SET NOT NULL, ALTER COLUMN validation_reference_set_id SET NOT NULL"
            )
        if revision < "0013_training_archive":
            connection.execute("ALTER TABLE regimex_training_runs DROP COLUMN archived_at")
        if revision < "0012_training_metrics":
            connection.execute("DROP TABLE regimex_training_metrics")
            connection.execute("ALTER TABLE regimex_models DROP COLUMN weights_deletion_id")
            connection.execute(
                "ALTER TABLE regimex_deletion_plans ALTER COLUMN source_id SET NOT NULL, "
                "DROP CONSTRAINT regimex_deletion_plans_source_check, "
                "DROP CONSTRAINT regimex_deletion_plans_target_kind_check, "
                "ADD CONSTRAINT regimex_deletion_plans_target_kind_check "
                "CHECK (target_kind IN ('analysis','experiment'))"
            )
        if revision < "0011_training_progress":
            connection.execute("DROP TABLE regimex_training_progress")
        if revision < "0010_evaluation_reuse":
            connection.execute("DROP TABLE regimex_evaluation_keys")
            connection.execute("DROP INDEX regimex_evaluation_inputs")
        if revision in {"0007_resource_deletion", "0008_analysis_metadata"}:
            connection.execute(
                "ALTER TABLE regimex_training_runs "
                "DROP COLUMN display_name, DROP COLUMN name_revision"
            )
        if revision == "0007_resource_deletion":
            connection.execute("ALTER TABLE regimex_snapshots DROP COLUMN identifiers_search")
            connection.execute("ALTER TABLE regimex_analyses DROP COLUMN model_version_search")
            connection.execute("ALTER TABLE regimex_analyses DROP COLUMN retry_of")
        connection.execute("UPDATE alembic_version SET version_num=%s", (revision,))
    directory = tmp_path / "previous-backup"
    with monkeypatch.context() as patch:
        patch.setattr(recovery, "RECOVERY_REVISION", revision)
        saved = recovery.backup_service(
            SecretStr(minimal.source._dsn),
            minimal.source.schema,
            minimal.store,
            directory,
            tools=tools,
            attestation=stopped(),
        )
    data = json_bytes(saved.manifest.model_dump(mode="json", exclude={"evidence_timezone"}))
    (directory / "manifest.json").write_bytes(data)
    target = databases(minimal.source.schema)
    store = local(tmp_path / "previous-schema-objects")
    recovery.restore_service(
        SecretStr(target._dsn),
        store,
        directory,
        expected_manifest_sha256=hashlib.sha256(data).hexdigest(),
        tools=tools,
        attestation=stopped(),
    )
    assert not target.readiness()
    target.migrate()
    assert target.readiness()
    restored = analysis_service(target, store, tmp_path / "restored-work").read_snapshot(
        target.authenticate(minimal.token), minimal.snapshot_id
    )
    assert (
        restored.content_sha256
        == minimal.service.read_snapshot(
            minimal.source.authenticate(minimal.token), minimal.snapshot_id
        ).content_sha256
    )


def test_recovery_preserves_committed_usage_and_uncertain_upload_reservations(
    minimal: MinimalBackup,
    databases: DatabaseFactory,
    tools: PostgresTools,
    tmp_path: Path,
) -> None:
    from regimex.infrastructure.storage_quotas import StorageQuotaRepository
    from regimex.storage_contracts import StorageQuotaExceeded

    owner = minimal.source.authenticate(minimal.token)
    committed = minimal.service.storage_usage(owner).used_bytes
    quota = StorageQuotaRepository(
        minimal.source,
        max_owner_storage_bytes=committed + 7,
        max_integration_storage_bytes=committed + 7,
    )
    pending = ObjectInfo(
        key="interrupted/snapshot", sha256="a" * 64, size_bytes=7, version_id="unwritten"
    )
    quota.reserve(owner.owner_id, "simulator", pending)
    before = quota.owner_usage(owner)
    directory = tmp_path / "quota-backup"
    backup = recovery.backup_service(
        SecretStr(minimal.source._dsn),
        minimal.source.schema,
        minimal.store,
        directory,
        tools=tools,
        attestation=stopped(),
    )
    assert backup.manifest.tables["regimex_storage_reservations"].rows == 2
    target = databases(minimal.source.schema)
    store = local(tmp_path / "quota-target")
    recovery.restore_service(
        SecretStr(target._dsn),
        store,
        directory,
        expected_manifest_sha256=backup.manifest_sha256,
        tools=tools,
        attestation=stopped(),
    )
    restored = StorageQuotaRepository(
        target,
        max_owner_storage_bytes=committed + 7,
        max_integration_storage_bytes=committed + 7,
    )
    assert restored.owner_usage(target.authenticate(minimal.token)) == before
    restored.reserve(owner.owner_id, "simulator", pending)
    with pytest.raises(StorageQuotaExceeded):
        restored.reserve(owner.owner_id, "simulator", pending.model_copy(update={"key": "new"}))


def test_recovery_preserves_integration_access_subjects_and_request_budgets(
    minimal: MinimalBackup,
    databases: DatabaseFactory,
    tools: PostgresTools,
    tmp_path: Path,
) -> None:
    from regimex.access_contracts import IntegrationKeyCreate, IntegrationUpdate
    from regimex.infrastructure.auth_repository import AuthRepository
    from regimex.infrastructure.integration_limits import (
        IntegrationLimitsRepository,
        IntegrationRateLimited,
    )
    from regimex.infrastructure.integration_repository import IntegrationRepository
    from regimex.service_contracts import Unauthorized

    source = minimal.source
    admin_token = source.create_token("recovery-admin", "admin")
    admin = source.authenticate(admin_token)
    export = minimal.service.read_snapshot(source.authenticate(minimal.token), minimal.snapshot_id)
    raw_key = source.create_token(
        "recovery-platform", "integration", source_namespace=export.source.source_id
    )
    delegated = AuthRepository(source).bind_subject(raw_key, "alice", export.source.run_id)
    assert delegated.integration_id is not None
    integration_id = delegated.integration_id
    snapshot = minimal.service.create_snapshot(delegated, export)
    management = IntegrationRepository(source)
    revoked = management.issue_key(
        admin, integration_id, IntegrationKeyCreate(name="Retired", scopes=("read",))
    )
    management.revoke_key(admin, integration_id, revoked.key.token_id)
    limits = IntegrationLimitsRepository(
        source, requests_per_minute=1, subject_requests_per_minute=1
    )
    limits.consume(delegated)
    integration = management.get(admin, integration_id)
    paused = management.update(
        admin,
        integration_id,
        IntegrationUpdate(expected_revision=integration.revision, enabled=False),
    )
    keys = management.list_keys(admin, integration_id)
    events = management.events(admin, integration_id)
    directory = tmp_path / "integration-backup"
    backup = recovery.backup_service(
        SecretStr(source._dsn),
        source.schema,
        minimal.store,
        directory,
        tools=tools,
        attestation=stopped(),
    )
    for table in (
        "regimex_integrations",
        "regimex_integration_subjects",
        "regimex_access_events",
        "regimex_integration_request_windows",
    ):
        assert backup.manifest.tables[table].rows > 0

    target = databases(source.schema)
    store = local(tmp_path / "integration-target")
    receipt = recovery.restore_service(
        SecretStr(target._dsn),
        store,
        directory,
        expected_manifest_sha256=backup.manifest_sha256,
        tools=tools,
        attestation=stopped(),
    )
    assert receipt.table_evidence == backup.manifest.tables
    target_admin = target.authenticate(admin_token)
    restored_management = IntegrationRepository(target)
    assert restored_management.get(target_admin, integration_id) == paused
    assert restored_management.list_keys(target_admin, integration_id) == keys
    assert restored_management.events(target_admin, integration_id) == events
    with pytest.raises(Unauthorized):
        target.authenticate(raw_key)
    restored_management.update(
        target_admin,
        integration_id,
        IntegrationUpdate(expected_revision=paused.revision, enabled=True),
    )
    with pytest.raises(Unauthorized):
        target.authenticate(revoked.token.get_secret_value())
    restored_principal = AuthRepository(target).bind_subject(raw_key, "alice", export.source.run_id)
    assert restored_principal == delegated
    assert (
        analysis_service(target, store, tmp_path / "integration-work").read_snapshot(
            restored_principal, snapshot.snapshot_id
        )
        == export
    )
    with target.connection() as connection:
        assert connection.execute(
            "SELECT count(*),sum(request_count) FROM regimex_integration_request_windows"
        ).fetchone() == (2, 2)
        # Restore duration is unrelated to the rate-limit behavior being checked.
        # Advance the restored window's test clock without changing its spent budget.
        connection.execute(
            "UPDATE regimex_integration_request_windows SET started_at=clock_timestamp()"
        )
    with pytest.raises(IntegrationRateLimited):
        IntegrationLimitsRepository(
            target, requests_per_minute=1, subject_requests_per_minute=1
        ).consume(restored_principal)


def test_backup_evidence_distinguishes_signed_zero_and_float_integer_types(
    minimal: MinimalBackup,
    databases: DatabaseFactory,
    tools: PostgresTools,
    tmp_path: Path,
) -> None:
    from psycopg.types.json import Json

    payloads = (
        {"zero": -0.0, "large": 1e20},
        {"zero": 0.0, "large": 1e20},
        {"zero": -0.0, "large": 100000000000000000000},
    )
    proofs = []
    for index, payload in enumerate(payloads):
        with minimal.source.connection() as connection:
            if index == 0:
                connection.execute("CREATE TABLE regimex_numeric_probe (payload json NOT NULL)")
                connection.execute(
                    "INSERT INTO regimex_numeric_probe VALUES (%s)", (Json(payload),)
                )
            else:
                connection.execute("UPDATE regimex_numeric_probe SET payload=%s", (Json(payload),))
        directory = tmp_path / f"numeric-backup-{index}"
        backup = recovery.backup_service(
            SecretStr(minimal.source._dsn),
            minimal.source.schema,
            minimal.store,
            directory,
            tools=tools,
            attestation=stopped(),
        )
        proofs.append(backup.manifest.tables["regimex_numeric_probe"].portable_sha256)
        target = databases(minimal.source.schema)
        store = local(tmp_path / f"numeric-restored-{index}")
        recovery.restore_service(
            SecretStr(target._dsn),
            store,
            directory,
            expected_manifest_sha256=backup.manifest_sha256,
            tools=tools,
            attestation=stopped(),
        )
        with target.connection() as connection:
            stored = connection.execute("SELECT payload FROM regimex_numeric_probe").fetchone()
            assert stored is not None and json_bytes(stored[0]) == json_bytes(payload)
    assert len(set(proofs)) == len(payloads)


def test_unrecognized_database_revision_blocks_backup_and_cannot_be_stamped_current(
    minimal: MinimalBackup, tools: PostgresTools, tmp_path: Path
) -> None:
    from alembic.util.exc import CommandError

    source = minimal.source
    with source.connection() as connection:
        connection.execute("UPDATE alembic_version SET version_num='unrecognized-schema'")
    assert not source.readiness()
    directory = tmp_path / "unsupported-backup"
    with pytest.raises(ValueError, match="current database schema revision"):
        recovery.backup_service(
            SecretStr(source._dsn),
            source.schema,
            minimal.store,
            directory,
            tools=tools,
            attestation=stopped(),
        )
    assert not (directory / "manifest.json").exists()
    with pytest.raises(CommandError, match="unrecognized-schema"):
        source.migrate()
    assert not source.readiness()
    with source.connection() as connection:
        assert connection.execute("SELECT version_num FROM alembic_version").fetchone() == (
            "unrecognized-schema",
        )
    export = ExportV2.model_validate_json(
        (Path(__file__).parent / "fixtures/experiment-v2.json").read_bytes()
    )
    assert (
        minimal.service.read_snapshot(source.authenticate(minimal.token), minimal.snapshot_id)
        == export
    )


@pytest.mark.parametrize("distributed", [False, True])
def test_s3_backup_round_trip_into_new_local_storage(
    minimal: MinimalBackup,
    databases: DatabaseFactory,
    tools: PostgresTools,
    tmp_path: Path,
    distributed: bool,
) -> None:
    from regimex.infrastructure.s3_coordination import S3Coordinator

    middle = databases(minimal.source.schema)
    config = s3_configuration(tmp_path / "s3-coordination")
    if distributed:
        config = config.model_copy(update={"coordination": "postgres"})
    remote = S3ObjectStore(
        config,
        coordinator=S3Coordinator(middle) if distributed else None,
        max_object_bytes=16 * 1024**2,
        max_total_bytes=128 * 1024**2,
    )
    try:
        first = restore(minimal, middle, remote, tools)
        if distributed:
            payload = tmp_path / "small-object"
            payload.write_bytes(b"verified")
            version = remote.put("roundtrip", payload, expected_sha256=digest(payload))
            remote.delete(version)
        directory = tmp_path / "s3-source-backup"
        backup = recovery.backup_service(
            SecretStr(middle._dsn),
            middle.schema,
            remote,
            directory,
            tools=tools,
            attestation=stopped(),
        )
        target = databases(minimal.source.schema)
        objects = local(tmp_path / "final-local-objects")
        second = recovery.restore_service(
            SecretStr(target._dsn),
            objects,
            directory,
            expected_manifest_sha256=backup.manifest_sha256,
            tools=tools,
            attestation=stopped(),
        )
        assert target.readiness() and second.changed_versions == first.changed_versions == 1
        assert second.new_installation_id != first.new_installation_id
        restored = analysis_service(target, objects, tmp_path / "final-work")
        assert restored.read_snapshot(target.authenticate(minimal.token), minimal.snapshot_id) == (
            minimal.service.read_snapshot(
                minimal.source.authenticate(minimal.token), minimal.snapshot_id
            )
        )
        assert second.table_evidence == backup.manifest.tables
        assert backup.manifest.source_maintenance_identity is not None
        assert (
            backup.manifest.source_maintenance_identity.installation_id == first.new_installation_id
        )
    finally:
        cleanup_owned_test_prefix(remote)


@pytest.mark.parametrize("member", ("database.dump", "object"))
def test_corruption_is_rejected_before_restore_writes(
    minimal: MinimalBackup,
    databases: DatabaseFactory,
    tools: PostgresTools,
    tmp_path: Path,
    member: str,
) -> None:
    target = databases(minimal.source.schema)
    store = local(tmp_path / "target-objects")
    victim = minimal.directory / (
        "objects/" + minimal.backup.manifest.objects[0].sha256 if member == "object" else member
    )
    original = victim.read_bytes()
    victim.write_bytes(bytes((original[0] ^ 1,)) + original[1:])
    with pytest.raises(ValueError):
        restore(minimal, target, store, tools)
    store.assert_empty_store()
    with psycopg.connect(target._dsn) as connection:
        assert connection.execute(
            "SELECT count(*) FROM pg_tables "
            "WHERE schemaname NOT IN ('pg_catalog','information_schema')"
        ).fetchone() == (0,)


def test_restore_refuses_existing_database_or_store(
    minimal: MinimalBackup,
    databases: DatabaseFactory,
    tools: PostgresTools,
    tmp_path: Path,
) -> None:
    occupied = databases(minimal.source.schema)
    with psycopg.connect(occupied._dsn) as connection:
        connection.execute("CREATE TABLE public.preserved (value integer)")
        connection.execute("INSERT INTO public.preserved VALUES (17)")
    store = local(tmp_path / "target-objects")
    with pytest.raises(ValueError, match="existing"):
        restore(minimal, occupied, store, tools)
    store.assert_empty_store()
    with psycopg.connect(occupied._dsn) as connection:
        assert connection.execute("SELECT value FROM public.preserved").fetchone() == (17,)
    existing = tmp_path / "prior"
    existing.write_bytes(b"preexisting target content")
    prior = store.put("preserved", existing, expected_sha256=digest(existing))
    target = databases(minimal.source.schema)
    with pytest.raises(ValueError, match="empty"):
        restore(minimal, target, store, tools)
    assert store.stat(prior.key) == prior
    with psycopg.connect(target._dsn) as connection:
        assert connection.execute(
            "SELECT to_regclass(%s)", (target.schema + ".regimex_restore_state",)
        ).fetchone() == (None,)


def test_partial_object_restore_stays_closed_after_migration_and_restart(
    minimal: MinimalBackup,
    databases: DatabaseFactory,
    tools: PostgresTools,
    tmp_path: Path,
) -> None:
    owner = minimal.source.authenticate(minimal.token)
    export = minimal.service.read_snapshot(owner, minimal.snapshot_id)
    another_snapshot = export.source.model_copy(update={"snapshot_id": "another-snapshot"})
    minimal.service.create_snapshot(
        owner, seal_export(export.model_copy(update={"source": another_snapshot}))
    )
    directory = tmp_path / "two-snapshot-backup"
    backup = recovery.backup_service(
        SecretStr(minimal.source._dsn),
        minimal.source.schema,
        minimal.store,
        directory,
        tools=tools,
        attestation=stopped(),
    )
    first, second = backup.manifest.objects
    capacity = max(first.size_bytes, second.size_bytes)
    target = databases(minimal.source.schema)
    store = LocalObjectStore(
        tmp_path / "target-objects", max_object_bytes=capacity, max_total_bytes=capacity
    )
    with pytest.raises(ObjectCapacityError):
        recovery.restore_service(
            SecretStr(target._dsn),
            store,
            directory,
            expected_manifest_sha256=backup.manifest_sha256,
            tools=tools,
            attestation=stopped(),
        )
    assert store.stat(first.key).sha256 == first.sha256
    with pytest.raises(FileNotFoundError):
        store.stat(second.key)
    target.migrate()
    restarted = DatabaseRepository(target._dsn, schema=target.schema)
    assert not restarted.readiness()
    with pytest.raises(ServiceError) as blocked:
        restarted.create_token("new-user", "user")
    assert (blocked.value.status_code, blocked.value.code) == (503, "restore_incomplete")


@pytest.mark.parametrize(
    "statement",
    ['CREATE COLLATION public.saved FROM "C"', "SELECT lo_create(0)"],
)
def test_restore_checks_other_database_objects_before_writing(
    minimal: MinimalBackup,
    databases: DatabaseFactory,
    tools: PostgresTools,
    tmp_path: Path,
    statement: str,
) -> None:
    target = databases(minimal.source.schema)
    with psycopg.connect(target._dsn) as connection:
        connection.execute(statement)
    store = local(tmp_path / "target-objects")
    with pytest.raises(ValueError, match="empty database"):
        restore(minimal, target, store, tools)
    store.assert_empty_store()
    assert not target.readiness()


@pytest.mark.parametrize("stage", ["materialize", "put", "stat"])
def test_object_transfer_checks_recovery_deadline_before_completion(
    minimal: MinimalBackup,
    databases: DatabaseFactory,
    tools: PostgresTools,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    target = databases(minimal.source.schema)
    store = local(tmp_path / "target-objects")
    adapter = minimal.store if stage == "materialize" else store
    transfer = getattr(adapter, stage)
    expired = False
    finished = False
    original_deadline = recovery_objects.check_deadline

    def deadline(limit: float) -> None:
        original_deadline(0 if expired else limit)

    def delayed(*args: object, **kwargs: object) -> ObjectInfo:
        nonlocal expired, finished
        expired = True
        result: ObjectInfo = transfer(*args, **kwargs)
        finished = True
        return result

    monkeypatch.setattr(recovery_objects, "check_deadline", deadline)
    monkeypatch.setattr(adapter, stage, delayed)
    destination = tmp_path / "interrupted-backup"
    with pytest.raises(TimeoutError, match="wall-time"):
        if stage == "materialize":
            recovery.backup_service(
                SecretStr(minimal.source._dsn),
                minimal.source.schema,
                adapter,
                destination,
                tools=tools,
                attestation=stopped(),
            )
        else:
            restore(minimal, target, store, tools)
    assert expired and not finished
    assert not (destination / "manifest.json").exists()
    assert not target.readiness()
    if stage != "stat":
        store.assert_empty_store()


def test_mid_rebind_failure_rolls_back_and_cannot_be_served_or_reseeded(
    minimal: MinimalBackup,
    databases: DatabaseFactory,
    tools: PostgresTools,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = databases(minimal.source.schema)
    store = local(tmp_path / "target-objects")
    original = recovery._rewrite_document

    def fail_after_ledger_change(
        table: str,
        document: dict[str, JsonValue],
        replacements: dict[str, ObjectInfo],
        old: dict[str, ObjectInfo],
    ) -> dict[str, JsonValue]:
        if table == "regimex_snapshots" and any(
            replacements[key].version_id != info.version_id
            and replacements[key].version_id != "portable-content"
            for key, info in old.items()
        ):
            raise ValueError("injected failure before resource visibility")
        return original(table, document, replacements, old)

    monkeypatch.setattr(recovery, "_rewrite_document", fail_after_ledger_change)
    with pytest.raises(ValueError, match="injected"):
        restore(minimal, target, store, tools)
    assert not target.readiness()
    with psycopg.connect(target._dsn) as connection:
        connection.execute(sql.SQL("SET search_path TO {}").format(sql.Identifier(target.schema)))
        assert connection.execute("SELECT count(*) FROM regimex_restore_state").fetchone() == (0,)
        assert connection.execute("SELECT version_id FROM regimex_objects").fetchone() == (
            minimal.backup.manifest.objects[0].version_id,
        )
        assert connection.execute(
            "SELECT count(*) FROM regimex_object_version_claims"
        ).fetchone() == (1,)
    target.migrate()
    assert not target.readiness()
    with pytest.raises(ValueError, match="empty"):
        restore(minimal, target, local(tmp_path / "another-store"), tools)


def test_backup_refuses_a_queued_operation(
    minimal: MinimalBackup,
    tools: PostgresTools,
    tmp_path: Path,
) -> None:
    from regimex.admin_contracts import GenerationSpec, RawGenerationPlan
    from regimex.application.admin_service import AdminService
    from regimex.infrastructure.admin_repository import AdminRepository
    from regimex.service_settings import ServiceSettings

    token = minimal.source.create_token("operator", "admin")
    service = AdminService(
        minimal.service,
        AdminRepository(minimal.source),
        ServiceSettings(database_dsn=SecretStr(minimal.source._dsn)),
    )
    with minimal.source.queue_app.open():
        service.submit_generation(
            minimal.source.authenticate(token),
            GenerationSpec(
                dataset_id="pending",
                plan=RawGenerationPlan(seed=18, experiment_count=3, step_count=64),
            ),
            idempotency_key="pending",
        )
    directory = tmp_path / "incomplete"
    with pytest.raises(Conflict, match="zero queued"):
        recovery.backup_service(
            SecretStr(minimal.source._dsn),
            minimal.source.schema,
            minimal.store,
            directory,
            tools=tools,
            attestation=stopped(),
        )
    assert not (directory / "manifest.json").exists()


def test_backup_verifies_each_revision_even_when_objects_share_bytes(
    minimal: MinimalBackup,
    tools: PostgresTools,
    tmp_path: Path,
) -> None:
    second_token = minimal.source.create_token("another-owner", "user")
    export = minimal.service.read_snapshot(
        minimal.source.authenticate(minimal.token), minimal.snapshot_id
    )
    second = minimal.service.create_snapshot(minimal.source.authenticate(second_token), export)
    assert second.object.sha256 == minimal.backup.manifest.objects[0].sha256
    path = minimal.store.root / second.object.key
    path.touch()
    with pytest.raises(ObjectIntegrityError):
        recovery.backup_service(
            SecretStr(minimal.source._dsn),
            minimal.source.schema,
            minimal.store,
            tmp_path / "duplicate-bytes-backup",
            tools=tools,
            attestation=stopped(),
        )


def environment(database: DatabaseRepository, store: ObjectStore, root: Path) -> dict[str, str]:
    result = {key: value for key, value in os.environ.items() if not key.startswith("REGIMEX_")}
    result.update(
        {
            "REGIMEX_DATABASE_DSN": database._dsn,
            "REGIMEX_DATABASE_SCHEMA": database.schema,
            "REGIMEX_WORK_ROOT": str(root),
            "REGIMEX_ALLOW_LOOPBACK_HTTP": "true",
            "REGIMEX_PUBLIC_ORIGIN": "http://127.0.0.1:8000",
            "REGIMEX_MAX_OBJECT_BYTES": str(16 * 1024**2),
            "REGIMEX_MAX_STORE_BYTES": str(128 * 1024**2),
        }
    )
    if isinstance(store, LocalObjectStore):
        result.update({"REGIMEX_OBJECT_BACKEND": "local", "REGIMEX_OBJECT_ROOT": str(store.root)})
    elif isinstance(store, S3ObjectStore):
        result["REGIMEX_OBJECT_BACKEND"] = "s3"
        for name, value in store.config.model_dump(mode="json").items():
            if value is not None:
                result["REGIMEX_S3_" + name.upper()] = (
                    str(value).lower() if isinstance(value, bool) else str(value)
                )
    else:
        raise AssertionError("Unexpected fixture backend")
    return result


@contextmanager
def process(root: Path, env: dict[str, str], *arguments: str) -> Iterator[subprocess.Popen[bytes]]:
    root.parent.mkdir(parents=True, exist_ok=True)
    with root.open("wb") as log:
        child = subprocess.Popen(
            [sys.executable, "-m", "regimex", "service", *arguments],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            yield child
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=5)
                    raise AssertionError("The fixture service did not stop gracefully.") from None


@dataclass
class DiscardedCandidate:
    run_id: UUID
    deletion_id: UUID
    weights_key: str
    metrics: dict[str, JsonValue]


@dataclass
class ScientificBackup:
    database: DatabaseRepository
    store: LocalObjectStore
    directory: Path
    backup: BackupResult
    admin_token: str = field(repr=False)
    user_token: str = field(repr=False)
    other_token: str = field(repr=False)
    snapshot_id: UUID
    predictions: dict[str, tuple[UUID, dict[str, JsonValue]]]
    reports: dict[str, str]
    registry: dict[str, JsonValue]
    discarded: DiscardedCandidate
    old_plan_id: UUID
    old_plan_sha256: str


@pytest.fixture(scope="module")
def scientific_backup(
    databases: DatabaseFactory, tools: PostgresTools, tmp_path_factory: pytest.TempPathFactory
) -> ScientificBackup:
    pytest.importorskip("numpy")
    pytest.importorskip("sklearn")
    pytest.importorskip("torch")
    pytest.importorskip("plotly")
    from test_admin_workflow import benchmarks, completed, make_admin, prepared_references
    from test_application_training import source

    from regimex.admin_contracts import (
        EvaluationSpec,
        GenerationSpec,
        PublicationSpec,
        RawGenerationPlan,
        TrainingSpec,
    )
    from regimex.application.deletion import DeletionService
    from regimex.application.retention import RetentionService
    from regimex.infrastructure.deletion_repository import DeletionRepository
    from regimex.infrastructure.retention_repository import RetentionRepository
    from regimex.model_registry import InferenceRuntime, TrainingPolicy, TrainingRuntime

    root = tmp_path_factory.mktemp("recovery-science")
    database = databases("public")
    database.migrate()
    admin_token = database.create_token("recovery-admin", "admin")
    user_token = database.create_token("recovery-user", "user")
    other_token = database.create_token("other-user", "user")
    principal = database.authenticate(admin_token)
    owner = database.authenticate(user_token)
    service = make_admin(database, root)
    original_jobs = {}
    discarded = None
    with database.queue_app.open():
        dataset, validation = prepared_references(service, principal)
        value = source(0).export.model_dump(mode="json", exclude={"content_sha256"})
        value["time_grid"]["origin"] = -0.0
        value["instruments"][0]["initial_price"] = 1e20
        numeric_export = seal_export(ExportPayloadV2.model_validate_json(json.dumps(value)))
        snapshot = service.analysis_service.create_snapshot(owner, numeric_export)
        previous = None
        for architecture, model in (
            ("baseline", "discarded-baseline"),
            ("baseline", "recovery-baseline"),
            ("tcn", "recovery-tcn"),
        ):
            tcn = architecture == "tcn"
            parameters: dict[str, JsonValue] = (
                {"channels": 4, "dilations": [1, 2], "dropout": 0.0, "smoothing": 0.0}
                if tcn
                else {"regularization": [1.0], "context_sizes": [4, 16]}
            )
            policy = (
                TrainingPolicy(
                    optimizer="adamw",
                    seeds=(11,),
                    epochs=1,
                    batch_size=1,
                    patience=1,
                    validation_interval=1,
                )
                if tcn
                else TrainingPolicy(optimizer="lbfgs", seeds=(11,))
            )
            training = service.submit_training(
                principal,
                TrainingSpec(
                    dataset_id=dataset.dataset_id,
                    labels_reference_set_id="fixture-labels",
                    validation_reference_set_id=validation.references.reference_set_id,
                    model_version_id=model,
                    architecture_id=architecture,
                    parameters=parameters,
                    policy=policy,
                    training_runtime=TrainingRuntime(threads=1, time_budget_seconds=90),
                    inference_runtime=InferenceRuntime(threads=1, max_batch_size=1),
                    analysis_profile_id=DEFAULT_ANALYSIS_PROFILE,
                    build_id="recovery-fixture",
                    benchmark_selection_id="fixture-final-benchmark",
                ),
                idempotency_key="fit-" + model,
            )
            completed(service, training.run_id)
            if model == "discarded-baseline":
                weights_key = database.get_model(model).members["weights.npz"].key
                metrics = service.training_metrics(principal, training.run_id)
                preview = service.preview_weight_deletion(principal, training.run_id)
                deletion = DeletionService(
                    DeletionRepository(database), service.analysis_service.store
                )
                removal = deletion.apply(principal, preview.plan_id, expected_sha256=preview.sha256)
                assert removal.state == "completed" and removal.reclaimed_bytes > 0
                service.repository.set_training_archive(
                    principal, (training.run_id,), archived=True
                )
                discarded = DiscardedCandidate(
                    training.run_id,
                    removal.deletion_id,
                    weights_key,
                    metrics.model_dump(mode="json"),
                )
                continue
            request = EvaluationSpec(
                model_version_id=model,
                dataset_id=dataset.dataset_id,
                reference_set_id=validation.references.reference_set_id,
            )
            evaluation = service.submit_evaluation(
                principal, request, idempotency_key="evaluate-" + architecture
            )
            completed(service, evaluation.run_id)
            reused = service.submit_evaluation(
                principal, request, idempotency_key="reuse-" + architecture
            )
            assert reused.run_id == evaluation.run_id
            publication = service.submit_publication(
                principal,
                PublicationSpec(
                    model_version_id=model,
                    evaluation_run_id=evaluation.run_id,
                    expected_previous=previous,
                ),
                idempotency_key="publish-" + architecture,
            )
            completed(service, publication.run_id)
            original_jobs[architecture] = service.analysis_service.submit_analysis(
                owner,
                snapshot.snapshot_id,
                model_version_id=model,
                idempotency_key="original-" + architecture,
            )
            previous = model
        generated = service.submit_generation(
            principal,
            GenerationSpec(
                dataset_id="generation-provenance",
                plan=RawGenerationPlan(seed=12, experiment_count=3, step_count=64),
            ),
            idempotency_key="generation",
        )
        completed(service, generated.run_id)
        benchmarks(service).freeze(principal, "fixture-final-benchmark", "recovery-tcn")
    with process(
        root / "source-analysis-worker.log",
        environment(database, service.analysis_service.store, root / "work"),
        "worker",
        "--queue",
        "analysis",
    ) as worker:
        deadline = time.monotonic() + 90
        while any(
            database.get_analysis(owner, job.analysis_id).operation.state != "succeeded"
            for job in original_jobs.values()
        ):
            assert worker.poll() is None and time.monotonic() < deadline
            time.sleep(0.1)
    predictions = {
        architecture: (job.analysis_id, service.analysis_service.get_result(owner, job.analysis_id))
        for architecture, job in original_jobs.items()
    }
    reports = {
        architecture: hashlib.sha256(
            service.analysis_service.read_report(owner, job.analysis_id)
        ).hexdigest()
        for architecture, job in original_jobs.items()
    }
    registry = benchmarks(service).get_registry(principal).model_dump(mode="json")
    assert isinstance(service.analysis_service.store, LocalObjectStore)
    store = service.analysis_service.store
    payload = root / "orphan"
    payload.write_bytes(b"A deliberate unregistered fixture for immutable GC history.")
    store.put("orphan-history", payload, expected_sha256=digest(payload))
    retention = RetentionService(
        RetentionRepository(database), store, clock=lambda: datetime.now(UTC) + timedelta(days=2)
    )
    plan = retention.plan(principal, attestation=stopped())
    receipt = retention.apply(
        principal, plan.plan_id, expected_sha256=plan.sha256, attestation=stopped()
    )
    assert receipt.complete and len(receipt.events) == 1
    directory = root / "backup"
    result = recovery.backup_service(
        SecretStr(database._dsn),
        database.schema,
        store,
        directory,
        tools=tools,
        attestation=stopped(),
    )
    assert discarded is not None
    assert discarded.weights_key not in {info.key for info in result.manifest.objects}
    return ScientificBackup(
        database,
        store,
        directory,
        result,
        admin_token,
        user_token,
        other_token,
        snapshot.snapshot_id,
        predictions,
        reports,
        registry,
        discarded,
        plan.plan_id,
        plan.sha256,
    )


@pytest.mark.parametrize("backend", ("local", "s3", "s3-postgres"))
def test_full_recovery_reloads_both_models_in_fresh_service_processes(
    scientific_backup: ScientificBackup,
    databases: DatabaseFactory,
    tools: PostgresTools,
    tmp_path: Path,
    backend: str,
) -> None:
    from regimex.application.retention import RetentionService
    from regimex.infrastructure.retention_repository import RetentionRepository
    from regimex.web.report_runtime import preview_html

    original = scientific_backup
    target = databases(original.database.schema)
    objects: LocalObjectStore | S3ObjectStore
    if backend == "local":
        objects = local(tmp_path / "objects")
    else:
        from regimex.infrastructure.s3_coordination import S3Coordinator

        config = s3_configuration(tmp_path / "s3-coordination")
        coordinator = None
        if backend == "s3-postgres":
            config = config.model_copy(update={"coordination": "postgres"})
            coordinator = S3Coordinator(target)
        objects = S3ObjectStore(
            config,
            coordinator=coordinator,
            max_object_bytes=16 * 1024**2,
            max_total_bytes=128 * 1024**2,
        )
    try:
        receipt = recovery.restore_service(
            SecretStr(target._dsn),
            objects,
            original.directory,
            expected_manifest_sha256=original.backup.manifest_sha256,
            tools=tools,
            attestation=stopped(),
        )
        assert receipt.changed_versions == receipt.object_count > 0
        assert receipt.new_registered_claims == receipt.object_count
        assert receipt.table_evidence == original.backup.manifest.tables
        assert receipt.table_evidence["regimex_evaluation_keys"].rows == 2
        assert receipt.table_evidence["regimex_training_metrics"].rows == 3
        with pytest.raises(FileNotFoundError):
            objects.stat(original.discarded.weights_key)
        assert target.readiness()
        admin = target.authenticate(original.admin_token)
        assert original.backup.manifest.source_maintenance_identity is not None
        assert (
            receipt.new_installation_id
            != original.backup.manifest.source_maintenance_identity.installation_id
        )
        retention = RetentionService(RetentionRepository(target), objects)
        with pytest.raises(Conflict, match="another database installation"):
            retention.apply(
                admin,
                original.old_plan_id,
                expected_sha256=original.old_plan_sha256,
                attestation=stopped(),
            )
        with socket.socket() as selected:
            selected.bind(("127.0.0.1", 0))
            port = selected.getsockname()[1]
        base = f"http://127.0.0.1:{port}"
        env = environment(target, objects, tmp_path / "fresh-work")
        env["REGIMEX_PUBLIC_ORIGIN"] = base

        def http(
            path: str,
            *,
            body: dict[str, str] | None = None,
            token: str = original.user_token,
            expected: int = 200,
        ) -> dict[str, JsonValue]:
            headers = {"Authorization": "Bearer " + token, "Idempotency-Key": uuid4().hex}
            if body is not None:
                headers["Content-Type"] = "application/json"
            request = urllib.request.Request(
                base + path,
                data=None if body is None else json.dumps(body).encode(),
                headers=headers,
            )
            try:
                with urllib.request.urlopen(request, timeout=10) as response:
                    status, value = response.status, json.loads(response.read())
            except urllib.error.HTTPError as error:
                try:
                    status, value = error.code, json.loads(error.read())
                finally:
                    error.close()
            assert status == expected, (path, status, value)
            assert isinstance(value, dict)
            return value

        with (
            process(tmp_path / "api.log", env, "serve", "--port", str(port)) as api,
            process(tmp_path / "worker.log", env, "worker", "--queue", "analysis") as worker,
        ):
            for _ in range(150):
                try:
                    http("/health/ready")
                    break
                except OSError, urllib.error.URLError:
                    assert api.poll() is None
                    time.sleep(0.1)
            else:
                raise AssertionError("The restored API did not become ready.")
            assert (
                http("/api/v1/benchmarks/registry", token=original.admin_token) == original.registry
            )
            run_url = f"/api/v1/training-runs/{original.discarded.run_id}"
            restored_run = http(run_url, token=original.admin_token)
            assert restored_run["weights_deletion_id"] == str(original.discarded.deletion_id)
            assert restored_run["archived_at"] is not None
            assert (
                http(f"{run_url}/metrics", token=original.admin_token) == original.discarded.metrics
            )
            for architecture, (analysis_id, prediction) in original.predictions.items():
                actual_prediction = http(f"/api/v1/analyses/{analysis_id}/result")
                assert json_bytes(actual_prediction) == json_bytes(prediction)
                report_request = urllib.request.Request(
                    base + f"/api/v1/analyses/{analysis_id}/report",
                    headers={"Authorization": "Bearer " + original.user_token},
                )
                with urllib.request.urlopen(report_request, timeout=10) as report_response:
                    report_body = report_response.read(16 * 1024**2 + 1)
                    assert report_response.headers.get_content_type() == "text/html"
                    assert hashlib.sha256(report_body).hexdigest() == original.reports[architecture]
                preview_request = urllib.request.Request(
                    base + f"/api/v1/analyses/{analysis_id}/report/preview",
                    headers={"Authorization": "Bearer " + original.user_token},
                )
                with urllib.request.urlopen(preview_request, timeout=10) as preview_response:
                    assert preview_response.read(16 * 1024**2 + 1) == preview_html(report_body)
                http(
                    f"/api/v1/analyses/{analysis_id}/report",
                    token=original.other_token,
                    expected=404,
                )
                http(
                    f"/api/v1/analyses/{analysis_id}/result",
                    token=original.other_token,
                    expected=404,
                )
                queued = http(
                    "/api/v1/analyses",
                    body={
                        "snapshot_id": str(original.snapshot_id),
                        "model_version_id": "recovery-" + architecture,
                    },
                    expected=202,
                )
                deadline = time.monotonic() + 60
                while time.monotonic() < deadline:
                    assert worker.poll() is None
                    state = http(str(queued["polling_url"]))
                    job = state["job"]
                    assert isinstance(job, dict) and job["state"] not in {"failed", "cancelled"}, (
                        state
                    )
                    if job["state"] == "succeeded":
                        break
                    time.sleep(0.1)
                else:
                    raise AssertionError("The restored analysis worker timed out.")
                restored = http(str(state["result_url"]))
                assert restored["sequences"] == prediction["sequences"]
                assert restored["model"] == prediction["model"]
    finally:
        if isinstance(objects, S3ObjectStore):
            cleanup_owned_test_prefix(objects)


@pytest.mark.parametrize(
    "revision",
    ["unrecognized-schema", "9999_future", ""],
)
def test_unsupported_backup_revisions_are_rejected_before_target_access(
    minimal: MinimalBackup, tmp_path: Path, revision: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    backup = tmp_path / "unsupported-archive"
    backup.mkdir()
    document = minimal.backup.manifest.model_dump(mode="json")
    document["migration_revision"] = revision
    data = json_bytes(document)
    (backup / "manifest.json").write_bytes(data)
    store = local(tmp_path / "untouched-target")

    def forbidden_target_access() -> None:
        pytest.fail("An unsupported archive reached the target store.")

    monkeypatch.setattr(store, "assert_empty_store", forbidden_target_access)

    with pytest.raises(ValueError, match="schema validation"):
        recovery.restore_service(
            SecretStr("invalid target DSN must not be read"),
            store,
            backup,
            expected_manifest_sha256=hashlib.sha256(data).hexdigest(),
            tools=PostgresTools(
                pg_dump=tmp_path / "absent-dump", pg_restore=tmp_path / "absent-restore"
            ),
            attestation=stopped(),
        )


def test_postgres_version_probe_honors_remaining_recovery_time(tmp_path: Path) -> None:
    executable = tmp_path / "pg_dump"
    executable.write_text("#!/bin/sh\nsleep 2\necho 'pg_dump (PostgreSQL) 16.0'\n")
    executable.chmod(0o700)
    with pytest.raises(TimeoutError, match="wall-time"):
        recovery._tool_version(executable, deadline=time.monotonic() + 0.1)


def test_failed_postgres_version_probe_reports_a_recovery_error(tmp_path: Path) -> None:
    executable = tmp_path / "pg_restore"
    executable.write_text("#!/bin/sh\necho 'local diagnostic' >&2\nexit 1\n")
    executable.chmod(0o700)
    with pytest.raises(
        RuntimeError, match=r"^pg_restore exited with status 1 during offline recovery\.$"
    ):
        recovery._tool_version(executable, deadline=time.monotonic() + 10)
