"""Real PostgreSQL admission and local objects under concurrent quota pressure."""

from __future__ import annotations

import hashlib
import os
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest

from regimex.exports import ExportV2
from regimex.infrastructure.object_store import LocalObjectStore, ObjectInfo
from regimex.service_contracts import Conflict, Forbidden, NotFound, Principal, SnapshotRecord
from regimex.storage import json_bytes
from regimex.storage_contracts import StorageQuotaExceeded, StorageUsage

if TYPE_CHECKING:
    from regimex.application.service import AnalysisService
    from regimex.infrastructure.database import DatabaseRepository
    from regimex.infrastructure.storage_quotas import StorageQuotaRepository


@pytest.fixture
def database() -> Iterator[DatabaseRepository]:
    dsn = os.environ.get("REGIMEX_TEST_DATABASE_DSN")
    if not dsn:
        pytest.skip("set REGIMEX_TEST_DATABASE_DSN to a disposable PostgreSQL database")
    psycopg = pytest.importorskip("psycopg")
    pytest.importorskip("alembic")
    pytest.importorskip("procrastinate")
    from regimex.infrastructure.database import DatabaseRepository

    value = DatabaseRepository(dsn, schema="regimex_storage_quota_" + uuid4().hex)
    try:
        value.migrate()
        yield value
    finally:
        with psycopg.connect(dsn, autocommit=True) as connection:
            connection.execute(
                psycopg.sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                    psycopg.sql.Identifier(value.schema)
                )
            )


def principal(owner: str = "alice", *, admin: bool = False) -> Principal:
    return Principal(owner_id=owner, role="admin" if admin else "user", token_id=uuid4())


def object_info(key: str, size: int = 10) -> ObjectInfo:
    return ObjectInfo(key=key, sha256="a" * 64, size_bytes=size, version_id="fixture")


def quotas(
    database: DatabaseRepository, owner: int = 10, integration: int = 20
) -> StorageQuotaRepository:
    from regimex.infrastructure.storage_quotas import StorageQuotaRepository

    return StorageQuotaRepository(
        database, max_owner_storage_bytes=owner, max_integration_storage_bytes=integration
    )


def service(database: DatabaseRepository, root: Path, *, maximum: int = 1024**2) -> AnalysisService:
    from regimex.application.service import AnalysisService
    from regimex.model_registry import ArchitectureRegistry

    return AnalysisService(
        database,
        LocalObjectStore(root / "objects", max_object_bytes=1024**2, max_total_bytes=8 * 1024**2),
        ArchitectureRegistry(),
        root / "work",
        max_owner_storage_bytes=maximum,
        max_integration_storage_bytes=maximum * 2,
    )


def export() -> ExportV2:
    return ExportV2.model_validate_json(
        (Path(__file__).parent / "fixtures" / "experiment-v2.json").read_bytes()
    )


def register(database: DatabaseRepository, owner: str, info: ObjectInfo) -> SnapshotRecord:
    return database.add_snapshot(
        SnapshotRecord(
            snapshot_id=uuid4(),
            owner_id=owner,
            source_id="simulator",
            run_id=uuid4().hex,
            source_snapshot_id=uuid4().hex,
            content_sha256="b" * 64,
            object=info,
            created_at=datetime.now(UTC),
        )
    )


def test_storage_usage_reports_existing_over_limit_data_without_negative_capacity() -> None:
    value = StorageUsage(used_bytes=20, reserved_bytes=3, limit_bytes=10, available_bytes=0)
    assert value.used_bytes == 20
    with pytest.raises(ValueError, match="Available storage"):
        StorageUsage(used_bytes=2, reserved_bytes=3, limit_bytes=10, available_bytes=8)


def test_concurrent_reservations_cannot_exceed_owner_limit(database: DatabaseRepository) -> None:
    repository = quotas(database)

    def reserve(index: int) -> bool:
        try:
            repository.reserve("alice", "simulator", object_info(f"snapshots/{index}", 6))
        except StorageQuotaExceeded:
            return False
        return True

    with ThreadPoolExecutor(max_workers=6) as executor:
        accepted = list(executor.map(reserve, range(6)))
    assert sum(accepted) == 1
    assert repository.owner_usage(principal()).reserved_bytes == 6
    repository.reserve("bob", "simulator", object_info("snapshots/bob", 10))
    assert repository.owner_usage(principal("bob")).reserved_bytes == 10


def test_same_object_replays_at_capacity_and_registration_does_not_double_charge(
    database: DatabaseRepository,
) -> None:
    repository = quotas(database)
    info = object_info("snapshots/alice")
    repository.reserve("alice", "simulator", info)
    repository.reserve("alice", "simulator", info)
    assert repository.owner_usage(principal()).model_dump() == {
        "used_bytes": 0,
        "reserved_bytes": 10,
        "limit_bytes": 10,
        "available_bytes": 0,
    }
    register(database, "alice", info)
    usage = repository.owner_usage(principal())
    assert (usage.used_bytes, usage.reserved_bytes, usage.available_bytes) == (10, 0, 0)
    repository.reserve("alice", "simulator", info)
    with pytest.raises(StorageQuotaExceeded, match="account"):
        repository.reserve("alice", "simulator", object_info("snapshots/new", 1))
    with pytest.raises(Conflict, match="different storage reservation"):
        repository.reserve("bob", "simulator", info)
    with pytest.raises(Conflict, match="different storage reservation"):
        repository.reserve("alice", "simulator", info.model_copy(update={"sha256": "b" * 64}))


def test_reduced_limit_preserves_old_objects_and_allows_exact_replays(
    database: DatabaseRepository,
) -> None:
    large = quotas(database, owner=30, integration=40)
    info = object_info("snapshots/alice", 20)
    large.reserve("alice", "simulator", info)
    register(database, "alice", info)
    small = quotas(database)
    assert small.owner_usage(principal()).available_bytes == 0
    small.reserve("alice", "simulator", info)
    with pytest.raises(StorageQuotaExceeded):
        small.reserve("alice", "simulator", object_info("snapshots/new", 1))


def test_connection_budget_is_shared_between_subjects_and_rotated_keys(
    database: DatabaseRepository,
) -> None:
    from regimex.infrastructure.auth_repository import AuthRepository

    token = database.create_token("backend", "integration", source_namespace="simulator")
    auth = AuthRepository(database)
    alice = auth.bind_subject(token, "alice", "run-1")
    bob = auth.bind_subject(token, "bob", "run-1")
    repository = quotas(database, owner=10, integration=15)
    repository.reserve(alice.owner_id, "simulator", object_info("alice", 10), principal=alice)
    repository.reserve(bob.owner_id, "simulator", object_info("bob", 5), principal=bob)
    replacement = database.create_token("backend", "integration", source_namespace="simulator")
    bob_rotated = auth.bind_subject(replacement, "bob", "run-1")
    with pytest.raises(StorageQuotaExceeded, match="connection"):
        repository.reserve(
            bob_rotated.owner_id, "simulator", object_info("bob-more", 1), principal=bob_rotated
        )
    assert alice.integration_id is not None
    usage = repository.integration_usage(principal("operator", admin=True), alice.integration_id)
    assert usage.reserved_bytes == 15
    with pytest.raises(Forbidden):
        repository.integration_usage(alice, alice.integration_id)
    with pytest.raises(NotFound):
        repository.integration_usage(principal("operator", admin=True), uuid4())
    other_token = database.create_token("other", "integration", source_namespace="simulator")
    other = auth.bind_subject(other_token, "bob", "run-1")
    repository.reserve(other.owner_id, "simulator", object_info("other", 10), principal=other)


@pytest.mark.parametrize("after_write", [False, True])
def test_failed_object_write_keeps_its_reservation_and_retry_reuses_it(
    database: DatabaseRepository,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    after_write: bool,
) -> None:
    value = export()
    size = len(json_bytes(value.model_dump(mode="json")))
    application = service(database, tmp_path, maximum=size)
    original = application.store.put

    def fail(key: str, source: Path, *, expected_sha256: str) -> ObjectInfo:
        if after_write:
            original(key, source, expected_sha256=expected_sha256)
        raise OSError("simulated ambiguous upload failure")

    with monkeypatch.context() as patch:
        patch.setattr(application.store, "put", fail)
        with pytest.raises(OSError, match="ambiguous"):
            application.create_snapshot(principal(), value)
    usage = application.storage_usage(principal())
    assert (usage.used_bytes, usage.reserved_bytes) == (0, size)
    with pytest.raises(StorageQuotaExceeded):
        application.storage_quotas.reserve("alice", "simulator", object_info("another", 1))
    saved = application.create_snapshot(principal(), value)
    assert application.create_snapshot(principal(), value) == saved
    usage = application.storage_usage(principal())
    assert (usage.used_bytes, usage.reserved_bytes, usage.available_bytes) == (size, 0, 0)


def test_database_commit_failure_keeps_uploaded_bytes_charged(
    database: DatabaseRepository, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    application = service(database, tmp_path)

    def fail(record: SnapshotRecord) -> SnapshotRecord:
        raise OSError("simulated database disconnect")

    with monkeypatch.context() as patch:
        patch.setattr(database, "add_snapshot", fail)
        with pytest.raises(OSError, match="disconnect"):
            application.create_snapshot(principal(), export())
    reserved = application.storage_usage(principal()).reserved_bytes
    assert reserved > 0
    application.create_snapshot(principal(), export())
    assert application.storage_usage(principal()).used_bytes == reserved
    assert application.storage_usage(principal()).reserved_bytes == 0


@pytest.mark.parametrize("incomplete_upload", [False, True])
def test_offline_cleanup_releases_only_absent_old_unregistered_reservations(
    database: DatabaseRepository, tmp_path: Path, incomplete_upload: bool
) -> None:
    from regimex.application.retention import RetentionLimits, RetentionService
    from regimex.infrastructure.maintenance import ServicesStoppedAttestation
    from regimex.infrastructure.retention_repository import RetentionRepository

    application = service(database, tmp_path)
    repository = application.storage_quotas
    repository.reserve("alice", "simulator", object_info("failed-before-write", 11))
    repository.reserve("alice", "simulator", object_info("recent-failure", 13))
    missing = object_info("registered-but-missing", 17)
    register(database, "alice", missing)
    with database.connection() as connection:
        connection.execute(
            "UPDATE regimex_storage_reservations "
            "SET created_at=clock_timestamp()-interval '2 days' "
            "WHERE object_key!='recent-failure'"
        )
    # The orphan is physically present and has not passed its retention grace.
    path = tmp_path / "payload"
    path.write_bytes(b"orphan")
    application.store_owned_object(
        "alice",
        "simulator",
        "recent-orphan",
        path,
        expected_sha256=hashlib.sha256(b"orphan").hexdigest(),
    )
    if incomplete_upload:
        # Local staging names deliberately cannot reveal their destination key.
        (tmp_path / "objects" / (".regimex-temporary-" + uuid4().hex)).write_bytes(b"partial")
    attestation = ServicesStoppedAttestation(
        stopped_at=datetime.now(UTC) - timedelta(seconds=1),
        statement="Disposable test installation; all other processes and writers are stopped.",
        api_stopped=True,
        analysis_workers_stopped=True,
        administrative_workers_stopped=True,
        maintenance_stopped=True,
        garbage_collection_stopped=True,
        exclusive_object_access=True,
    )
    retention = RetentionService(RetentionRepository(database), application.store)
    admin = principal("operator", admin=True)
    plan = retention.plan(admin, attestation=attestation)
    retention.apply(admin, plan.plan_id, expected_sha256=plan.sha256, attestation=attestation)
    usage = repository.owner_usage(principal())
    assert (usage.used_bytes, usage.reserved_bytes) == (
        17,
        13 + len(b"orphan") + (11 if incomplete_upload else 0),
    )
    # The storage adapter independently requires a real past cutoff before
    # deleting local staging. Use its shortest supported grace in this fixture.
    time.sleep(1.1)
    later = RetentionService(
        RetentionRepository(database),
        application.store,
        limits=RetentionLimits(minimum_grace_seconds=1),
    )
    plan = later.plan(admin, attestation=attestation, grace_seconds=1)
    later.apply(admin, plan.plan_id, expected_sha256=plan.sha256, attestation=attestation)
    usage = repository.owner_usage(principal())
    assert (usage.used_bytes, usage.reserved_bytes) == (17, 0)


def test_previous_schema_upgrade_backfills_existing_payloads(
    database: DatabaseRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import psycopg
    from alembic import command
    from alembic.config import Config
    from psycopg.types.json import Jsonb

    # Build the actual old schema. Replacing only its recorded revision leaves
    # later tables/triggers behind and does not exercise a real forward upgrade.
    with psycopg.connect(database._dsn, autocommit=True) as connection:
        connection.execute(
            psycopg.sql.SQL("DROP SCHEMA {} CASCADE").format(
                psycopg.sql.Identifier(database.schema)
            )
        )
    original_upgrade = command.upgrade

    def old_schema(config: Config, revision: str) -> None:
        original_upgrade(config, "0005_integration_limits" if revision == "head" else revision)

    with monkeypatch.context() as context:
        context.setattr(command, "upgrade", old_schema)
        database.migrate()
    info = object_info("existing/snapshot", 20)
    saved = SnapshotRecord(
        snapshot_id=uuid4(),
        owner_id="alice",
        source_id="simulator",
        run_id="old-run",
        source_snapshot_id="old-snapshot",
        content_sha256="b" * 64,
        object=info,
        created_at=datetime.now(UTC),
    )
    with database.connection() as connection:
        connection.execute(
            "INSERT INTO regimex_objects(object_key,sha256,size_bytes,version_id,document) "
            "VALUES (%s,%s,%s,%s,%s)",
            (
                info.key,
                info.sha256,
                info.size_bytes,
                info.version_id,
                Jsonb(info.model_dump(mode="json")),
            ),
        )
        connection.execute(
            "INSERT INTO regimex_snapshots "
            "(snapshot_id,owner_id,source_id,run_id,source_snapshot_id,"
            "content_sha256,object_key,document) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
            (
                saved.snapshot_id,
                saved.owner_id,
                saved.source_id,
                saved.run_id,
                saved.source_snapshot_id,
                saved.content_sha256,
                info.key,
                Jsonb(saved.model_dump(mode="json")),
            ),
        )
    database.migrate()
    assert database.get_snapshot(principal(), saved.snapshot_id) == saved
    usage = quotas(database).owner_usage(principal())
    assert (usage.used_bytes, usage.reserved_bytes, usage.available_bytes) == (20, 0, 0)
