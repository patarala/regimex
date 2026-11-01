"""Actual PostgreSQL and local bytes prove reviewed orphan deletion fencing."""

import hashlib
import os
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Never
from uuid import uuid4

import pytest

pytest.importorskip("psycopg", reason="install the service extra")
pytest.importorskip("alembic", reason="install the service extra")
pytest.importorskip("procrastinate", reason="install the service extra")

import psycopg

from regimex.application.retention import RetentionLimits, RetentionService
from regimex.infrastructure.database import DatabaseRepository
from regimex.infrastructure.maintenance import (
    ServicesStoppedAttestation,
    maintenance_lock,
    quiescent_transaction,
)
from regimex.infrastructure.object_store import LocalObjectStore, ObjectInfo
from regimex.infrastructure.retention_repository import RetentionRepository
from regimex.model_registry import InferenceRuntime
from regimex.operations import Operation
from regimex.service_contracts import (
    CapacityExceeded,
    Conflict,
    Forbidden,
    ModelRecord,
    NotFound,
    Principal,
)


@dataclass
class Clock:
    now: datetime

    def __call__(self) -> datetime:
        return self.now


@dataclass
class Context:
    repository: RetentionRepository
    service: RetentionService
    store: LocalObjectStore
    principal: Principal
    root: Path
    clock: Clock


def stopped() -> ServicesStoppedAttestation:
    return ServicesStoppedAttestation(
        stopped_at=datetime.now(UTC) - timedelta(seconds=1),
        statement="Artificial test installation; all other processes and storage writers stopped.",
        api_stopped=True,
        analysis_workers_stopped=True,
        administrative_workers_stopped=True,
        maintenance_stopped=True,
        garbage_collection_stopped=True,
        exclusive_object_access=True,
    )


@pytest.fixture
def context(tmp_path: Path) -> Iterator[Context]:
    dsn = os.environ.get("REGIMEX_TEST_DATABASE_DSN")
    if not dsn:
        pytest.skip("set REGIMEX_TEST_DATABASE_DSN to a disposable PostgreSQL database")
    database = DatabaseRepository(dsn, schema="regimex_retention_" + uuid4().hex)
    try:
        database.migrate()
        repository = RetentionRepository(database)
        store = LocalObjectStore(
            tmp_path / "objects", max_object_bytes=1024**2, max_total_bytes=8 * 1024**2
        )
        clock = Clock(datetime.now(UTC) + timedelta(days=2))
        identity = Principal(owner_id="operator", role="admin", token_id=uuid4())
        yield Context(
            repository,
            RetentionService(repository, store, clock=clock),
            store,
            identity,
            tmp_path,
            clock,
        )
    finally:
        with psycopg.connect(dsn, autocommit=True) as connection:
            connection.execute(
                psycopg.sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                    psycopg.sql.Identifier(database.schema)
                )
            )


def upload(
    context: Context, key: str, data: bytes = b"payload", *, register: bool = False
) -> ObjectInfo:
    path = context.root / uuid4().hex
    path.write_bytes(data)
    info = context.store.put(key, path, expected_sha256=hashlib.sha256(data).hexdigest())
    if register:
        with context.repository.database.connection() as connection:
            context.repository.database._object(connection, info)
    return info


def test_preserves_every_ledger_object_and_collects_only_reviewed_old_orphans(
    context: Context,
) -> None:
    registered = upload(context, "reproducibility/unreferenced-ledger-entry", register=True)
    orphan = upload(context, "models/cancelled-attempt/weights")
    young = upload(context, "snapshots/still-within-grace")
    # Advance only this artificial payload's observed timestamp to represent a
    # recent upload; real version verification still uses actual local bytes.
    os.utime(context.store.root / young.key, (context.clock.now.timestamp(),) * 2)
    young = context.store.stat(young.key)
    plan = context.service.plan(context.principal, attestation=stopped())
    assert [item.object for item in plan.objects] == [orphan]
    assert plan.registered_objects == 1 and not plan.multipart
    review = context.root / "review.json"
    assert context.service.export_plan(context.principal, plan.plan_id, review) == plan
    assert review.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        context.service.export_plan(context.principal, plan.plan_id, review)
    with pytest.raises(Conflict, match="review digest"):
        context.service.apply(
            context.principal, plan.plan_id, expected_sha256="f" * 64, attestation=stopped()
        )
    assert context.store.stat(orphan.key) == orphan
    receipt = context.service.apply(
        context.principal, plan.plan_id, expected_sha256=plan.sha256, attestation=stopped()
    )
    assert receipt.complete and [event.status for event in receipt.events] == ["deleted"]
    assert context.store.stat(registered.key) == registered
    assert context.store.stat(young.key) == young
    with pytest.raises(FileNotFoundError):
        context.store.stat(orphan.key)
    replay = context.service.apply(
        context.principal, plan.plan_id, expected_sha256=plan.sha256, attestation=stopped()
    )
    assert replay.complete and replay.events == receipt.events


@pytest.mark.parametrize("isolation", ["READ COMMITTED", "REPEATABLE READ"])
def test_stale_writer_cannot_register_deleted_version_but_new_generation_can(
    context: Context, isolation: str
) -> None:
    orphan = upload(context, "snapshots/late-registration")
    database = context.repository.database
    with database.connection() as stale:
        stale.execute("SET application_name='artificial-stale-client'")
        stale.execute(psycopg.sql.SQL("SET TRANSACTION ISOLATION LEVEL " + isolation))
        stale.execute("SELECT 1")
        plan = context.service.plan(context.principal, attestation=stopped())
        context.service.apply(
            context.principal, plan.plan_id, expected_sha256=plan.sha256, attestation=stopped()
        )
        with pytest.raises(psycopg.Error) as failure:
            database._object(stale, orphan)
        assert failure.value.sqlstate in {"P0001", "40001"}
        stale.rollback()
    fresh = upload(context, orphan.key)
    assert fresh.version_id != orphan.version_id
    with database.connection() as connection:
        database._object(connection, fresh)
    assert context.store.stat(fresh.key) == fresh
    with (
        database.connection() as connection,
        pytest.raises(psycopg.errors.RaiseException, match="fenced"),
    ):
        # The trigger runs even when ON CONFLICT would find a newer registered
        # generation at this key and otherwise discard the stale INSERT.
        database._object(connection, orphan)


def test_crash_after_physical_deletion_keeps_fence_and_resumes_audit(
    context: Context, monkeypatch: pytest.MonkeyPatch
) -> None:
    orphan = upload(context, "orphan/crash")
    plan = context.service.plan(context.principal, attestation=stopped())
    actual_delete = context.store.delete_version

    def crash_after_delete(info: ObjectInfo) -> Never:
        actual_delete(info)
        raise RuntimeError("artificial process death after deletion before audit")

    monkeypatch.setattr(context.store, "delete_version", crash_after_delete)
    with pytest.raises(RuntimeError, match="artificial process death"):
        context.service.apply(
            context.principal, plan.plan_id, expected_sha256=plan.sha256, attestation=stopped()
        )
    with (
        context.repository.database.connection() as connection,
        pytest.raises(psycopg.errors.RaiseException, match="fenced"),
    ):
        context.repository.database._object(connection, orphan)
    monkeypatch.setattr(context.store, "delete_version", actual_delete)
    receipt = context.service.apply(
        context.principal, plan.plan_id, expected_sha256=plan.sha256, attestation=stopped()
    )
    assert receipt.complete and receipt.events[-1].status == "already_missing"


def test_candidate_registered_after_review_aborts_before_any_deletion(context: Context) -> None:
    first = upload(context, "orphan/first")
    second = upload(context, "orphan/second")
    plan = context.service.plan(context.principal, attestation=stopped())
    with context.repository.database.connection() as connection:
        context.repository.database._object(connection, second)
    with pytest.raises(Conflict, match="became registered"):
        context.service.apply(
            context.principal, plan.plan_id, expected_sha256=plan.sha256, attestation=stopped()
        )
    assert context.store.stat(first.key) == first and context.store.stat(second.key) == second
    with context.repository.database.connection() as connection:
        assert connection.execute(
            "SELECT count(*) FROM regimex_object_delete_fences"
        ).fetchone() == (0,)


def test_fenced_bundle_member_rolls_back_the_entire_model_registration(context: Context) -> None:
    weights = upload(context, "models/abandoned/weights")
    plan = context.service.plan(context.principal, attestation=stopped())
    context.service.apply(
        context.principal, plan.plan_id, expected_sha256=plan.sha256, attestation=stopped()
    )
    manifest = upload(context, "models/retry/manifest", b"non-executable SQL transaction fixture")
    # This fixture exercises the commit boundary only; it is never loaded as a
    # scientific bundle or treated as model-quality evidence.
    model = ModelRecord(
        model_version_id="retry-candidate",
        architecture_id="baseline",
        bundle_sha256="a" * 64,
        manifest=manifest,
        members={"weights.npz": weights},
        inference_runtime=InferenceRuntime(),
        inference_policy_sha256="b" * 64,
        created_at=datetime.now(UTC),
    )
    database = context.repository.database
    with pytest.raises(psycopg.errors.RaiseException, match="fenced"):
        database.register_model(context.principal, model)
    with database.connection() as connection:
        assert connection.execute("SELECT 1 FROM regimex_models").fetchone() is None
        assert (
            connection.execute(
                "SELECT 1 FROM regimex_objects WHERE object_key=%s", (manifest.key,)
            ).fetchone()
            is None
        )
        assert (
            connection.execute(
                "SELECT 1 FROM regimex_object_version_claims WHERE object_key=%s", (manifest.key,)
            ).fetchone()
            is None
        )
    fresh = upload(context, weights.key)
    assert fresh.version_id != weights.version_id
    retried = model.model_copy(update={"members": {"weights.npz": fresh}})
    assert database.register_model(context.principal, retried) == retried
    assert database.get_active_model() is None


def test_killed_collector_leaves_committed_fence_and_resumes_from_missing_payload(
    context: Context,
) -> None:
    orphan = upload(context, "orphan/actual-process-death")
    plan = context.service.plan(context.principal, attestation=stopped())
    script = """
import os, signal
from datetime import UTC, datetime, timedelta
from pathlib import Path
from regimex.application.retention import RetentionService
from regimex.infrastructure.database import DatabaseRepository
from regimex.infrastructure.maintenance import ServicesStoppedAttestation
from regimex.infrastructure.object_store import LocalObjectStore
from regimex.infrastructure.retention_repository import RetentionRepository
from regimex.service_contracts import Principal

class KilledCollector(LocalObjectStore):
    def delete_version(self, info):
        super().delete_version(info)
        os.kill(os.getpid(), signal.SIGKILL)

database = DatabaseRepository(os.environ['REGIMEX_TEST_DATABASE_DSN'],
    schema=os.environ['REGIMEX_GC_TEST_SCHEMA'])
store = KilledCollector(Path(os.environ['REGIMEX_GC_TEST_ROOT']),
    max_object_bytes=1024**2, max_total_bytes=8*1024**2)
principal = Principal.model_validate_json(os.environ['REGIMEX_GC_TEST_PRINCIPAL'])
attestation = ServicesStoppedAttestation.model_validate_json(os.environ['REGIMEX_GC_TEST_STOPPED'])
service = RetentionService(RetentionRepository(database), store,
    clock=lambda: datetime.now(UTC)+timedelta(days=2))
from uuid import UUID
service.apply(principal, UUID(os.environ['REGIMEX_GC_TEST_PLAN']),
    expected_sha256=os.environ['REGIMEX_GC_TEST_SHA'], attestation=attestation)
"""
    environment = dict(os.environ)
    environment.update(
        REGIMEX_GC_TEST_SCHEMA=context.repository.database.schema,
        REGIMEX_GC_TEST_ROOT=str(context.store.root),
        REGIMEX_GC_TEST_PRINCIPAL=context.principal.model_dump_json(),
        REGIMEX_GC_TEST_STOPPED=stopped().model_dump_json(),
        REGIMEX_GC_TEST_PLAN=str(plan.plan_id),
        REGIMEX_GC_TEST_SHA=plan.sha256,
    )
    process = subprocess.run(
        [sys.executable, "-c", script], env=environment, capture_output=True, text=True, timeout=30
    )
    assert process.returncode == -signal.SIGKILL, process.stderr
    with pytest.raises(FileNotFoundError):
        context.store.stat(orphan.key)
    with (
        context.repository.database.connection() as connection,
        pytest.raises(psycopg.errors.RaiseException, match="fenced"),
    ):
        context.repository.database._object(connection, orphan)
    receipt = context.service.apply(
        context.principal, plan.plan_id, expected_sha256=plan.sha256, attestation=stopped()
    )
    assert receipt.complete and receipt.events[-1].status == "already_missing"


def test_killed_local_upload_is_reconciled_only_after_review_and_grace(context: Context) -> None:
    source = context.root / "large-source"
    source.write_bytes(b"partial-upload" * 4096)
    environment = dict(os.environ)
    environment.update(
        REGIMEX_GC_TEST_ROOT=str(context.store.root),
        REGIMEX_GC_TEST_SOURCE=str(source),
    )
    script = """
import hashlib, os, signal
from pathlib import Path
from regimex.infrastructure import object_store
original = object_store._stream
def killed_copy(source, destination, *, limit):
    if destination is not None:
        os.write(destination, os.read(source, 64))
        os.kill(os.getpid(), signal.SIGKILL)
    return original(source, destination, limit=limit)
store = object_store.LocalObjectStore(Path(os.environ['REGIMEX_GC_TEST_ROOT']),
    max_object_bytes=1024**2, max_total_bytes=8*1024**2)
source = Path(os.environ['REGIMEX_GC_TEST_SOURCE'])
object_store._stream = killed_copy
store.put('uploads/unfinished', source,
    expected_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
"""
    child = subprocess.run(
        [sys.executable, "-c", script], env=environment, capture_output=True, text=True, timeout=30
    )
    assert child.returncode == -signal.SIGKILL, child.stderr
    assert not context.store.inventory()
    uploads = context.store.list_multipart_uploads()
    assert len(uploads) == 1
    assert (context.store.root / uploads[0].key).read_bytes() == source.read_bytes()[:64]
    service = RetentionService(
        context.repository, context.store, limits=RetentionLimits(minimum_grace_seconds=1)
    )
    young = service.plan(context.principal, attestation=stopped(), grace_seconds=1)
    assert not young.multipart
    time.sleep(1.1)
    plan = service.plan(context.principal, attestation=stopped(), grace_seconds=1)
    assert plan.multipart == uploads and not plan.objects
    receipt = service.apply(
        context.principal, plan.plan_id, expected_sha256=plan.sha256, attestation=stopped()
    )
    assert receipt.complete and receipt.events[-1].status == "mpu_aborted"
    assert not context.store.list_multipart_uploads()
    context.store.assert_empty_store()


def test_plan_is_bound_to_admin_owner_installation_and_store(context: Context) -> None:
    orphan = upload(context, "orphan/identity")
    plan = context.service.plan(context.principal, attestation=stopped())
    with pytest.raises(Forbidden):
        context.service.get_plan(
            context.principal.model_copy(update={"role": "user"}), plan.plan_id
        )
    with pytest.raises(NotFound):
        context.service.get_plan(
            context.principal.model_copy(update={"owner_id": "other"}), plan.plan_id
        )
    other_store = LocalObjectStore(
        context.root / "another-store", max_object_bytes=1024**2, max_total_bytes=1024**2
    )
    different_scope = RetentionService(context.repository, other_store, clock=context.clock)
    with pytest.raises(Conflict, match="store location"):
        different_scope.apply(
            context.principal, plan.plan_id, expected_sha256=plan.sha256, attestation=stopped()
        )
    with context.repository.database.connection() as connection:
        connection.execute(
            "UPDATE regimex_maintenance_identity SET installation_id=%s WHERE singleton", (uuid4(),)
        )
    with pytest.raises(Conflict, match="database installation"):
        context.service.apply(
            context.principal, plan.plan_id, expected_sha256=plan.sha256, attestation=stopped()
        )
    assert context.store.stat(orphan.key) == orphan


def test_plan_limits_fail_before_admitting_evidence_or_fences(context: Context) -> None:
    first = upload(context, "orphan/first")
    second = upload(context, "orphan/second")
    service = RetentionService(
        context.repository,
        context.store,
        clock=context.clock,
        limits=RetentionLimits(max_plan_entries=1),
    )
    with pytest.raises(CapacityExceeded):
        service.plan(context.principal, attestation=stopped())
    with context.repository.database.connection() as connection:
        assert connection.execute("SELECT count(*) FROM regimex_orphan_plans").fetchone() == (0,)
        assert connection.execute(
            "SELECT count(*) FROM regimex_object_delete_fences"
        ).fetchone() == (0,)
    assert context.store.stat(first.key) == first and context.store.stat(second.key) == second


def test_cleanup_audit_fences_and_claim_states_cannot_be_rewritten(context: Context) -> None:
    upload(context, "orphan/immutable")
    plan = context.service.plan(context.principal, attestation=stopped())
    context.service.apply(
        context.principal, plan.plan_id, expected_sha256=plan.sha256, attestation=stopped()
    )
    for table in (
        "regimex_orphan_plans",
        "regimex_object_delete_fences",
        "regimex_orphan_cleanup_events",
    ):
        with (
            context.repository.database.connection() as connection,
            pytest.raises(psycopg.errors.RaiseException, match="immutable"),
        ):
            connection.execute(
                psycopg.sql.SQL("DELETE FROM {}").format(psycopg.sql.Identifier(table))
            )
    with (
        context.repository.database.connection() as connection,
        pytest.raises(psycopg.errors.RaiseException, match="cannot be changed"),
    ):
        connection.execute("UPDATE regimex_object_version_claims SET state='registered'")


def test_shared_table_locks_block_resource_commits(context: Context) -> None:
    orphan = upload(context, "orphan/blocked-commit")
    database = context.repository.database

    def writer() -> str | None:
        with database.connection() as connection:
            connection.execute("SET LOCAL lock_timeout='100ms'")
            try:
                database._object(connection, orphan)
            except psycopg.errors.LockNotAvailable as exc:
                return exc.sqlstate
        return None

    with database.connection() as maintenance:
        maintenance.autocommit = True
        maintenance.execute("SET application_name='artificial-maintenance'")
        maintenance_lock(maintenance, database.schema)
        with (
            quiescent_transaction(maintenance, schema=database.schema),
            ThreadPoolExecutor(max_workers=1) as pool,
        ):
            assert pool.submit(writer).result(timeout=5) == "55P03"
    with database.connection() as connection:
        database._object(connection, orphan)


def test_queued_operation_and_open_service_session_prevent_cleanup(context: Context) -> None:
    now = datetime.now(UTC)
    operation = Operation(
        operation_id=uuid4(),
        owner_id=context.principal.owner_id,
        kind="generation",
        resource_id="artificial",
        request_sha256="a" * 64,
        idempotency_key="pending",
        created_at=now,
        updated_at=now,
    )
    with context.repository.database.connection() as connection:
        connection.execute(
            "INSERT INTO regimex_operations(operation_id,owner_id,kind,idempotency_key,"
            "request_sha256,state,document) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s)",
            (
                operation.operation_id,
                operation.owner_id,
                operation.kind,
                operation.idempotency_key,
                operation.request_sha256,
                operation.state,
                psycopg.types.json.Jsonb(operation.model_dump(mode="json")),
            ),
        )
    with pytest.raises(Conflict, match="zero queued"):
        context.service.plan(context.principal, attestation=stopped())
    with context.repository.database.connection() as connection:
        connection.execute(
            "DELETE FROM regimex_operations WHERE operation_id=%s", (operation.operation_id,)
        )
    with context.repository.database.connection() as active:
        active.execute("SELECT 1")
        with pytest.raises(Conflict, match="sessions remain"):
            context.service.plan(context.principal, attestation=stopped())


def test_initial_schema_registers_object_claims_and_preserves_them_on_migrate(
    context: Context,
) -> None:
    database = context.repository.database
    info = upload(context, "reproducibility/artifact", register=True)
    with database.connection() as connection:
        identity = connection.execute(
            "SELECT installation_id FROM regimex_maintenance_identity WHERE singleton"
        ).fetchone()
        assert identity is not None
    database.migrate()
    assert database.readiness()
    with database.connection() as connection:
        assert connection.execute(
            "SELECT state FROM regimex_object_version_claims WHERE object_key=%s AND version_id=%s",
            (info.key, info.version_id),
        ).fetchone() == ("registered",)
        assert connection.execute("SELECT ready FROM regimex_restore_state").fetchone() == (True,)
        assert (
            connection.execute(
                "SELECT installation_id FROM regimex_maintenance_identity WHERE singleton"
            ).fetchone()
            == identity
        )
        database._object(connection, info)
    assert context.store.stat(info.key) == info
