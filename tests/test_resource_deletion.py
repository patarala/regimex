"""Real ownership, concurrency and failure recovery for resource removal."""

import hashlib
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Event
from typing import cast
from uuid import UUID, uuid4

import pytest

pytest.importorskip("psycopg")
pytest.importorskip("alembic")
pytest.importorskip("procrastinate")

from psycopg.types.json import Jsonb
from test_object_store_contract import Backend
from test_object_store_contract import backend as backend
from test_s3_object_store import Context
from test_s3_object_store import actual_s3 as actual_s3
from test_service_database import _model, _result
from test_storage_quotas import database as database
from test_storage_quotas import export, principal, service

from regimex.application.deletion import DeletionService
from regimex.application.service import AnalysisService
from regimex.deletion_contracts import DeletionKind
from regimex.infrastructure.database import DatabaseRepository
from regimex.infrastructure.deletion_repository import DeletionRepository
from regimex.infrastructure.object_store import ObjectInfo
from regimex.infrastructure.storage_quotas import release_absent_reservations
from regimex.model_registry import ArchitectureRegistry
from regimex.operations import OperationFailure
from regimex.service_contracts import Conflict, NotFound, SnapshotRecord, StaleAttempt


def deletion(value: AnalysisService) -> DeletionService:
    return DeletionService(DeletionRepository(value.repository), value.store)


def submit(value: AnalysisService, snapshot: SnapshotRecord, *, key: str = "analysis") -> UUID:
    database = value.repository
    if database.get_active_model() is None:
        model = _model(database)
        database.activate_model(
            principal("administrator", admin=True), model.model_version_id, expected_previous=None
        )
    return database.submit_analysis(
        principal(snapshot.owner_id),
        snapshot.snapshot_id,
        idempotency_key=key,
        request_sha256=hashlib.sha256(key.encode()).hexdigest(),
    ).analysis_id


def complete(value: AnalysisService, identity: UUID, path: Path) -> ObjectInfo:
    claimed = value.repository.claim_operation(identity)
    assert claimed is not None and claimed.attempt_token is not None
    result, report = _result(value.repository, identity, {})
    path.write_bytes(b"<!doctype html><title>Deliberate report fixture</title>")
    info = value.store_owned_object(
        report.owner_id,
        report.source.source_id,
        f"reports/{identity}/{claimed.attempt_token}.html",
        path,
        expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    value.repository.complete_analysis(
        identity, claimed.attempt_token, result, report=report.model_copy(update={"object": info})
    )
    return info


def test_delete_experiment_reclaims_quota_and_rejects_identity_replay(
    database: DatabaseRepository, tmp_path: Path
) -> None:
    value = service(database, tmp_path)
    owner = principal()
    snapshot = value.create_snapshot(owner, export())
    deleter = deletion(value)
    preview = deleter.preview(owner, "experiment", snapshot.snapshot_id)
    assert (preview.experiment_count, preview.analysis_count, preview.object_count) == (1, 0, 1)
    assert preview.size_bytes == snapshot.object.size_bytes
    assert not preview.blockers
    assert database.get_snapshot(owner, snapshot.snapshot_id) == snapshot
    with pytest.raises(Conflict, match="confirmation"):
        deleter.apply(owner, preview.plan_id, expected_sha256="f" * 64)
    receipt = deleter.apply(owner, preview.plan_id, expected_sha256=preview.sha256)
    assert receipt.state == "completed"
    assert receipt.reclaimed_bytes == preview.size_bytes
    assert receipt.pending_bytes == receipt.pending_objects == 0
    assert value.storage_usage(owner).used_bytes == value.storage_usage(owner).reserved_bytes == 0
    with pytest.raises(NotFound):
        database.get_snapshot(owner, snapshot.snapshot_id)
    with pytest.raises(FileNotFoundError):
        value.store.stat(snapshot.object.key)
    with pytest.raises(Conflict, match="deleted"):
        value.create_snapshot(owner, export())
    assert value.storage_usage(owner).reserved_bytes == 0
    assert deleter.apply(owner, preview.plan_id, expected_sha256=preview.sha256) == receipt


@pytest.mark.parametrize("alteration", ["object", "count", "owner"])
def test_modified_review_cannot_authorize_removal(
    database: DatabaseRepository, tmp_path: Path, alteration: str
) -> None:
    value = service(database, tmp_path)
    owner = principal()
    snapshot = value.create_snapshot(owner, export())
    foreign = value.create_snapshot(principal("other"), export())
    deleter = deletion(value)
    preview = deleter.preview(owner, "experiment", snapshot.snapshot_id)
    with database.connection() as connection:
        if alteration == "owner":
            connection.execute(
                "UPDATE regimex_deletion_plans SET owner_id='other' WHERE plan_id=%s",
                (preview.plan_id,),
            )
            owner = principal("other")
        else:
            row = connection.execute(
                "SELECT document FROM regimex_deletion_plans WHERE plan_id=%s", (preview.plan_id,)
            ).fetchone()
            assert row is not None
            document = cast(dict[str, object], row[0])
            if alteration == "object":
                document["objects"] = [foreign.object.model_dump(mode="json")]
            else:
                cast(dict[str, object], document["preview"])["size_bytes"] = 0
            connection.execute(
                "UPDATE regimex_deletion_plans SET document=%s WHERE plan_id=%s",
                (Jsonb(document), preview.plan_id),
            )
    with pytest.raises(Conflict, match=r"integrity|identity"):
        deleter.apply(owner, preview.plan_id, expected_sha256=preview.sha256)
    assert database.get_snapshot(principal(), snapshot.snapshot_id) == snapshot
    assert database.get_snapshot(principal("other"), foreign.snapshot_id) == foreign
    assert value.store.stat(snapshot.object.key) == snapshot.object
    assert value.store.stat(foreign.object.key) == foreign.object


def test_modified_pending_object_cannot_redirect_retry(
    database: DatabaseRepository, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = service(database, tmp_path)
    snapshot = value.create_snapshot(principal(), export())
    foreign = value.create_snapshot(principal("other"), export())
    deleter = deletion(value)
    preview = deleter.preview(principal(), "experiment", snapshot.snapshot_id)
    remove = value.store.delete_version

    def fail(info: ObjectInfo) -> None:
        raise OSError("deliberate outage")

    monkeypatch.setattr(value.store, "delete_version", fail)
    assert (
        deleter.apply(principal(), preview.plan_id, expected_sha256=preview.sha256).state
        == "pending"
    )
    monkeypatch.setattr(value.store, "delete_version", remove)
    with database.connection() as connection:
        connection.execute(
            "UPDATE regimex_deletion_objects SET document=%s WHERE deletion_id=%s",
            (Jsonb(foreign.object.model_dump(mode="json")), preview.plan_id),
        )
    with pytest.raises(Conflict, match="inventory"):
        deleter.apply(principal(), preview.plan_id, expected_sha256=preview.sha256)
    assert value.store.stat(snapshot.object.key) == snapshot.object
    assert value.store.stat(foreign.object.key) == foreign.object
    assert value.storage_usage(principal()).reserved_bytes == snapshot.object.size_bytes


@pytest.mark.parametrize("stale_error", [StaleAttempt, RuntimeError])
def test_stale_worker_unwinds_after_cancelled_analysis_is_deleted(
    database: DatabaseRepository,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stale_error: type[Exception],
) -> None:
    value = service(database, tmp_path)
    owner = principal()
    snapshot = value.create_snapshot(owner, export())
    identity = submit(value, snapshot)

    def stale_read(operation_id: UUID) -> None:
        running = database.get_operation_internal(operation_id)
        assert running.attempt_token is not None
        database.cancel_operation(owner, operation_id)
        cancelled = database.fail_operation(
            operation_id,
            running.attempt_token,
            OperationFailure(
                code="infrastructure_failure", message="Deliberate cancelled attempt."
            ),
        )
        assert cancelled.state == "cancelled"
        deleter = deletion(value)
        preview = deleter.preview(owner, "analysis", operation_id)
        assert (
            deleter.apply(owner, preview.plan_id, expected_sha256=preview.sha256).state
            == "completed"
        )
        raise stale_error()

    monkeypatch.setattr(database, "get_analysis_internal", stale_read)
    result = value.run_analysis(identity)
    assert result is not None and result.state == "cancelled"
    assert value.run_analysis(identity) is None
    with pytest.raises(NotFound):
        database.get_analysis_summary(owner, identity)


def test_resource_deletion_on_actual_storage_provider(
    database: DatabaseRepository, tmp_path: Path, backend: Backend
) -> None:
    value = AnalysisService(
        database, backend.create(1024**2, 8 * 1024**2), ArchitectureRegistry(), tmp_path / "work"
    )
    snapshot = value.create_snapshot(principal(), export())
    deleter = deletion(value)
    preview = deleter.preview(principal(), "experiment", snapshot.snapshot_id)
    receipt = deleter.apply(principal(), preview.plan_id, expected_sha256=preview.sha256)
    assert receipt.state == "completed" and receipt.reclaimed_bytes == snapshot.object.size_bytes
    assert not value.store.inventory()
    assert (
        value.storage_usage(principal()).used_bytes
        == value.storage_usage(principal()).reserved_bytes
        == 0
    )


def test_s3_replacement_version_is_never_deleted_or_hidden_from_quota(
    database: DatabaseRepository, tmp_path: Path, actual_s3: Context
) -> None:
    path = tmp_path / "payload"
    path.write_bytes(b"reviewed generation")
    info = actual_s3.objects.put(
        "snapshots/one", path, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest()
    )
    snapshot = database.add_snapshot(
        SnapshotRecord(
            snapshot_id=uuid4(),
            owner_id="alice",
            source_id="simulator",
            run_id="run",
            source_snapshot_id="snapshot",
            content_sha256="a" * 64,
            object=info,
            created_at=datetime.now(UTC),
        )
    )
    deleter = DeletionService(DeletionRepository(database), actual_s3.objects)
    preview = deleter.preview(principal(), "experiment", snapshot.snapshot_id)
    # Out-of-band mutation deliberately violates the supported writer contract.
    # Removal must still preserve that unreviewed version and keep the charge.
    actual_s3.client.put_object(
        Bucket=actual_s3.bucket, Key=actual_s3.key(info.key), Body=b"replacement"
    )
    replacement = actual_s3.objects.stat(info.key)
    assert replacement.version_id != info.version_id
    receipt = deleter.apply(principal(), preview.plan_id, expected_sha256=preview.sha256)
    assert receipt.state == "pending" and receipt.pending_bytes == info.size_bytes
    assert actual_s3.objects.stat(info.key) == replacement


def test_delete_analysis_preserves_snapshot_and_removes_result_report_and_job_visibility(
    database: DatabaseRepository, tmp_path: Path
) -> None:
    value = service(database, tmp_path)
    owner = principal()
    snapshot = value.create_snapshot(owner, export())
    identity = submit(value, snapshot)
    report = complete(value, identity, tmp_path / "report.html")
    deleter = deletion(value)
    preview = deleter.preview(owner, "analysis", identity)
    assert (preview.experiment_count, preview.analysis_count) == (0, 1)
    assert preview.size_bytes == report.size_bytes
    result = deleter.apply(owner, preview.plan_id, expected_sha256=preview.sha256)
    assert result.state == "completed"
    assert value.read_snapshot(owner, snapshot.snapshot_id) == export()
    assert value.storage_usage(owner).used_bytes == snapshot.object.size_bytes
    for getter in (
        database.get_analysis_summary,
        database.get_analysis_report,
        database.get_operation,
    ):
        with pytest.raises(NotFound):
            getter(owner, identity)
    with pytest.raises(NotFound):
        submit(value, snapshot)
    new = submit(value, snapshot, key="new-analysis")
    assert new != identity
    database.cancel_operation(owner, new)
    remaining = deleter.preview(owner, "experiment", snapshot.snapshot_id)
    assert remaining.analysis_count == remaining.experiment_count == 1
    assert (
        deleter.apply(owner, remaining.plan_id, expected_sha256=remaining.sha256).state
        == "completed"
    )


@pytest.mark.parametrize("state", ["queued", "running", "cancel_requested"])
def test_active_analyses_block_both_deletion_targets(
    database: DatabaseRepository, tmp_path: Path, state: str
) -> None:
    value = service(database, tmp_path)
    owner = principal()
    snapshot = value.create_snapshot(owner, export())
    identity = submit(value, snapshot)
    if state != "queued":
        assert database.claim_operation(identity) is not None
    if state == "cancel_requested":
        database.cancel_operation(owner, identity)
    for kind, target in (("analysis", identity), ("experiment", snapshot.snapshot_id)):
        preview = deletion(value).preview(owner, cast(DeletionKind, kind), target)
        assert preview.blockers[0].code == "active_analyses"
        with pytest.raises(Conflict, match="blocked"):
            deletion(value).apply(owner, preview.plan_id, expected_sha256=preview.sha256)
    assert database.get_snapshot(owner, snapshot.snapshot_id) == snapshot


def test_changed_membership_invalidates_review_without_deleting_anything(
    database: DatabaseRepository, tmp_path: Path
) -> None:
    value = service(database, tmp_path)
    owner = principal()
    snapshot = value.create_snapshot(owner, export())
    deleter = deletion(value)
    preview = deleter.preview(owner, "experiment", snapshot.snapshot_id)
    identity = submit(value, snapshot)
    database.cancel_operation(owner, identity)
    with pytest.raises(Conflict, match="changed"):
        deleter.apply(owner, preview.plan_id, expected_sha256=preview.sha256)
    assert database.get_snapshot(owner, snapshot.snapshot_id) == snapshot
    assert database.get_analysis_summary(owner, identity).operation.state == "cancelled"


def test_catalog_binding_and_shared_objects_are_retained(
    database: DatabaseRepository, tmp_path: Path
) -> None:
    value = service(database, tmp_path)
    snapshot = value.create_snapshot(principal(), export())
    with database.connection() as connection:
        connection.execute(
            "INSERT INTO regimex_catalog_bindings "
            "(snapshot_id,source_namespace,group_namespace,group_id,document) "
            "VALUES (%s,'fixture','fixture','fixture',%s)",
            (snapshot.snapshot_id, Jsonb({})),
        )
    preview = deletion(value).preview(principal(), "experiment", snapshot.snapshot_id)
    assert preview.blockers[0].code == "catalog_reference"
    with pytest.raises(Conflict, match="blocked"):
        deletion(value).apply(principal(), preview.plan_id, expected_sha256=preview.sha256)
    with database.connection() as connection:
        connection.execute(
            "DELETE FROM regimex_catalog_bindings WHERE snapshot_id=%s", (snapshot.snapshot_id,)
        )
    shared = database.add_snapshot(
        snapshot.model_copy(
            update={"snapshot_id": uuid4(), "source_snapshot_id": "another-identity"}
        )
    )
    preview = deletion(value).preview(principal(), "experiment", snapshot.snapshot_id)
    assert preview.blockers[0].code == "shared_objects"
    with pytest.raises(Conflict, match="blocked"):
        deletion(value).apply(principal(), preview.plan_id, expected_sha256=preview.sha256)
    assert database.get_snapshot(principal(), shared.snapshot_id) == shared


def test_failed_physical_removal_is_discoverable_charged_and_retryable(
    database: DatabaseRepository, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = service(database, tmp_path)
    owner = principal()
    snapshot = value.create_snapshot(owner, export())
    deleter = deletion(value)
    preview = deleter.preview(owner, "experiment", snapshot.snapshot_id)
    remove = value.store.delete_version

    def fail(info: ObjectInfo) -> None:
        raise OSError("private provider diagnostic")

    monkeypatch.setattr(value.store, "delete_version", fail)
    receipt = deleter.apply(owner, preview.plan_id, expected_sha256=preview.sha256)
    assert receipt.state == "pending"
    assert receipt.pending_bytes == snapshot.object.size_bytes
    assert "private provider" not in receipt.model_dump_json()
    assert value.storage_usage(owner).reserved_bytes == snapshot.object.size_bytes
    assert deleter.list_pending(owner) == (receipt,)
    with pytest.raises(NotFound):
        deleter.receipt(principal("another"), preview.plan_id)
    with database.connection() as connection:
        assert (
            release_absent_reservations(
                connection, frozenset(), created_before=datetime.now(UTC) + timedelta(days=1)
            )
            == 0
        )
    monkeypatch.setattr(value.store, "delete_version", remove)
    restored_controller = deletion(value)
    completed = restored_controller.apply(
        owner, receipt.deletion_id, expected_sha256=receipt.sha256
    )
    assert completed.state == "completed"
    assert completed.reclaimed_bytes == snapshot.object.size_bytes
    assert not restored_controller.list_pending(owner)
    assert value.storage_usage(owner).reserved_bytes == 0


def test_retry_after_crash_between_unlink_and_quota_commit(
    database: DatabaseRepository, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = service(database, tmp_path)
    snapshot = value.create_snapshot(principal(), export())
    deleter = deletion(value)
    preview = deleter.preview(principal(), "experiment", snapshot.snapshot_id)
    original = deleter.repository.record_object

    def crash(*args: object, **kwargs: object) -> None:
        raise RuntimeError("simulated process interruption")

    monkeypatch.setattr(deleter.repository, "record_object", crash)
    with pytest.raises(RuntimeError, match="interruption"):
        deleter.apply(principal(), preview.plan_id, expected_sha256=preview.sha256)
    assert value.storage_usage(principal()).reserved_bytes == snapshot.object.size_bytes
    with pytest.raises(FileNotFoundError):
        value.store.stat(snapshot.object.key)
    monkeypatch.setattr(deleter.repository, "record_object", original)
    assert (
        deleter.apply(principal(), preview.plan_id, expected_sha256=preview.sha256).state
        == "completed"
    )


def test_inflight_upload_cannot_recreate_deleted_bytes(
    database: DatabaseRepository, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = service(database, tmp_path)
    owner = principal()
    snapshot = value.create_snapshot(owner, export())
    deleter = deletion(value)
    preview = deleter.preview(owner, "experiment", snapshot.snapshot_id)
    started, release = Event(), Event()
    put = value.store.put

    def paused_put(key: str, path: Path, *, expected_sha256: str) -> ObjectInfo:
        started.set()
        assert release.wait(5)
        return put(key, path, expected_sha256=expected_sha256)

    monkeypatch.setattr(value.store, "put", paused_put)
    with ThreadPoolExecutor(max_workers=2) as executor:
        upload = executor.submit(value.create_snapshot, owner, export())
        assert started.wait(5)
        removal = executor.submit(
            deleter.apply, owner, preview.plan_id, expected_sha256=preview.sha256
        )
        release.set()
        # Metadata registration may lose to the confirmed deletion.
        with suppress(Conflict):
            upload.result(timeout=10)
        assert removal.result(timeout=10).state == "completed"
    with pytest.raises(FileNotFoundError):
        value.store.stat(snapshot.object.key)
    assert value.storage_usage(owner).reserved_bytes == 0
