"""Completed removals survive restoration; unfinished physical removal blocks backup."""

from pathlib import Path

import pytest
from pydantic import SecretStr
from test_recovery import DatabaseFactory, analysis_service, local, stopped
from test_recovery import databases as databases
from test_recovery import tools as tools
from test_storage_quotas import export

from regimex.application import recovery
from regimex.application.deletion import DeletionService
from regimex.exports import ExportPayloadV2, seal_export
from regimex.infrastructure.deletion_repository import DeletionRepository
from regimex.infrastructure.object_store import ObjectInfo
from regimex.recovery_contracts import PostgresTools
from regimex.service_contracts import Conflict, NotFound


def test_completed_deletion_survives_backup_and_restore(
    databases: DatabaseFactory, tools: PostgresTools, tmp_path: Path
) -> None:
    database = databases("removed_resources")
    database.migrate()
    store = local(tmp_path / "source-objects")
    source = analysis_service(database, store, tmp_path / "source-work")
    token = database.create_token("owner", "user")
    owner = database.authenticate(token)
    original = export()
    retained_export = seal_export(
        ExportPayloadV2.model_validate_json(
            original.model_dump_json(exclude={"content_sha256"})
        ).model_copy(
            update={"source": original.source.model_copy(update={"snapshot_id": "retained"})}
        )
    )
    removed = source.create_snapshot(owner, original)
    retained = source.create_snapshot(owner, retained_export)
    deletion = DeletionService(DeletionRepository(database), store)
    preview = deletion.preview(owner, "experiment", removed.snapshot_id)
    receipt = deletion.apply(owner, preview.plan_id, expected_sha256=preview.sha256)
    assert receipt.state == "completed"
    unapplied = deletion.preview(owner, "experiment", retained.snapshot_id)
    usage = source.storage_usage(owner)
    assert usage.used_bytes == retained.object.size_bytes
    assert usage.reserved_bytes == 0
    backup_path = tmp_path / "backup"
    backup = recovery.backup_service(
        SecretStr(database._dsn),
        database.schema,
        store,
        backup_path,
        tools=tools,
        attestation=stopped(),
    )
    assert backup.manifest.objects == (retained.object,)
    assert backup.manifest.tables["regimex_deleted_snapshots"].rows == 1
    assert backup.manifest.tables["regimex_deletion_plans"].rows == 2

    target = databases(database.schema)
    target_store = local(tmp_path / "restored-objects")
    restore = recovery.restore_service(
        SecretStr(target._dsn),
        target_store,
        backup_path,
        expected_manifest_sha256=backup.manifest_sha256,
        assert_empty_store=target_store.assert_empty_store,
        tools=tools,
        attestation=stopped(),
    )
    assert target.readiness()
    assert restore.table_evidence == backup.manifest.tables
    assert restore.changed_versions == restore.object_count == 1
    restored_owner = target.authenticate(token)
    restored = analysis_service(target, target_store, tmp_path / "restored-work")
    restored_deletion = DeletionService(DeletionRepository(target), target_store)
    assert restored_deletion.receipt(restored_owner, preview.plan_id) == receipt
    assert (
        restored_deletion.apply(restored_owner, preview.plan_id, expected_sha256=preview.sha256)
        == receipt
    )
    assert restored.storage_usage(restored_owner) == usage
    assert restored.read_snapshot(restored_owner, retained.snapshot_id) == retained_export
    with pytest.raises(NotFound):
        target.get_snapshot(restored_owner, removed.snapshot_id)
    with pytest.raises(FileNotFoundError):
        target_store.stat(removed.object.key)
    with pytest.raises(Conflict, match="deleted"):
        restored.create_snapshot(restored_owner, original)
    with pytest.raises(Conflict, match="Storage changed"):
        restored_deletion.apply(restored_owner, unapplied.plan_id, expected_sha256=unapplied.sha256)
    assert restored.storage_usage(restored_owner) == usage
    assert restored.read_snapshot(restored_owner, retained.snapshot_id) == retained_export
    assert source.read_snapshot(owner, retained.snapshot_id) == retained_export


def test_pending_physical_deletion_prevents_a_usable_backup(
    databases: DatabaseFactory,
    tools: PostgresTools,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = databases("public")
    database.migrate()
    store = local(tmp_path / "objects")
    source = analysis_service(database, store, tmp_path / "work")
    owner = database.authenticate(database.create_token("owner", "user"))
    snapshot = source.create_snapshot(owner, export())
    deletion = DeletionService(DeletionRepository(database), store)
    preview = deletion.preview(owner, "experiment", snapshot.snapshot_id)

    def fail(info: ObjectInfo) -> None:
        raise OSError("deliberate storage outage")

    with monkeypatch.context() as patch:
        patch.setattr(store, "delete_version", fail)
        pending = deletion.apply(owner, preview.plan_id, expected_sha256=preview.sha256)
    assert pending.state == "pending"
    assert source.storage_usage(owner).reserved_bytes == snapshot.object.size_bytes
    failed_path = tmp_path / "incomplete-backup"
    with pytest.raises(Conflict, match="Complete pending resource deletions"):
        recovery.backup_service(
            SecretStr(database._dsn),
            database.schema,
            store,
            failed_path,
            tools=tools,
            attestation=stopped(),
        )
    assert not (failed_path / "manifest.json").exists()
    assert deletion.receipt(owner, preview.plan_id) == pending
    assert store.stat(snapshot.object.key) == snapshot.object
    assert source.storage_usage(owner).reserved_bytes == snapshot.object.size_bytes
    assert database.readiness()

    completed = deletion.apply(owner, preview.plan_id, expected_sha256=preview.sha256)
    assert completed.state == "completed"
    backup = recovery.backup_service(
        SecretStr(database._dsn),
        database.schema,
        store,
        tmp_path / "complete-backup",
        tools=tools,
        attestation=stopped(),
    )
    assert backup.manifest.objects == ()
    assert source.storage_usage(owner).used_bytes == source.storage_usage(owner).reserved_bytes == 0
