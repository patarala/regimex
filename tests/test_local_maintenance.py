"""Backup interruption, service-state restoration and isolated verification safety."""

import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from regimex.local_installation import install, maintenance, systemd, verification
from regimex.local_installation.maintenance import WRITERS, Maintenance, MaintenanceError


@pytest.fixture
def managed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Maintenance:
    root = tmp_path / "service"
    root.mkdir(mode=0o700)
    (root / "credentials").mkdir(mode=0o700)
    install.write_private(root / "credentials/database.dsn", "postgresql://test@localhost/test")
    install.write_private(
        root / "local-installation.json",
        json.dumps(
            {
                "schema_version": 1,
                "phase": "ready",
                "name": "regimex-test",
                "port": 18000,
                "postgres_bin": "/opt/postgres/bin",
                "active_release": str(root / "releases/current"),
                "source_commit": "a" * 40,
            }
        ),
    )
    values = install.environment_file(root, 18000, existing=False)
    install.write_private(
        root / "service.env", "".join(f"{key}={value}\n" for key, value in values.items())
    )
    monkeypatch.setattr(systemd, "check_existing", lambda *args, **kwargs: None)
    return Maintenance(root)


def fake_units(
    value: Maintenance, monkeypatch: pytest.MonkeyPatch, *, api: bool = True
) -> tuple[dict[str, bool], list[tuple[str, ...]]]:
    states = {
        "postgres": True,
        "migrate": True,
        "api": api,
        "analysis": True,
        "admin": False,
        "maintenance": True,
    }
    events: list[tuple[str, ...]] = []

    def control(action: str, *parts: str) -> None:
        assert value.journal.exists()
        events.append((action, *parts))
        for part in parts:
            states[part] = action == "start"

    monkeypatch.setattr(value, "active", lambda part: states[part])
    monkeypatch.setattr(value, "control", control)
    monkeypatch.setattr(value, "idle", lambda: None)
    monkeypatch.setattr(value, "wait_ready", lambda: None)
    return states, events


def patch_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("psycopg")
    from regimex.recovery_contracts import PostgresTools

    monkeypatch.setattr(
        PostgresTools,
        "discover",
        lambda **kwargs: PostgresTools(Path("/pg_dump"), Path("/pg_restore")),
    )


def test_backup_failure_restores_exact_initial_units(
    managed: Maintenance, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch_engine(monkeypatch)
    from regimex.application import recovery
    from regimex.service_contracts import Conflict

    states, events = fake_units(managed, monkeypatch, api=False)
    original = states.copy()

    def reject(*args: object, **kwargs: object) -> None:
        assert not any(states[x] for x in WRITERS)
        raise Conflict("Resource deletion cleanup is pending.")

    monkeypatch.setattr(recovery, "backup_service", reject)
    with pytest.raises(Conflict, match="cleanup"):
        managed.backup(managed.root.parent / "backup")
    assert states == original
    assert not any(event[0] == "start" and ("api" in event or "admin" in event) for event in events)
    assert not managed.journal.exists()


def test_pending_work_preserves_workers_and_resumes_intake(
    managed: Maintenance, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch_engine(monkeypatch)
    states, events = fake_units(managed, monkeypatch)
    original = states.copy()

    def busy() -> None:
        raise MaintenanceError("Backup refused: queued work remains.")

    monkeypatch.setattr(managed, "idle", busy)
    with pytest.raises(MaintenanceError, match="queued work"):
        managed.backup(managed.root.parent / "backup")
    assert states == original
    assert [event for event in events if event[0] == "stop"] == [("stop", "api")]
    assert not managed.journal.exists()


def test_success_waits_for_content_verification_and_resumes(
    managed: Maintenance, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch_engine(monkeypatch)
    from regimex.application import recovery

    states, _ = fake_units(managed, monkeypatch)
    original = states.copy()
    verified = []
    result = SimpleNamespace(
        manifest_sha256="b" * 64, manifest=SimpleNamespace(backup_id=uuid4(), objects=(1, 2))
    )
    monkeypatch.setattr(recovery, "backup_service", lambda *args, **kwargs: result)

    def verify(path: Path, digest: str) -> None:
        assert not any(states[x] for x in WRITERS)
        verified.append((path, digest))

    monkeypatch.setattr(verification, "verify_files", verify)
    destination = managed.root.parent / "backup"
    actual = managed.backup(destination)
    assert verified == [(destination, "b" * 64)]
    assert actual["object_count"] == 2
    assert states == original
    assert not managed.journal.exists()


def test_resume_recovers_journal_after_process_death(
    managed: Maintenance, monkeypatch: pytest.MonkeyPatch
) -> None:
    states, _ = fake_units(managed, monkeypatch)
    original = {part: states[part] for part in WRITERS}
    managed.save_journal({"version": 1, "name": managed.name, "kind": "backup", "active": original})
    for part in WRITERS:
        states[part] = False
    with pytest.raises(MaintenanceError, match="resume-maintenance"):
        maintenance.pending_maintenance(managed.root)
    managed.resume()
    assert {part: states[part] for part in WRITERS} == original
    assert not managed.journal.exists()


def test_failed_resume_retains_journal_for_retry(
    managed: Maintenance, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = dict.fromkeys(WRITERS, True)
    managed.save_journal({"version": 1, "name": managed.name, "kind": "backup", "active": original})

    def fail(*args: str) -> None:
        raise RuntimeError("start failed")

    monkeypatch.setattr(managed, "control", fail)
    with pytest.raises(RuntimeError, match="start failed"):
        managed.resume()
    assert managed.journal.exists()


@pytest.mark.parametrize(
    "part", ["objects", "work", "releases", "credentials", "postgres", "verification"]
)
def test_backup_cannot_overlap_live_directories(managed: Maintenance, part: str) -> None:
    (managed.root / part).mkdir(exist_ok=True, mode=0o700)
    with pytest.raises(MaintenanceError, match="backups/"):
        managed.destination(managed.root / part / "copy")


def test_backup_rejects_symlink_into_live_store(managed: Maintenance) -> None:
    (managed.root / "objects").mkdir(mode=0o700)
    link = managed.root.parent / "alias"
    link.symlink_to(managed.root / "objects")
    with pytest.raises(MaintenanceError, match="backups/"):
        managed.destination(link / "copy")


def test_maintenance_and_installer_share_exclusive_lock(managed: Maintenance) -> None:
    with (
        install.installation_lock(managed.root),
        pytest.raises(ValueError, match="maintenance command"),
        install.installation_lock(managed.root),
    ):
        pytest.fail("a second maintainer acquired the lock")


def test_configured_environment_isolated_and_restored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REGIMEX_DATABASE_DSN", "unrelated")
    monkeypatch.setenv("PGHOST", "unrelated")
    monkeypatch.setenv("PYTHONPATH", "unrelated")
    before = dict(os.environ)
    with maintenance.configured_environment({"REGIMEX_OBJECT_ROOT": "/owned"}):
        assert "REGIMEX_DATABASE_DSN" not in os.environ
        assert "PGHOST" not in os.environ
        assert "PYTHONPATH" not in os.environ
        assert os.environ["REGIMEX_OBJECT_ROOT"] == "/owned"
        assert os.environ.get("PATH") == before.get("PATH")
    assert dict(os.environ) == before


def test_verification_cleanup_refuses_unowned_path(managed: Maintenance) -> None:
    unrelated = managed.root.parent / "keep"
    unrelated.mkdir()
    with pytest.raises(MaintenanceError, match="does not belong"):
        verification.cleanup_workspace(managed, unrelated)
    assert unrelated.exists()


def test_verification_cleanup_keeps_running_cluster(
    managed: Maintenance, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = managed.root / "verification"
    parent.mkdir(mode=0o700)
    workspace = parent / ("restore-" + "a" * 32)
    workspace.mkdir(mode=0o700)
    (workspace / "postgres").mkdir()
    (workspace / "postgres/postmaster.pid").write_text("deliberate pid fixture")
    install.write_private(
        workspace / "owner.json",
        json.dumps({"installation": str(managed.root), "workspace": workspace.name}),
    )
    monkeypatch.setattr(managed, "run", lambda *args, **kwargs: "")
    with pytest.raises(MaintenanceError, match="remains active"):
        verification.cleanup_workspace(managed, workspace)
    assert workspace.exists()


def backup_files(tmp_path: Path) -> tuple[Path, str]:
    pytest.importorskip("psycopg")
    from regimex.infrastructure.maintenance import ServicesStoppedAttestation
    from regimex.infrastructure.object_store import ObjectInfo
    from regimex.recovery_contracts import (
        RECOVERY_REVISION,
        BackupManifest,
        MaintenanceIdentity,
        TableEvidence,
    )

    path = tmp_path / "files"
    path.mkdir(mode=0o700)
    (path / "objects").mkdir(mode=0o700)
    dump, payload = b"deliberate dump fixture", b"deliberate immutable payload"
    (path / "database.dump").write_bytes(dump)
    checksum = hashlib.sha256(payload).hexdigest()
    (path / "objects" / checksum).write_bytes(payload)
    attestation = ServicesStoppedAttestation.model_validate(maintenance.stopped_attestation())
    manifest = BackupManifest(
        backup_id=uuid4(),
        schema_name="public",
        migration_revision=RECOVERY_REVISION,
        created_at=datetime.now(UTC),
        attestation=attestation,
        postgres_version="16.2",
        dump_tool_version="pg_dump 16.2",
        dump_size_bytes=len(dump),
        dump_sha256=hashlib.sha256(dump).hexdigest(),
        objects=(
            ObjectInfo(
                key="payload", size_bytes=len(payload), sha256=checksum, version_id="fixture-1"
            ),
        ),
        tables={"fixture": TableEvidence(rows=1, portable_sha256="a" * 64)},
        source_maintenance_identity=MaintenanceIdentity(
            singleton=True,
            installation_id=uuid4(),
            created_at=datetime.now(UTC),
            restored_from_installation_id=None,
            restored_from_backup_sha256=None,
        ),
    )
    raw = manifest.model_dump_json().encode()
    (path / "manifest.json").write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest()


def test_backup_verification_rejects_modified_payload_and_wrong_digest(tmp_path: Path) -> None:
    path, checksum = backup_files(tmp_path)
    assert len(verification.verify_files(path, checksum).objects) == 1
    with pytest.raises(MaintenanceError, match="SHA-256"):
        verification.verify_files(path, "0" * 64)
    next((path / "objects").iterdir()).write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="recorded size"):
        verification.verify_files(path, checksum)


def test_generated_helper_routes_mutations_through_maintenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from test_local_installation import installer

    value = installer(tmp_path, monkeypatch)
    value.write_commands({})
    content = (value.root / "service.sh").read_text()
    assert "start|stop|restart|backup|verify-backup|resume-maintenance)" in content
    assert "regimex.local_installation.maintenance" in content
    assert f"--root {value.root}" in content
    subprocess.run(["sh", "-n", str(value.root / "service.sh")], check=True)


def test_backup_creates_dedicated_parent(managed: Maintenance) -> None:
    target = managed.root / "backups/first"
    assert managed.destination(target) == target
    assert target.parent.is_dir()
    assert target.parent.stat().st_mode & 0o777 == 0o700
    assert not target.exists()


def test_interrupted_verification_start_stops_its_cluster(
    managed: Maintenance, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tempfile

    patch_engine(monkeypatch)
    path, checksum = backup_files(managed.root.parent)
    calls = []
    with tempfile.TemporaryDirectory(prefix="rx-v-", dir="/tmp") as temporary:
        root = Path(temporary)
        managed.root, managed.journal = root, root / maintenance.JOURNAL

        def run(command: list[str], *, timeout: int = 90) -> str:
            calls.append(command[-1])
            data = Path(command[command.index("-D") + 1])
            if command[0].endswith("initdb"):
                data.mkdir()
            elif command[-1] == "start":
                (data / "postmaster.pid").write_text("deliberate pid fixture")
                raise KeyboardInterrupt
            elif command[-1] == "stop":
                (data / "postmaster.pid").unlink()
            return ""

        monkeypatch.setattr(managed, "run", run)
        with pytest.raises(KeyboardInterrupt):
            managed.verify_backup(path, checksum)
        assert "start" in calls and calls[-1] == "stop"
        assert not managed.journal.exists()
        assert list((root / "verification").iterdir()) == []


def test_real_isolated_restore_preserves_live_source(
    managed: Maintenance, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import tempfile

    psycopg = pytest.importorskip("psycopg")
    pytest.importorskip("numpy")
    pytest.importorskip("sklearn")
    pytest.importorskip("plotly")
    pytest.importorskip("procrastinate")
    pytest.importorskip("alembic")
    from psycopg import sql
    from psycopg.conninfo import make_conninfo
    from pydantic import SecretStr
    from test_service_workflow import published_model

    from regimex.application.recovery import backup_service
    from regimex.application.service import AnalysisService
    from regimex.infrastructure.database import DatabaseRepository
    from regimex.infrastructure.maintenance import ServicesStoppedAttestation
    from regimex.infrastructure.object_store import LocalObjectStore
    from regimex.model_registry import ArchitectureRegistry
    from regimex.recovery_contracts import PostgresTools
    from regimex.service_settings import ServiceSettings

    dsn = os.environ.get("REGIMEX_TEST_DATABASE_DSN")
    if not dsn:
        pytest.skip("isolated verification requires a disposable PostgreSQL cluster with CREATEDB")
    try:
        tools = PostgresTools.discover()
    except ValueError, FileNotFoundError:
        pytest.skip("isolated verification requires PostgreSQL recovery tools")
    pg = tools.pg_dump.parent
    if not all((pg / name).is_file() for name in ("initdb", "pg_ctl")):
        pytest.skip("isolated verification requires initdb and pg_ctl beside pg_dump")
    name = "regimex_backup_command_" + uuid4().hex
    with psycopg.connect(dsn, autocommit=True) as connection:
        connection.execute(
            sql.SQL("CREATE DATABASE {} TEMPLATE template0").format(sql.Identifier(name))
        )
    try:
        database = DatabaseRepository(make_conninfo(dsn, dbname=name))
        database.migrate()
        store = LocalObjectStore(
            tmp_path / "source-objects",
            max_object_bytes=16 * 1024**2,
            max_total_bytes=128 * 1024**2,
        )
        service = AnalysisService(
            database, store, ArchitectureRegistry.discover(), tmp_path / "source-work"
        )
        published_model(service, tmp_path)
        original = database.get_active_model()
        backup = backup_service(
            SecretStr(database._dsn),
            database.schema,
            store,
            tmp_path / "backup",
            tools=tools,
            attestation=ServicesStoppedAttestation.model_validate(
                maintenance.stopped_attestation()
            ),
        )
        settings = ServiceSettings(
            database_dsn=SecretStr(database._dsn),
            object_root=tmp_path / "source-objects",
            work_root=tmp_path / "source-work",
        )
        with tempfile.TemporaryDirectory(prefix="rx-v-", dir="/tmp") as temporary:
            root = Path(temporary)
            (root / "logs").mkdir(mode=0o700)
            managed.root, managed.journal, managed.pg = root, root / maintenance.JOURNAL, pg
            monkeypatch.setattr(managed, "settings", lambda: settings)

            def forbid_control(*args: str) -> None:
                pytest.fail("restore verification attempted to control the live service")

            monkeypatch.setattr(managed, "control", forbid_control)
            result = managed.verify_backup(tmp_path / "backup", backup.manifest_sha256)
            assert result["status"] == "restore_verified"
            assert original is not None
            assert result["active_model"] == original.model_version_id
            assert result["object_count"] == len(backup.manifest.objects)
            assert result["live_service_changed"] is False
            assert not managed.journal.exists()
            assert list((root / "verification").iterdir()) == []
        assert database.get_active_model() == original
        assert database.readiness()
        assert (
            tuple(store.stat(item.key) for item in backup.manifest.objects)
            == backup.manifest.objects
        )
    finally:
        with psycopg.connect(dsn, autocommit=True) as connection:
            connection.execute(
                sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name))
            )


def test_unhealthy_resumed_api_keeps_recovery_journal(
    managed: Maintenance, monkeypatch: pytest.MonkeyPatch
) -> None:
    states, _ = fake_units(managed, monkeypatch)
    original = {part: states[part] for part in WRITERS}
    managed.save_journal({"version": 1, "name": managed.name, "kind": "backup", "active": original})

    def not_ready() -> None:
        raise MaintenanceError("The resumed API is not ready.")

    monkeypatch.setattr(managed, "wait_ready", not_ready)
    with pytest.raises(MaintenanceError, match="not ready"):
        managed.resume()
    assert managed.journal.exists()
    assert {part: states[part] for part in WRITERS} == original
