"""Verify a backup in a fresh private PostgreSQL cluster and object namespace."""

import hashlib
import json
import os
import re
import shutil
import stat
import time
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from .install import private_directory, read_private, write_private
from .maintenance import MaintenanceError, configured_environment, stopped_attestation

if TYPE_CHECKING:
    from regimex.recovery_contracts import BackupManifest

    from .maintenance import Maintenance


def verify_files(source: Path, expected: str) -> BackupManifest:
    from regimex.application.recovery import verified_file
    from regimex.recovery_contracts import BackupManifest, RecoveryLimits

    limits = RecoveryLimits()
    deadline = time.monotonic() + limits.timeout_seconds
    descriptor = os.open(source / "manifest.json", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise MaintenanceError("The backup manifest must be a regular file.")
        data = stream.read(limits.max_manifest_bytes + 1)
    if len(data) > limits.max_manifest_bytes or hashlib.sha256(data).hexdigest() != expected:
        raise MaintenanceError("The backup manifest differs from the supplied SHA-256.")
    manifest = BackupManifest.model_validate_json(data)
    if sum(item.size_bytes for item in manifest.objects) > limits.max_total_object_bytes:
        raise MaintenanceError("The backup exceeds the recovery object-byte limit.")
    verified_file(
        source / "database.dump",
        maximum=limits.max_dump_bytes,
        size=manifest.dump_size_bytes,
        sha256=manifest.dump_sha256,
        deadline=deadline,
    )
    for item in manifest.objects:
        verified_file(
            source / "objects" / item.sha256,
            maximum=limits.max_total_object_bytes,
            size=item.size_bytes,
            sha256=item.sha256,
            deadline=deadline,
        )
    return manifest


def cleanup_workspace(maintenance: Maintenance, workspace: Path) -> None:
    parent = maintenance.root / "verification"
    if workspace.parent != parent or re.fullmatch(r"restore-[0-9a-f]{32}", workspace.name) is None:
        raise MaintenanceError("The verification workspace does not belong to this installation.")
    if not workspace.exists() and not workspace.is_symlink():
        return
    private_directory(parent)
    private_directory(workspace)
    marker = json.loads(read_private(workspace / "owner.json"))
    if marker != {"installation": str(maintenance.root), "workspace": workspace.name}:
        raise MaintenanceError("The verification workspace ownership marker differs.")
    data = workspace / "postgres"
    if (data / "postmaster.pid").exists():
        maintenance.run(
            [
                str(maintenance.pg / "pg_ctl"),
                "-D",
                str(data),
                "-m",
                "fast",
                "-w",
                "-t",
                "60",
                "stop",
            ]
        )
    # Never remove a database directory unless its server has really stopped.
    if (data / "postmaster.pid").exists():
        raise MaintenanceError("The verification server remains active; its files were retained.")
    # Model probes may seal cache directories. Change only owned directory modes;
    # immutable files may be hard links and must retain their permissions.
    for directory, children, _ in os.walk(workspace):
        os.chmod(directory, 0o700)
        for child in children:
            path = Path(directory) / child
            if not path.is_symlink():
                os.chmod(path, 0o700)
    shutil.rmtree(workspace)


def verify_restore(maintenance: Maintenance, source: Path, expected: str) -> dict[str, object]:
    import psycopg
    from psycopg.conninfo import make_conninfo
    from pydantic import SecretStr

    from regimex.application.recovery import restore_service
    from regimex.infrastructure.database import DatabaseRepository
    from regimex.infrastructure.maintenance import ServicesStoppedAttestation
    from regimex.infrastructure.object_store import LocalObjectStore
    from regimex.recovery_contracts import PostgresTools
    from regimex.service_diagnostics import diagnose

    source = source.resolve(strict=True)
    private_directory(source)
    manifest = verify_files(source, expected)
    tools = PostgresTools.discover(
        pg_dump=maintenance.pg / "pg_dump", pg_restore=maintenance.pg / "pg_restore"
    )
    parent = maintenance.root / "verification"
    private_directory(parent)
    required = (
        sum(item.size_bytes for item in manifest.objects)
        + 2 * manifest.dump_size_bytes
        + 256 * 1024**2
    )
    if shutil.disk_usage(parent).free < required:
        raise MaintenanceError("There is insufficient disk space for an isolated restore check.")
    workspace = parent / ("restore-" + uuid4().hex)
    if len(os.fsencode(workspace / "socket")) > 90:
        raise MaintenanceError(
            "The installation path is too long for a private verification socket."
        )
    private_directory(workspace)
    write_private(
        workspace / "owner.json",
        json.dumps({"installation": str(maintenance.root), "workspace": workspace.name}),
    )
    maintenance.save_journal(
        {
            "version": 1,
            "name": maintenance.name,
            "kind": "verification",
            "workspace": str(workspace),
        }
    )
    try:
        data, socket = workspace / "postgres", workspace / "socket"
        private_directory(socket)
        private_directory(workspace / "tmp")
        maintenance.run(
            [
                str(maintenance.pg / "initdb"),
                "-D",
                str(data),
                "--username=regimex_verify",
                "--auth-local=trust",
                "--auth-host=reject",
                "--encoding=UTF8",
                "--locale=C",
            ]
        )
        write_private(
            data / "postgresql.auto.conf",
            "listen_addresses = ''\nport = 5432\n"
            f"unix_socket_directories = '{socket}'\nunix_socket_permissions = 0700\n",
        )
        maintenance.run(
            [
                str(maintenance.pg / "pg_ctl"),
                "-D",
                str(data),
                "-l",
                str(workspace / "postgres.log"),
                "-w",
                "-t",
                "60",
                "start",
            ]
        )
        parameters = make_conninfo(
            host=str(socket), port=5432, user="regimex_verify", connect_timeout=5
        )
        with psycopg.connect(
            make_conninfo(parameters, dbname="postgres"), autocommit=True
        ) as connection:
            connection.execute("CREATE DATABASE regimex_restore TEMPLATE template0")
        dsn = SecretStr(make_conninfo(parameters, dbname="regimex_restore"))
        configured = maintenance.settings()
        largest = max((item.size_bytes for item in manifest.objects), default=1)
        total = max(largest, sum(item.size_bytes for item in manifest.objects), 1)
        store = LocalObjectStore(
            workspace / "objects", max_object_bytes=largest, max_total_bytes=total
        )
        with configured_environment({**maintenance.environment, "TMPDIR": str(workspace / "tmp")}):
            receipt = restore_service(
                dsn,
                store,
                source,
                expected_manifest_sha256=expected,
                assert_empty_store=store.assert_empty_store,
                tools=tools,
                attestation=ServicesStoppedAttestation.model_validate(stopped_attestation()),
            )
            settings = configured.model_copy(
                update={
                    "database_dsn": dsn,
                    "database_schema": manifest.schema_name,
                    "object_root": workspace / "objects",
                    "work_root": workspace / "work",
                    "max_object_bytes": largest,
                    "max_store_bytes": total,
                }
            )
            report = diagnose(settings)
        required_checks = {"database", "schema", "storage", "active_model", "model_load"}
        checks = {item.name: item.status for item in report.checks if item.name in required_checks}
        if set(checks) != required_checks or any(state != "passed" for state in checks.values()):
            raise MaintenanceError(
                "The restored copy failed readiness or active-model verification."
            )
        repository = DatabaseRepository(dsn.get_secret_value(), schema=manifest.schema_name)
        model = repository.get_active_model()
        assert model is not None
        return {
            "status": "restore_verified",
            "backup_id": str(receipt.backup_id),
            "manifest_sha256": expected,
            "object_count": receipt.object_count,
            "tables_verified": len(receipt.table_evidence),
            "active_model": model.model_version_id,
            "checks": checks,
            "live_service_changed": False,
        }
    finally:
        maintenance.resume()
