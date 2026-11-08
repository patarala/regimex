"""Coordinated maintenance for the installer-owned native service topology."""

import argparse
import http.client
import json
import os
import re
import signal
import subprocess
import sys
import time
import traceback
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from regimex.service_contracts import ServiceError

from . import systemd
from .install import (
    environment_file,
    installation_lock,
    private_directory,
    read_private,
    write_private,
)

if TYPE_CHECKING:
    from regimex.service_settings import ServiceSettings

WRITERS = ("api", *systemd.WORKERS, "migrate")
JOURNAL = "maintenance-state.json"


class MaintenanceError(ValueError):
    """A safe operator-facing maintenance error."""


def command_environment(values: dict[str, str]) -> dict[str, str]:
    return {
        **{
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("REGIMEX_", "PG"))
            and key not in {"PYTHONPATH", "VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT"}
        },
        **values,
    }


@contextmanager
def configured_environment(values: dict[str, str]) -> Iterator[None]:
    previous = dict(os.environ)
    environment = command_environment(values)
    os.environ.clear()
    os.environ.update(environment)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(previous)


def stopped_attestation() -> dict[str, object]:
    return {
        "stopped_at": datetime.now(UTC),
        "statement": "Installer-owned writers are stopped; maintenance owns the private store.",
        "api_stopped": True,
        "analysis_workers_stopped": True,
        "administrative_workers_stopped": True,
        "maintenance_stopped": True,
        "garbage_collection_stopped": True,
        "exclusive_object_access": True,
    }


def pending_maintenance(root: Path) -> None:
    if (root / JOURNAL).exists() or (root / JOURNAL).is_symlink():
        raise MaintenanceError(
            "Maintenance is incomplete. Run service.sh resume-maintenance first."
        )


class Maintenance:
    def __init__(self, root: Path) -> None:
        self.root = systemd.safe_path(root.resolve(strict=True))
        private_directory(self.root)
        state = json.loads(read_private(self.root / "local-installation.json"))
        if state.get("schema_version") != 1 or state.get("phase") != "ready":
            raise MaintenanceError("Complete or recover the installation before maintenance.")
        self.name = str(state["name"])
        systemd.unit_name(self.name, "target")
        self.pg = systemd.safe_path(Path(state["postgres_bin"]))
        self.environment = environment_file(self.root, int(state["port"]), existing=True)
        self.release = Path(state["active_release"])
        if self.release.parent != self.root / "releases":
            raise MaintenanceError("The active runtime does not belong to this installation.")
        self.source_commit = str(state["source_commit"])
        self.journal = self.root / JOURNAL
        systemd.check_existing(self.root, self.name, adopt=True, require_complete=True)
        private_directory(self.root / "logs")

    def run(self, command: list[str], *, timeout: int = 90) -> str:
        with (self.root / "logs/maintenance.log").open("ab") as log:
            process = subprocess.Popen(
                command,
                env=command_environment(self.environment),
                cwd=self.root,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=log,
                start_new_session=True,
            )
            try:
                output, _ = process.communicate(timeout=timeout)
            except BaseException as error:
                # A cancelled pg_ctl/initdb must not keep creating the workspace
                # while the caller resumes services or removes its private copy.
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
                if isinstance(error, subprocess.TimeoutExpired):
                    raise MaintenanceError("Maintenance command exceeded its time limit.") from None
                raise
            if process.returncode:
                raise MaintenanceError("Maintenance command failed; inspect logs/maintenance.log.")
            return output.decode()

    def control(self, action: str, *components: str) -> None:
        if components:
            self.run(
                systemd.command(action, *(systemd.unit_name(self.name, x) for x in components))
            )

    def active(self, component: str) -> bool:
        state = self.run(
            systemd.command(
                "show", "--property=ActiveState", "--value", systemd.unit_name(self.name, component)
            )
        ).strip()
        if state not in {"active", "inactive"}:
            raise MaintenanceError(
                "A service is transitioning or failed; inspect service.sh status first."
            )
        return state == "active"

    def wait_ready(self) -> None:
        deadline = time.monotonic() + 45
        port = int(self.environment["REGIMEX_PUBLIC_ORIGIN"].rsplit(":", 1)[1])
        while time.monotonic() < deadline:
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
            try:
                connection.request("GET", "/health/ready")
                response = connection.getresponse()
                raw = response.read(8193)
                if response.status == 200 and len(raw) <= 8192:
                    payload = json.loads(raw)
                    if isinstance(payload, dict) and payload.get("status") == "ready":
                        return
            except OSError, ValueError, http.client.HTTPException:
                pass
            finally:
                connection.close()
            time.sleep(1)
        raise MaintenanceError(
            "The resumed API is not ready; inspect it and retry resume-maintenance."
        )

    def settings(self) -> ServiceSettings:
        from regimex.service_settings import ServiceSettings

        with configured_environment(self.environment):
            return ServiceSettings.from_env()

    def idle(self) -> None:
        from .probe import inspect

        with configured_environment(self.environment):
            status = inspect(self.root)
        if status["pending"] != 0 or status["doing"] != 0:
            raise MaintenanceError(
                "Backup refused: wait for queued/running work to finish, then retry."
            )

    def save_journal(self, value: dict[str, object]) -> None:
        write_private(self.journal, json.dumps(value, indent=2) + "\n")
        self.sync_root()

    def clear_journal(self) -> None:
        self.journal.unlink()
        self.sync_root()

    def sync_root(self) -> None:
        descriptor = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def resume(self) -> None:
        if not self.journal.exists():
            return
        value = json.loads(read_private(self.journal))
        if value.get("version") != 1 or value.get("name") != self.name:
            raise MaintenanceError("Maintenance journal does not match this installation.")
        if value.get("kind") == "verification":
            from .verification import cleanup_workspace

            cleanup_workspace(self, Path(value["workspace"]))
        elif value.get("kind") == "backup":
            active = value.get("active")
            if (
                not isinstance(active, dict)
                or set(active) != set(WRITERS)
                or any(type(item) is not bool for item in active.values())
            ):
                raise MaintenanceError("Maintenance journal has invalid service states.")
            # Restore dependencies before intake; never start a previously stopped writer.
            for component in ("migrate", *systemd.WORKERS, "api"):
                if active[component]:
                    self.control("start", component)
            for component in WRITERS:
                if self.active(component) != active[component]:
                    raise MaintenanceError(
                        "Service states were not restored; retry resume-maintenance."
                    )
            if active["api"]:
                self.wait_ready()
        else:
            raise MaintenanceError("Maintenance journal has an unknown operation.")
        self.clear_journal()

    def destination(self, destination: Path) -> Path:
        if destination.is_symlink():
            raise MaintenanceError("Backup destinations cannot be symbolic links.")
        destination = destination.absolute()
        if destination.parent == self.root / "backups":
            private_directory(destination.parent)
        parent = destination.parent.resolve(strict=True)
        destination = parent / destination.name
        if destination.exists():
            raise MaintenanceError(
                "A backup destination must be new; existing paths are never overwritten."
            )
        if self.root.is_relative_to(destination):
            raise MaintenanceError("A backup destination cannot contain the installation.")
        if destination.is_relative_to(self.root) and not destination.is_relative_to(
            self.root / "backups"
        ):
            raise MaintenanceError(
                "Inside the installation, place backups only in its backups/ directory."
            )
        private_directory(parent)
        return destination

    def backup(self, destination: Path) -> dict[str, object]:
        from regimex.application.recovery import backup_service
        from regimex.infrastructure.maintenance import ServicesStoppedAttestation
        from regimex.recovery_contracts import PostgresTools
        from regimex.service_bootstrap import create_object_store

        from .verification import verify_files

        pending_maintenance(self.root)
        destination = self.destination(destination)
        settings = self.settings()
        tools = PostgresTools.discover(
            pg_dump=self.pg / "pg_dump", pg_restore=self.pg / "pg_restore"
        )
        if not self.active("postgres"):
            raise MaintenanceError("Start PostgreSQL before creating a backup.")
        active = {part: self.active(part) for part in WRITERS}
        if any(active[x] for x in ("api", *systemd.WORKERS)) and not active["migrate"]:
            raise MaintenanceError(
                "Writer dependencies are inconsistent; inspect the service first."
            )
        self.save_journal({"version": 1, "name": self.name, "kind": "backup", "active": active})
        try:
            if active["api"]:
                self.control("stop", "api")
            self.idle()
            self.control("stop", *systemd.WORKERS, "migrate")
            if any(self.active(part) for part in WRITERS):
                raise MaintenanceError("Some writers remain active; backup did not start.")
            self.idle()
            result = backup_service(
                settings.database_dsn,
                settings.database_schema,
                create_object_store(settings),
                destination,
                tools=tools,
                attestation=ServicesStoppedAttestation.model_validate(stopped_attestation()),
            )
            verify_files(destination, result.manifest_sha256)
            return {
                "status": "backup_verified",
                "backup": str(destination),
                "manifest_sha256": result.manifest_sha256,
                "backup_id": str(result.manifest.backup_id),
                "object_count": len(result.manifest.objects),
                "source_commit": self.source_commit,
            }
        finally:
            self.resume()

    def verify_backup(self, source: Path, expected: str) -> dict[str, object]:
        from .verification import verify_restore

        pending_maintenance(self.root)
        return verify_restore(self, source, expected)


def _interrupt(signum: int, frame: object) -> None:
    raise KeyboardInterrupt


def main() -> int:
    parser = argparse.ArgumentParser(description="Maintain one installed native Regimex service.")
    parser.add_argument("--root", required=True, type=Path)
    commands = parser.add_subparsers(dest="action", required=True)
    backup = commands.add_parser(
        "backup", help="Pause writers, verify a backup and resume services."
    )
    backup.add_argument("--output", required=True, type=Path)
    verify = commands.add_parser(
        "verify-backup", help="Restore and check a disposable private copy."
    )
    verify.add_argument("--input", required=True, type=Path)
    verify.add_argument("--manifest-sha256", required=True)
    commands.add_parser("resume-maintenance", help="Recover interrupted maintenance safely.")
    for action in ("start", "stop", "restart"):
        commands.add_parser(action)
    args = parser.parse_args()
    os.umask(0o077)
    signal.signal(signal.SIGTERM, _interrupt)
    try:
        root = systemd.safe_path(args.root.resolve(strict=True))
        with installation_lock(root):
            maintenance = Maintenance(root)
            if args.action == "backup":
                result = maintenance.backup(args.output)
            elif args.action == "verify-backup":
                if re.fullmatch(r"[0-9a-f]{64}", args.manifest_sha256) is None:
                    raise MaintenanceError(
                        "Pass the externally recorded 64-character manifest SHA-256."
                    )
                result = maintenance.verify_backup(args.input, args.manifest_sha256)
            elif args.action == "resume-maintenance":
                maintenance.resume()
                result = {"status": "resumed"}
            else:
                pending_maintenance(root)
                maintenance.control(args.action, "target")
                result = {"status": args.action}
        print(json.dumps(result, indent=2))
        return 0
    except KeyboardInterrupt:
        print(
            "Maintenance interrupted. Run service.sh resume-maintenance if needed.", file=sys.stderr
        )
        return 130
    except (MaintenanceError, ServiceError) as error:
        print(f"Maintenance stopped: {error}", file=sys.stderr)
        return 1
    except Exception:
        # Driver exceptions can include DSNs; keep details in the private log.
        try:
            with (args.root / "logs/maintenance.log").open("a") as log:
                traceback.print_exc(file=log)
        except OSError:
            pass
        print(
            "Maintenance failed. Check logs/maintenance.log and service.sh status. "
            "An unfinished operation requires service.sh resume-maintenance.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
