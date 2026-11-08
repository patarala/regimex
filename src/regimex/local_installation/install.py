"""Stage, quiesce and update one supported local installation without data rollback."""

import fcntl
import json
import os
import re
import secrets
import shlex
import shutil
import socket
import subprocess
import time
import tomllib
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener
from uuid import uuid4

from . import systemd
from .archive import Release


@dataclass(frozen=True)
class Options:
    archive: Path
    sha256: str
    root: Path
    postgres_bin: Path | None = None
    name: str = "regimex"
    port: int = 8000
    postgres_port: int = 55439
    adopt: bool = False


def private_directory(path: Path) -> None:
    if path.is_symlink():
        raise ValueError("Installation directories cannot be symbolic links.")
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    info = path.stat()
    if info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError("Installation directories must be private and owned by this user.")


def write_private(path: Path, content: str) -> None:
    if path.is_symlink():
        raise ValueError("Installation files cannot be symbolic links.")
    temporary = path.with_name(path.name + ".incoming-" + uuid4().hex)
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_private(path: Path) -> str:
    info = path.lstat()
    if (
        path.is_symlink()
        or not path.is_file()
        or info.st_uid != os.getuid()
        or info.st_mode & 0o077
    ):
        raise ValueError("Installation files must be private, owned regular files.")
    if info.st_size > 65536:
        raise ValueError("Installation metadata exceeds its size limit.")
    return path.read_text()


@contextmanager
def installation_lock(root: Path) -> Iterator[None]:
    descriptor = os.open(root / ".installation.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError(
                "Another installer or maintenance command owns this installation."
            ) from None
        yield


def postgres_bin(selected: Path | None) -> Path:
    candidates = (
        [selected]
        if selected is not None
        else [
            Path(shutil.which("pg_ctl") or "/missing").parent,
            Path("/usr/lib/postgresql/16/bin"),
        ]
    )
    for candidate in candidates:
        assert candidate is not None
        candidate = systemd.safe_path(candidate.resolve())
        if all((candidate / name).is_file() for name in ("pg_ctl", "postgres", "initdb")):
            version = subprocess.run(
                [str(candidate / "postgres"), "--version"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            if re.fullmatch(r"postgres \(PostgreSQL\) 16\.\d+\s*", version):
                return candidate
    raise ValueError("Install PostgreSQL 16 binaries or pass --postgres-bin /path/to/pg16/bin.")


def environment_file(root: Path, port: int, *, existing: bool) -> dict[str, str]:
    expected = {
        "REGIMEX_DATABASE_DSN_FILE": str(root / "credentials/database.dsn"),
        "REGIMEX_OBJECT_ROOT": str(root / "objects"),
        "REGIMEX_WORK_ROOT": str(root / "work"),
        "REGIMEX_PUBLIC_ORIGIN": f"http://127.0.0.1:{port}",
        "REGIMEX_ALLOW_LOOPBACK_HTTP": "true",
    }
    if not existing:
        return expected
    values = {}
    for line in read_private(root / "service.env").splitlines():
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator or re.fullmatch(r"[A-Z][A-Z0-9_]*", key) is None or "\x00" in value:
            raise ValueError("Existing service.env is not a supported plain environment file.")
        values[key] = value
    if any(values.get(key) != value for key, value in expected.items()):
        raise ValueError(
            "Existing installation roots/origin differ; use its original --root and --port."
        )
    if values.get("REGIMEX_OBJECT_BACKEND", "local") != "local" or "REGIMEX_DATABASE_DSN" in values:
        raise ValueError("Adoption supports only local storage with a private database DSN file.")
    read_private(root / "credentials/database.dsn")
    return values


class Installer:
    def __init__(self, options: Options) -> None:
        self.options = options
        self.root = systemd.safe_path(options.root.resolve())
        systemd.unit_name(options.name, "target")
        if not 1 <= options.port <= 65535 or not 1 <= options.postgres_port <= 65535:
            raise ValueError("Service and PostgreSQL ports must be between 1 and 65535.")
        self.release = Release(options.archive.resolve(), options.sha256)
        self.source = self.root / "releases" / self.release.sha256 / "source"
        self.python = self.source.parent / "venv/bin/python"
        self.state_path = self.root / "local-installation.json"
        selected = options.postgres_bin
        if selected is None and self.state_path.is_file():
            recorded = json.loads(read_private(self.state_path))
            selected = Path(recorded["postgres_bin"])
        if selected is None and options.adopt:
            supplied = self.root / "runtime/pgserver/pginstall/bin"
            selected = supplied if supplied.is_dir() else None
        self.pg = postgres_bin(selected)
        self.state: dict[str, object] = {}
        self.environment = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("REGIMEX_")
            and key not in {"PYTHONPATH", "VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT"}
        }

    def state_change(self, phase: str) -> None:
        self.state["phase"] = phase
        write_private(self.state_path, json.dumps(self.state, indent=2, sort_keys=True) + "\n")
        print(phase.replace("_", " "), flush=True)

    def run(self, command: list[str], *, timeout: int = 180, capture: bool = False) -> str:
        with (self.root / "logs/installation.log").open("ab") as log:
            try:
                result = subprocess.run(
                    command,
                    env=self.environment,
                    cwd=self.root,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE if capture else log,
                    stderr=log,
                    timeout=timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                raise RuntimeError("Installation command exceeded its time limit.") from None
            if result.returncode:
                raise RuntimeError(
                    "Installation command failed; inspect the private logs/installation.log."
                )
            return result.stdout.decode() if capture else ""

    def control(self, action: str, *components: str) -> None:
        self.run(
            systemd.command(
                action, *(systemd.unit_name(self.options.name, part) for part in components)
            )
        )

    def probe(self, action: str) -> dict[str, object] | None:
        result = self.run(
            [
                str(self.python),
                str(Path(__file__).with_name("probe.py")),
                action,
                "--root",
                str(self.root),
                "--source",
                str(self.source),
                "--port",
                str(self.options.postgres_port),
            ],
            timeout=300,
            capture=True,
        )
        value = json.loads(result)
        return value if isinstance(value, dict) else None

    def validate_database_configuration(self) -> None:
        """Read owned cluster configuration without granting the application extra privileges."""
        expected = {
            "port": str(self.options.postgres_port),
            "listen_addresses": "",
            "unix_socket_directories": str(self.root / "postgres/socket"),
        }
        for setting, value in expected.items():
            actual = self.run(
                [
                    str(self.pg / "postgres"),
                    "-D",
                    str(self.root / "postgres/data"),
                    "-C",
                    setting,
                ],
                capture=True,
            ).strip()
            if actual != value:
                raise ValueError(
                    "Existing PostgreSQL configuration differs; use its original port "
                    "and private socket configuration."
                )

    def stage(self) -> None:
        self.state_change("staging")
        self.release.extract(self.source)
        marker = self.source.parent / "staged.json"
        identity = {"archive_sha256": self.release.sha256, "wheel": self.release.wheel}
        if marker.exists():
            if json.loads(read_private(marker)) != identity or not self.python.is_file():
                raise ValueError("Staged runtime metadata differs; inspect it before recovery.")
            # Activation may have completed immediately before an interruption.
            # Completed candidate environments are immutable, including on retry.
            self.state_change("staged")
            return
        if shutil.disk_usage(self.root).free < 3 * 1024**3:
            raise ValueError("Dependency staging requires at least 3 GiB of free disk space.")
        configuration = tomllib.loads((self.source / "pyproject.toml").read_text())
        required = configuration["tool"]["uv"]["required-version"].removeprefix("==")
        uv = shutil.which("uv")
        if uv is None or self.run([uv, "--version"], capture=True).split()[:2] != ["uv", required]:
            raise ValueError(f"Install uv {required} before running this installer.")
        python_version = (self.source / ".python-version").read_text().strip()
        if re.fullmatch(r"3\.14\.\d+", python_version) is None:
            raise ValueError("This installer supports release-pinned Python 3.14 only.")
        self.environment["UV_PROJECT_ENVIRONMENT"] = str(self.python.parent.parent)
        self.run(
            [
                uv,
                "sync",
                "--locked",
                "--no-default-groups",
                "--all-extras",
                "--no-install-project",
                "--project",
                str(self.source),
                "--python",
                python_version,
            ],
            timeout=1800,
        )
        self.run(
            [
                uv,
                "pip",
                "install",
                "--python",
                str(self.python),
                "--no-deps",
                str(self.source / self.release.wheel),
            ],
            timeout=300,
        )
        self.run(
            [
                str(self.python),
                "-c",
                "import numpy, sklearn, torch, plotly, psycopg; assert (torch.ones(1)+1).item()==2",
            ]
        )
        write_private(marker, json.dumps(identity) + "\n")
        self.state_change("staged")

    def write_commands(self, environment: dict[str, str]) -> None:
        """Keep adoption's documented operator commands on the newly installed runtime."""
        write_private(
            self.root / "environment.sh",
            "".join(f"export {key}={shlex.quote(value)}\n" for key, value in environment.items()),
        )
        target = systemd.unit_name(self.options.name, "target")
        command = (
            "#!/bin/sh\nset -eu\n"
            'case "${1:-status}" in\n'
            f"  start|stop|restart|backup|verify-backup|resume-maintenance) "
            f"exec {self.python} -m regimex.local_installation.maintenance "
            f'--root {self.root} "$@" ;;\n'
            f"  status) exec systemctl --user --no-pager status {target} ;;\n"
            f"  doctor) shift; . {self.root / 'environment.sh'}; "
            f'exec {self.python} -m regimex service doctor "$@" ;;\n'
            '  *) echo "Usage: service.sh {start|stop|restart|status|doctor [--json]|'
            'backup|verify-backup|resume-maintenance}" '
            ">&2; exit 2 ;;\n"
            "esac\n"
        )
        write_private(self.root / "service.sh", command)
        (self.root / "service.sh").chmod(0o700)

    def provision(self) -> None:
        self.state_change("provisioning")
        for name in ("postgres", "postgres/socket", "credentials", "objects", "work"):
            private_directory(self.root / name)
        data = self.root / "postgres/data"
        password = self.root / "credentials/postgres-bootstrap.password"
        if not data.exists():
            if not password.exists():
                write_private(password, secrets.token_urlsafe(40) + "\n")
            self.run(
                [
                    str(self.pg / "initdb"),
                    "-D",
                    str(data),
                    "--username=regimex_bootstrap",
                    "--auth-local=scram-sha-256",
                    "--auth-host=reject",
                    "--encoding=UTF8",
                    "--locale=C",
                    "--pwfile=" + str(password),
                ]
            )
        elif (data / "PG_VERSION").read_text().strip() != "16":
            raise ValueError(
                "Existing PostgreSQL data is not version 16; "
                "no automatic major upgrade is attempted."
            )
        write_private(
            data / "regimex.conf",
            f"listen_addresses = ''\nport = {self.options.postgres_port}\n"
            f"unix_socket_directories = '{self.root / 'postgres/socket'}'\n"
            "unix_socket_permissions = 0700\npassword_encryption = 'scram-sha-256'\n",
        )
        inclusion = "include = 'regimex.conf'"
        if inclusion not in (data / "postgresql.conf").read_text().splitlines():
            with (data / "postgresql.conf").open("a") as output:
                output.write("\n" + inclusion + "\n")
        self.control("start", "postgres")
        self.probe("provision")

    def quiesce(self) -> None:
        self.state_change("checking_pending_jobs")
        self.control("stop", "api")
        try:
            status = self.probe("inspect")
            if status is not None and status.get("postgres_port") != self.options.postgres_port:
                raise ValueError("Use the existing database's --postgres-port for adoption/update.")
            if status is None or status.get("pending") != 0 or status.get("doing") != 0:
                raise ValueError(
                    "Update refused: wait for all queued/running work to finish, then retry."
                )
        except BaseException:
            self.control("start", "api")
            self.state_change("intake_resumed")
            raise
        self.control("stop", *systemd.WORKERS, "migrate")
        self.state_change("services_stopped")
        status = self.probe("inspect")
        if status is None or status.get("pending") != 0 or status.get("doing") != 0:
            raise ValueError(
                "Work appeared during shutdown. Services remain stopped; "
                "inspect and recover explicitly."
            )

    def verify(self) -> None:
        self.state_change("verifying")
        opener = build_opener(ProxyHandler({}))
        deadline = time.monotonic() + 60
        credential = read_private(self.root / "credentials/user.token").strip()
        while True:
            try:
                self.run(
                    systemd.command(
                        "is-active",
                        "--quiet",
                        *(
                            systemd.unit_name(self.options.name, component)
                            for component in ("postgres", "api", *systemd.WORKERS)
                        ),
                    )
                )
                for endpoint in ("ready", "models"):
                    with opener.open(
                        f"http://127.0.0.1:{self.options.port}/health/{endpoint}", timeout=3
                    ) as response:
                        value = json.loads(response.read(8192))
                        healthy = (
                            value.get("status") == "ready"
                            if endpoint == "ready"
                            else value.get("deployment_configured") is True
                        )
                        if response.status != 200 or not healthy:
                            raise ValueError("Service readiness is incomplete.")
                request = Request(
                    f"http://127.0.0.1:{self.options.port}/api/v1/service-status",
                    headers={"Authorization": "Bearer " + credential},
                )
                with opener.open(request, timeout=3) as response:
                    if json.loads(response.read(8192)).get("state") not in {"available", "busy"}:
                        raise ValueError("This installation has no available analysis worker.")
                report = json.loads(
                    self.run(
                        [
                            str(self.python),
                            "-m",
                            "regimex",
                            "service",
                            "doctor",
                            "--json",
                        ],
                        timeout=300,
                        capture=True,
                    )
                )
                if report.get("healthy") is not True:
                    raise ValueError("Service diagnostics did not pass.")
                break
            except (OSError, ValueError, RuntimeError) as error:
                if isinstance(error, HTTPError):
                    error.close()
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        "Service health/doctor checks failed; inspect the installation status."
                    ) from None
                time.sleep(1)
        self.state["active_release"] = str(self.source.parent)
        self.state_change("ready")

    def install(self) -> None:
        systemd.available()
        private_directory(self.root)
        private_directory(self.root / "logs")
        with installation_lock(self.root):
            from .maintenance import pending_maintenance

            pending_maintenance(self.root)
            recorded = self.state_path.exists()
            existing = self.options.adopt
            if recorded:
                self.state = json.loads(read_private(self.state_path))
                if (
                    self.state.get("schema_version") != 1
                    or type(self.state.get("initialized")) is not bool
                ):
                    raise ValueError("Installation state is invalid; inspect it before recovery.")
                if (
                    self.state.get("name") != self.options.name
                    or self.state.get("port") != self.options.port
                    or self.state.get("postgres_port") != self.options.postgres_port
                ):
                    raise ValueError(
                        "Use the installation's original name and HTTP/PostgreSQL ports."
                    )
                existing = self.state.get("initialized") is True
                if (
                    self.state.get("phase") not in {"ready", "intake_resumed", "staging", "staged"}
                    and self.state.get("archive_sha256") != self.release.sha256
                ):
                    raise ValueError(
                        "Resume the interrupted installation with the same archive first."
                    )
            elif self.options.adopt:
                if not (self.root / "runtime/installation.json").is_file():
                    raise ValueError(
                        "--adopt requires the documented existing native installation layout."
                    )
            elif any(self.root.iterdir()) and any(
                (self.root / name).exists()
                for name in ("postgres", "objects", "credentials", "service.env")
            ):
                raise ValueError(
                    "Existing runtime data requires explicit --adopt; it will not be overwritten."
                )
            systemd.check_existing(
                self.root,
                self.options.name,
                adopt=recorded or existing,
                require_complete=self.options.adopt and not recorded,
            )
            environment = environment_file(
                self.root,
                self.options.port,
                existing=existing and (self.root / "service.env").exists(),
            )
            self.environment.update(environment)
            if existing:
                self.validate_database_configuration()
            if not existing:
                with socket.socket() as listener:
                    try:
                        listener.bind(("127.0.0.1", self.options.port))
                    except OSError:
                        raise ValueError(
                            "The selected HTTP port is occupied; choose another --port."
                        ) from None
            if self.state.get("phase") == "ready" and self.state.get("active_release") == str(
                self.source.parent
            ):
                # Never synchronize dependencies into a currently running interpreter.
                self.verify()
                return
            self.state.update(
                {
                    "schema_version": 1,
                    "name": self.options.name,
                    "port": self.options.port,
                    "archive_sha256": self.release.sha256,
                    "source_commit": self.release.commit,
                    "postgres_bin": str(self.pg),
                    "postgres_port": self.options.postgres_port,
                    "initialized": existing,
                }
            )
            self.stage()
            rendered = systemd.units(
                self.root, self.python, self.pg, self.options.name, self.options.port
            )
            stopped = False
            try:
                if existing:
                    # Older native targets required every component. Explicitly
                    # stopping intake would then stop the target and PostgreSQL
                    # through PartOf. Change only this owned target before drain.
                    target = systemd.unit_name(self.options.name, "target")
                    systemd.write_units(self.root, {target: rendered[target]})
                    self.run(systemd.command("daemon-reload"))
                    self.quiesce()
                    stopped = True
                else:
                    write_private(
                        self.root / "service.env",
                        "".join(f"{key}={value}\n" for key, value in environment.items()),
                    )
                    systemd.write_units(self.root, rendered)
                    self.run(systemd.command("daemon-reload"))
                    self.provision()
                    stopped = True
                self.state_change("migrating")
                self.run([str(self.python), "-m", "regimex", "service", "migrate"])
                self.probe("tokens")
                if not existing:
                    self.probe("models")
                self.state["initialized"] = True
                self.state_change("activating")
                systemd.write_units(self.root, rendered)
                self.write_commands(environment)
                self.run(systemd.command("daemon-reload"))
                self.run(systemd.command("enable", systemd.unit_name(self.options.name, "target")))
                self.control("start", "migrate", *systemd.WORKERS, "api", "target")
                self.verify()
            except BaseException:
                if stopped or self.state.get("phase") == "services_stopped":
                    self.control("stop", "api", *systemd.WORKERS, "migrate")
                    self.state_change("recovery_required")
                raise
