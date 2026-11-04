"""Render and manage the small, explicitly supported user-service topology."""

import os
import re
import subprocess
from pathlib import Path
from uuid import uuid4

COMPONENTS = ("postgres", "migrate", "api", "analysis", "admin", "maintenance")
WORKERS = ("analysis", "admin", "maintenance")


def safe_path(path: Path) -> Path:
    path = path.absolute()
    if re.fullmatch(r"/[A-Za-z0-9_./-]+", str(path)) is None:
        raise ValueError("Installation paths must contain only letters, digits, /, _, . and -.")
    return path


def unit_name(name: str, component: str) -> str:
    if re.fullmatch(r"[a-z][a-z0-9-]{0,39}", name) is None:
        raise ValueError("--name must be a lowercase service name of at most 40 characters.")
    if component == "target":
        return f"{name}.target"
    if component not in COMPONENTS:
        raise ValueError("Unknown installation service component.")
    return f"{name}-{component}.service"


def units(root: Path, python: Path, postgres: Path, name: str, port: int) -> dict[str, str]:
    for path in (root, python, postgres):
        safe_path(path)
    target = unit_name(name, "target")
    pg = unit_name(name, "postgres")
    migrate = unit_name(name, "migrate")
    unit_paths = " ".join(unit_name(name, component) for component in COMPONENTS)
    result = {
        target: (
            "[Unit]\nDescription=Regimex local service\nWants="
            + unit_paths
            + "\nAfter="
            + unit_paths
            + "\n\n[Install]\nWantedBy=default.target\n"
        )
    }
    result[pg] = (
        f"[Unit]\nDescription=Regimex PostgreSQL\nPartOf={target}\n\n[Service]\n"
        f"Type=forking\nExecStart={postgres / 'pg_ctl'} -D {root / 'postgres/data'} "
        f"-l {root / 'logs/postgres.log'} -s -w -t 60 start\n"
        f"ExecStop={postgres / 'pg_ctl'} -D {root / 'postgres/data'} -s -m fast -w -t 60 stop\n"
        f"PIDFile={root / 'postgres/data/postmaster.pid'}\nRestart=on-failure\nRestartSec=3\n"
        "TimeoutStartSec=70\nTimeoutStopSec=70\nUMask=0077\n"
    )
    for component in COMPONENTS[1:]:
        dependency = pg if component == "migrate" else f"{pg} {migrate}"
        command = (
            "migrate"
            if component == "migrate"
            else (
                f"serve --host 127.0.0.1 --port {port}"
                if component == "api"
                else f"worker --queue {component}"
            )
        )
        options = (
            "Type=oneshot\nRemainAfterExit=yes\n"
            if component == "migrate"
            else (
                "Type=simple\nRestart=on-failure\nRestartSec=3\nKillMode=mixed\nTimeoutStopSec=45\n"
            )
        )
        result[unit_name(name, component)] = (
            f"[Unit]\nDescription=Regimex {component}\nRequires={dependency}\nAfter={dependency}\n"
            f"PartOf={target}\n\n[Service]\n{options}WorkingDirectory={root}\n"
            f"EnvironmentFile={root / 'service.env'}\n"
            f"ExecStart={python} -m regimex service {command}\nUMask=0077\n"
        )
    return result


def user_unit_directory() -> Path:
    base = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config")))
    return safe_path(base / "systemd/user")


def command(*arguments: str) -> list[str]:
    return ["systemctl", "--user", *arguments]


def available() -> None:
    if os.name != "posix" or not Path("/proc").is_dir():
        raise ValueError("Local installation requires Linux and a systemd user manager.")
    result = subprocess.run(command("show-environment"), capture_output=True, check=False)
    if result.returncode:
        raise ValueError(
            "The systemd user manager is unavailable. Start a normal user login session."
        )


def check_existing(root: Path, name: str, *, adopt: bool, require_complete: bool = False) -> None:
    """Never replace units owned by another installation or an unknown layout."""
    directory = user_unit_directory()
    for component in (*COMPONENTS, "target"):
        target = directory / unit_name(name, component)
        if not target.exists() and not target.is_symlink():
            if require_complete:
                raise ValueError(
                    "Adoption requires every existing unit under the requested --name."
                )
            continue
        if not adopt:
            raise ValueError(
                "Service units already exist. Select another --name or explicitly adopt."
            )
        content = target.read_text()
        recorded = root / "systemd" / target.name
        if not recorded.is_file() or content != recorded.read_text():
            raise ValueError(
                "Existing service units differ from the installation's recorded units."
            )
        if component != "target" and str(root) not in content:
            raise ValueError("An existing service unit belongs to another installation.")


def write_units(root: Path, rendered: dict[str, str]) -> None:
    directory = user_unit_directory()
    directory.mkdir(parents=True, exist_ok=True)
    owned = root / "systemd"
    owned.mkdir(exist_ok=True, mode=0o700)
    for name, content in rendered.items():
        path = owned / name
        path.write_text(content)
        path.chmod(0o600)
        target = directory / name
        # check_existing has verified ownership before the first mutation.
        pending = directory / (name + ".incoming-" + uuid4().hex)
        try:
            pending.symlink_to(path)
            pending.replace(target)
        finally:
            pending.unlink(missing_ok=True)
