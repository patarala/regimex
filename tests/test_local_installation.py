"""Release validation and safe interruption boundaries for local installation."""

import hashlib
import json
import os
import shutil
import signal
import socket
import stat
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipExtFile, ZipFile, ZipInfo

import pytest

from regimex.local_installation import archive, install, systemd
from regimex.local_installation.archive import Release
from regimex.local_installation.install import Installer, Options


def release_zip(root: Path, *, preview: bool = False) -> tuple[Path, str]:
    content = {
        "BUILD.json": json.dumps(
            {
                "schema_version": "1",
                "version": "0.1.0",
                "source": {"commit": "a" * 40, "working_tree_preview": preview},
                "assets": {"included": True},
            }
        ).encode(),
        "pyproject.toml": b'[tool.uv]\nrequired-version="==0.13.0"\n',
        "uv.lock": b"deliberate locked dependency fixture",
        ".python-version": b"3.14.4\n",
        "release/models/baseline/manifest.json": b"{}",
        "release/models/tcn/manifest.json": b"{}",
        "dist/regimex-0.1.0-py3-none-any.whl": b"deliberate archive-only wheel fixture",
    }
    content["SHA256SUMS"] = "".join(
        f"{hashlib.sha256(value).hexdigest()}  {name}\n" for name, value in content.items()
    ).encode()
    path = root / "release.zip"
    with ZipFile(path, "w") as output:
        for name, value in content.items():
            info = ZipInfo("regimex-0.1.0/" + name)
            info.external_attr = (stat.S_IFREG | 0o600) << 16
            output.writestr(info, value)
    return path, archive.digest(path)


@pytest.mark.parametrize("journal", ["absent", "file", "dangling_symlink"])
def test_standalone_installer_checks_existing_data_without_dependencies(
    tmp_path: Path, journal: str
) -> None:
    path, checksum = release_zip(tmp_path)
    root = tmp_path / "installation"
    root.mkdir(mode=0o700)
    objects = root / "objects"
    objects.mkdir(mode=0o700)
    payload = objects / "existing"
    payload.write_bytes(b"preserve existing data")
    pending = root / "maintenance-state.json"
    if journal == "file":
        pending.write_text("deliberately incomplete journal")
    elif journal == "dangling_symlink":
        pending.symlink_to(root / "missing-journal")

    binaries = tmp_path / "bin"
    binaries.mkdir()
    for name in ("postgres", "pg_ctl", "initdb", "systemctl"):
        command = binaries / name
        body = {
            "postgres": (
                '[ "$1" = "--version" ] || exit 99\n'
                'echo "postgres (PostgreSQL) 16.14 (Ubuntu 16.14-0ubuntu0.24.04.1)"'
            ),
            "systemctl": '[ "$*" = "--user show-environment" ] || exit 99',
        }.get(name, "exit 99")
        command.write_text("#!/bin/sh\n" + body + "\n")
        command.chmod(0o700)
    script = Path(__file__).resolve().parents[1] / "scripts/install_service.py"
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            str(script),
            "--archive",
            str(path),
            "--sha256",
            checksum,
            "--root",
            str(root),
            "--postgres-bin",
            str(binaries),
        ],
        cwd=tmp_path,
        env={**os.environ, "PATH": str(binaries)},
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 1
    assert "Traceback" not in result.stderr
    expected = "explicit --adopt" if journal == "absent" else "resume-maintenance first"
    assert expected in result.stderr
    assert payload.read_bytes() == b"preserve existing data"
    assert not (root / "local-installation.json").exists()


def test_archive_extracts_once_and_refuses_modified_stage(tmp_path: Path) -> None:
    path, checksum = release_zip(tmp_path)
    value = Release(path, checksum)
    destination = tmp_path / "stage"
    value.extract(destination)
    value.extract(destination)
    (destination / "uv.lock").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="staged release differs"):
        value.extract(destination)


def test_interrupted_member_is_not_published_and_can_be_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, checksum = release_zip(tmp_path)
    release = Release(path, checksum)
    read = ZipExtFile.read
    started = False

    def interrupt(stream: ZipExtFile, n: int = -1) -> bytes:
        nonlocal started
        if stream.name.endswith("/uv.lock"):
            if started:
                raise OSError("write interrupted")
            started = True
            return read(stream, 1)
        return read(stream, n)

    stage = tmp_path / "stage"
    with monkeypatch.context() as patch:
        patch.setattr(ZipExtFile, "read", interrupt)
        with pytest.raises(OSError, match="interrupted"):
            release.extract(stage)
    assert not (stage / "uv.lock").exists()
    release.extract(stage)
    assert archive.digest(stage / "uv.lock") == release.hashes["uv.lock"]


def test_killed_extractor_reclaims_its_draft_before_the_staging_space_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = installer(tmp_path, monkeypatch)
    scratch = value.root / "releases/.regimex-unpack"
    scratch.mkdir(mode=0o700, parents=True)
    neighbor = scratch / "operator-note"
    neighbor.write_bytes(b"preserve")
    child = """
import os, signal, sys
from pathlib import Path
from regimex.local_installation.archive import Release, extraction_scratch
from regimex.local_installation.install import installation_lock

archive, checksum, root, destination, scratch = sys.argv[1:]
release = Release(Path(archive), checksum)
with installation_lock(Path(root)), extraction_scratch(Path(scratch)) as draft:
    os.link = lambda *args: os.kill(os.getpid(), signal.SIGKILL)
    release.extract(Path(destination), draft=draft)
"""
    killed = subprocess.run(
        [
            sys.executable,
            "-c",
            child,
            str(value.release.path),
            value.release.sha256,
            str(value.root),
            str(value.source),
            str(scratch),
        ],
        timeout=15,
        check=False,
    )
    assert killed.returncode == -signal.SIGKILL
    draft = scratch / "member.partial"
    assert draft.stat().st_size > 0
    assert not (value.source / "BUILD.json").exists()

    def space_after_recovery(_root: Path) -> SimpleNamespace:
        assert not draft.exists()
        return SimpleNamespace(free=4 * 1024**3)

    monkeypatch.setattr(shutil, "disk_usage", space_after_recovery)
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    with install.installation_lock(value.root), pytest.raises(ValueError, match="Install uv"):
        value.stage()
    assert archive.digest(value.source / "uv.lock") == value.release.hashes["uv.lock"]
    assert neighbor.read_bytes() == b"preserve"
    assert not draft.exists()


def test_extraction_refuses_unexpected_draft_type(tmp_path: Path) -> None:
    path, checksum = release_zip(tmp_path)
    scratch = tmp_path / ".regimex-unpack"
    scratch.mkdir(mode=0o700)
    unrelated = tmp_path / "unrelated"
    unrelated.write_bytes(b"keep")
    (scratch / "member.partial").symlink_to(unrelated)
    with pytest.raises(ValueError, match="draft"):
        Release(path, checksum).extract(tmp_path / "stage")
    assert unrelated.read_bytes() == b"keep"
    assert (scratch / "member.partial").is_symlink()


def test_scratch_does_not_follow_releases_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = installer(tmp_path, monkeypatch)
    elsewhere = tmp_path / "other" / ".regimex-unpack"
    elsewhere.mkdir(parents=True, mode=0o700)
    draft = elsewhere / "member.partial"
    draft.write_bytes(b"leave alone")
    draft.chmod(0o600)
    (value.root / "releases").symlink_to(elsewhere.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="scratch parent"):
        value.stage()
    assert draft.read_bytes() == b"leave alone"


@pytest.mark.parametrize(
    "name,mode",
    [
        ("root/../outside", stat.S_IFREG),
        ("/absolute", stat.S_IFREG),
        ("root/link", stat.S_IFLNK),
        ("root\\escape", stat.S_IFREG),
    ],
)
def test_archive_rejects_traversal_and_links(tmp_path: Path, name: str, mode: int) -> None:
    path = tmp_path / "unsafe.zip"
    with ZipFile(path, "w") as output:
        member = ZipInfo(name)
        member.external_attr = (mode | 0o600) << 16
        output.writestr(member, b"payload")
    with pytest.raises(ValueError):
        Release(path, archive.digest(path))
    assert not (tmp_path / "outside").exists()


def test_archive_rejects_wrong_digest_preview_and_expansion_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, checksum = release_zip(tmp_path, preview=True)
    with pytest.raises(ValueError, match="checksum"):
        Release(path, "0" * 64)
    with pytest.raises(ValueError, match="preview"):
        Release(path, checksum)
    monkeypatch.setattr(archive, "MAX_EXPANDED_BYTES", 1)
    with pytest.raises(ValueError, match="expanded-size"):
        Release(path, checksum)


def test_archive_rejects_duplicate_and_unlisted_payload(tmp_path: Path) -> None:
    path, _ = release_zip(tmp_path)
    with ZipFile(path, "a") as output:
        member = ZipInfo("regimex-0.1.0/unlisted")
        member.external_attr = (stat.S_IFREG | 0o600) << 16
        output.writestr(member, b"extra")
    with pytest.raises(ValueError, match="every payload"):
        Release(path, archive.digest(path))
    # ZipFile warns when writing a duplicate; it is deliberate invalid input.
    with pytest.warns(UserWarning), ZipFile(path, "a") as output:
        output.writestr(member, b"duplicate")
    with pytest.raises(ValueError, match="duplicate"):
        Release(path, archive.digest(path))


def installer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Installer:
    path, checksum = release_zip(tmp_path)
    monkeypatch.setattr(install, "postgres_bin", lambda selected: Path("/opt/postgresql/16/bin"))
    value = Installer(
        Options(path, checksum, tmp_path / "installation", name="regimex-test", port=18000)
    )
    value.root.mkdir(mode=0o700)
    (value.root / "logs").mkdir(mode=0o700)
    return value


def test_pending_work_resumes_existing_api_without_stopping_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = installer(tmp_path, monkeypatch)
    commands = []
    monkeypatch.setattr(value, "control", lambda *command: commands.append(command))
    monkeypatch.setattr(
        value, "probe", lambda action: {"pending": 1, "doing": 0, "postgres_port": 55439}
    )
    with pytest.raises(ValueError, match="Update refused"):
        value.quiesce()
    assert commands == [("stop", "api"), ("start", "api")]
    assert value.state["phase"] == "intake_resumed"


def test_second_quiescence_check_keeps_workers_stopped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = installer(tmp_path, monkeypatch)
    commands = []
    status = iter(
        (
            {"pending": 0, "doing": 0, "postgres_port": 55439},
            {"pending": 1, "doing": 0, "postgres_port": 55439},
        )
    )
    monkeypatch.setattr(value, "control", lambda *command: commands.append(command))
    monkeypatch.setattr(value, "probe", lambda action: next(status))
    with pytest.raises(ValueError, match="Work appeared"):
        value.quiesce()
    assert not any(command[0] == "start" for command in commands)
    assert value.state["phase"] == "services_stopped"


def test_units_use_isolated_names_ordering_and_private_paths(tmp_path: Path) -> None:
    root = tmp_path / "runtime"
    units = systemd.units(
        root, root / "venv/bin/python", Path("/opt/pg/bin"), "regimex-test", 18000
    )
    api = units["regimex-test-api.service"]
    assert "Requires=regimex-test-postgres.service regimex-test-migrate.service" in api
    assert "--host 127.0.0.1 --port 18000" in api
    assert "UMask=0077" in api
    assert str(root / "venv/bin/python") in api
    assert "regimex-api.service" not in "".join(units.values())
    with pytest.raises(ValueError):
        systemd.units(root, root / "venv/bin/python", Path("/opt/pg/bin"), "bad\nname", 18000)


def test_environment_adoption_preserves_limits_and_never_reads_shell(tmp_path: Path) -> None:
    (tmp_path / "credentials").mkdir()
    install.write_private(tmp_path / "credentials/database.dsn", "private-fixture")
    expected = install.environment_file(tmp_path, 18000, existing=False)
    expected["REGIMEX_ANALYSIS_TIME_BUDGET_SECONDS"] = "300"
    install.write_private(
        tmp_path / "service.env", "".join(f"{k}={v}\n" for k, v in expected.items())
    )
    assert install.environment_file(tmp_path, 18000, existing=True) == expected
    with pytest.raises(ValueError, match="roots/origin"):
        install.environment_file(tmp_path, 8000, existing=True)


def managed(value: Installer) -> None:
    value.state = {
        "schema_version": 1,
        "postgres_port": value.options.postgres_port,
        "name": value.options.name,
        "port": value.options.port,
        "initialized": True,
        "phase": "ready",
        "archive_sha256": "b" * 64,
        "active_release": str(value.root / "releases/previous"),
    }
    install.write_private(value.state_path, json.dumps(value.state))
    (value.root / "credentials").mkdir(mode=0o700)
    install.write_private(value.root / "credentials/database.dsn", "deliberate-private-dsn")
    install.write_private(value.root / "credentials/user.token", "unchanged-private-token")
    (value.root / "objects").mkdir(mode=0o700)
    (value.root / "objects/retained").write_bytes(b"immutable-observation")
    env = install.environment_file(value.root, value.options.port, existing=False)
    install.write_private(value.root / "service.env", "".join(f"{k}={v}\n" for k, v in env.items()))


def test_migration_failure_keeps_services_stopped_and_preserves_existing_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = installer(tmp_path, monkeypatch)
    managed(value)
    monkeypatch.setattr(value, "validate_database_configuration", lambda: None)
    events: list[object] = []
    monkeypatch.setattr(systemd, "available", lambda: None)
    monkeypatch.setattr(systemd, "check_existing", lambda *args, **kwargs: None)
    monkeypatch.setattr(systemd, "write_units", lambda *args: events.append("units"))
    monkeypatch.setattr(value, "stage", lambda: events.append("staged"))
    monkeypatch.setattr(value, "control", lambda *args: events.append(args))
    monkeypatch.setattr(
        value, "probe", lambda action: {"pending": 0, "doing": 0, "postgres_port": 55439}
    )

    def fail_migration(command: list[str], **kwargs: object) -> str:
        if command[-1] == "daemon-reload":
            return ""
        assert command[-1] == "migrate"
        raise RuntimeError("deliberate migration failure")

    monkeypatch.setattr(value, "run", fail_migration)
    with pytest.raises(RuntimeError, match="migration failure"):
        value.install()
    assert events[0] == "staged"
    assert not any(isinstance(event, tuple) and event[0] == "start" for event in events)
    assert json.loads(value.state_path.read_text())["phase"] == "recovery_required"
    assert (value.root / "credentials/user.token").read_text() == "unchanged-private-token"
    assert (value.root / "credentials/database.dsn").read_text() == "deliberate-private-dsn"
    assert (value.root / "objects/retained").read_bytes() == b"immutable-observation"


@pytest.mark.parametrize("phase", ["ready", "verifying"])
def test_retry_health_check_of_active_release_keeps_it_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    from io import BytesIO
    from urllib.request import Request

    value = installer(tmp_path, monkeypatch)
    managed(value)
    monkeypatch.setattr(value, "validate_database_configuration", lambda: None)
    value.state["active_release"] = str(value.source.parent)
    value.state["phase"] = phase
    value.state["archive_sha256"] = value.release.sha256
    install.write_private(value.state_path, json.dumps(value.state))
    original = value.state_path.read_bytes()
    monkeypatch.setattr(systemd, "available", lambda: None)
    monkeypatch.setattr(systemd, "check_existing", lambda *args, **kwargs: None)
    monkeypatch.setattr(value, "stage", lambda: pytest.fail("must not sync the active venv"))
    monkeypatch.setattr(value, "control", lambda *args: pytest.fail("must not restart services"))
    maintenance_active = False

    def command(arguments: list[str], **kwargs: object) -> str:
        if arguments[-2:] == ["doctor", "--json"]:
            return '{"healthy":true}'
        assert arguments[:4] == ["systemctl", "--user", "is-active", "--quiet"]
        # Match systemctl's any-active semantics with only the maintenance worker stopped.
        if not maintenance_active and arguments[4:] == ["regimex-test-maintenance.service"]:
            raise RuntimeError("inactive")
        return ""

    class Response(BytesIO):
        status = 200

    class LocalAPI:
        def open(self, request: str | Request, timeout: int) -> Response:
            if isinstance(request, Request):
                return Response(b'{"state":"available"}')
            return Response(b'{"status":"ready","deployment_configured":true}')

    monkeypatch.setattr(value, "run", command)
    monkeypatch.setattr(install, "build_opener", lambda *args: LocalAPI())
    with monkeypatch.context() as clock:
        ticks = iter((0.0, 61.0))
        clock.setattr(time, "monotonic", lambda: next(ticks))
        with pytest.raises(RuntimeError, match="health/doctor checks failed"):
            value.install()
    assert value.state_path.read_bytes() == original

    maintenance_active = True
    value.install()
    assert json.loads(value.state_path.read_bytes())["phase"] == "ready"
    if phase == "ready":
        assert value.state_path.read_bytes() == original


def test_dependency_staging_failure_keeps_old_services_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = installer(tmp_path, monkeypatch)
    managed(value)
    monkeypatch.setattr(value, "validate_database_configuration", lambda: None)
    monkeypatch.setattr(systemd, "available", lambda: None)
    monkeypatch.setattr(systemd, "check_existing", lambda *args, **kwargs: None)
    monkeypatch.setattr(value, "control", lambda *args: pytest.fail("must not stop before staging"))

    def unavailable() -> None:
        raise RuntimeError("dependency download unavailable")

    monkeypatch.setattr(value, "stage", unavailable)
    with pytest.raises(RuntimeError, match="download unavailable"):
        value.install()
    assert (value.root / "credentials/user.token").read_text() == "unchanged-private-token"


def test_interrupted_verification_reuses_completed_candidate_without_uv_sync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = installer(tmp_path, monkeypatch)
    value.release.extract(value.source)
    value.python.parent.mkdir(parents=True)
    value.python.write_bytes(b"deliberate interpreter existence fixture")
    install.write_private(
        value.source.parent / "staged.json",
        json.dumps(
            {
                "archive_sha256": value.release.sha256,
                "wheel": value.release.wheel,
            }
        ),
    )
    value.state = {"phase": "verifying"}
    monkeypatch.setattr(
        value, "run", lambda *args, **kwargs: pytest.fail("must not modify a completed runtime")
    )
    value.stage()
    assert value.state["phase"] == "staged"


def test_old_requires_target_is_adopted_before_stopping_intake(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = installer(tmp_path, monkeypatch)
    managed(value)
    monkeypatch.setattr(value, "validate_database_configuration", lambda: None)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    old = systemd.units(
        value.root, value.root / "venv/bin/python", value.pg, value.options.name, value.options.port
    )
    target = systemd.unit_name(value.options.name, "target")
    old[target] = old[target].replace("Wants=", "Requires=")
    systemd.write_units(value.root, old)
    events: list[object] = []
    monkeypatch.setattr(systemd, "available", lambda: None)
    monkeypatch.setattr(value, "stage", lambda: events.append("staged"))
    monkeypatch.setattr(
        value, "probe", lambda action: {"pending": 1, "doing": 0, "postgres_port": 55439}
    )
    monkeypatch.setattr(value, "run", lambda *args, **kwargs: "")

    def control(action: str, *components: str) -> None:
        if (action, components) == ("stop", ("api",)):
            body = (systemd.user_unit_directory() / target).read_text()
            assert "Wants=" in body and "Requires=" not in body
            assert events == ["staged"]
        events.append((action, components))

    monkeypatch.setattr(value, "control", control)
    with pytest.raises(ValueError, match="Update refused"):
        value.install()
    assert events == ["staged", ("stop", ("api",)), ("start", ("api",))]
    assert json.loads(value.state_path.read_text())["phase"] == "intake_resumed"


def test_credentials_are_atomic_and_never_replace_an_existing_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from regimex.local_installation import probe

    path = tmp_path / "user.token"
    original = os.link

    def interrupted(source: Path, destination: Path) -> None:
        assert source.read_text() == "complete-token"
        assert not destination.exists()
        raise KeyboardInterrupt()

    with monkeypatch.context() as guard:
        guard.setattr(os, "link", interrupted)
        with pytest.raises(KeyboardInterrupt):
            probe.create_private(path, "complete-token")
    assert not path.exists()
    assert not tuple(tmp_path.iterdir())
    assert os.link is original
    probe.create_private(path, "complete-token")
    with pytest.raises(FileExistsError):
        probe.create_private(path, "replacement")
    assert path.read_text() == "complete-token"
    assert path.stat().st_mode & 0o777 == 0o600


def test_activation_intent_is_persisted_before_any_application_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = installer(tmp_path, monkeypatch)
    managed(value)
    monkeypatch.setattr(value, "validate_database_configuration", lambda: None)
    monkeypatch.setattr(systemd, "available", lambda: None)
    monkeypatch.setattr(systemd, "check_existing", lambda *args, **kwargs: None)
    monkeypatch.setattr(systemd, "write_units", lambda *args: None)
    monkeypatch.setattr(value, "stage", lambda: None)
    monkeypatch.setattr(value, "run", lambda *args, **kwargs: "")
    monkeypatch.setattr(
        value, "probe", lambda action: {"pending": 0, "doing": 0, "postgres_port": 55439}
    )
    started = []

    def control(action: str, *components: str) -> None:
        if action == "start":
            saved = json.loads(value.state_path.read_text())
            assert saved["initialized"] is True and saved["phase"] == "activating"
            started.append(components)
            raise KeyboardInterrupt()

    monkeypatch.setattr(value, "control", control)
    with pytest.raises(KeyboardInterrupt):
        value.install()
    assert started
    assert json.loads(value.state_path.read_text())["initialized"] is True


def test_adoption_requires_complete_owned_units_before_recording_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    with pytest.raises(ValueError, match="every existing unit"):
        systemd.check_existing(tmp_path, "wrong-name", adopt=True, require_complete=True)


def test_fresh_install_refuses_occupied_port_before_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from collections.abc import Iterator
    from contextlib import contextmanager

    value = installer(tmp_path, monkeypatch)
    monkeypatch.setattr(systemd, "available", lambda: None)
    monkeypatch.setattr(systemd, "check_existing", lambda *args, **kwargs: None)
    monkeypatch.setattr(value, "stage", lambda: pytest.fail("must not stage an occupied listener"))

    class Occupied:
        def bind(self, address: tuple[str, int]) -> None:
            raise OSError("deliberate occupied-port fixture")

    @contextmanager
    def occupied() -> Iterator[Occupied]:
        yield Occupied()

    monkeypatch.setattr(socket, "socket", occupied)
    with pytest.raises(ValueError, match="port is occupied"):
        value.install()
    assert not value.state_path.exists()


def test_verification_requires_own_units_and_authenticated_http(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from email.message import Message
    from urllib.error import HTTPError
    from urllib.request import Request

    value = installer(tmp_path, monkeypatch)
    managed(value)
    called = []

    class Response:
        status = 200

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def read(self, limit: int) -> bytes:
            return b'{"status":"ready","deployment_configured":true}'

    class ForeignAPI:
        def open(self, request: str | Request, timeout: int) -> Response:
            if isinstance(request, Request):
                assert request.get_header("Authorization") == "Bearer unchanged-private-token"
                called.append("authenticated")
                raise HTTPError(request.full_url, 401, "Unauthorized", Message(), None)
            return Response()

    def command(arguments: list[str], **kwargs: object) -> str:
        assert arguments[:4] == ["systemctl", "--user", "is-active", "--quiet"]
        assert len(arguments) == 5
        called.append(arguments[-1])
        return ""

    monkeypatch.setattr(value, "run", command)
    monkeypatch.setattr(install, "build_opener", lambda *args: ForeignAPI())
    clock = iter((0.0, 61.0))
    monkeypatch.setattr(time, "monotonic", lambda: next(clock))
    with pytest.raises(RuntimeError, match="health/doctor checks failed"):
        value.verify()
    assert called == [
        *(
            systemd.unit_name(value.options.name, part)
            for part in ("postgres", "api", *systemd.WORKERS)
        ),
        "authenticated",
    ]


def test_bootstrap_publishes_a_registered_model_after_interrupted_activation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from collections.abc import Iterator
    from contextlib import contextmanager
    from types import SimpleNamespace

    from pydantic import SecretStr

    from regimex import service_bootstrap
    from regimex.local_installation import probe
    from regimex.service_settings import ServiceSettings

    active = ["baseline"]
    published = {"baseline"}
    manifests = {}
    for architecture in ("baseline", "tcn"):
        path = tmp_path / "release/models" / architecture
        path.mkdir(parents=True)
        content = json.dumps({"model_version_id": architecture}).encode()
        (path / "manifest.json").write_bytes(content)
        manifests[architecture] = hashlib.sha256(content).hexdigest()
    (tmp_path / "credentials").mkdir()
    install.write_private(tmp_path / "credentials/admin.token", "deliberate-admin-token")

    class Repository:
        def get_model(self, identity: str) -> SimpleNamespace:
            return SimpleNamespace(bundle_sha256=manifests[identity])

        def get_active_model(self) -> SimpleNamespace:
            return SimpleNamespace(model_version_id=active[0])

        @contextmanager
        def connection(self) -> Iterator[Repository]:
            yield self

        def execute(self, query: str, parameters: tuple[str]) -> SimpleNamespace:
            return SimpleNamespace(fetchone=lambda: (1,) if parameters[0] in published else None)

    class Service:
        repository = Repository()

        def authenticate(self, token: str) -> str:
            return token

        def activate_model(self, principal: str, identity: str, *, expected_previous: str) -> None:
            assert expected_previous == active[0]
            active[0] = identity
            published.add(identity)

    settings = ServiceSettings(database_dsn=SecretStr("dbname=fixture"))
    monkeypatch.setattr(ServiceSettings, "from_env", classmethod(lambda cls: settings))
    monkeypatch.setattr(service_bootstrap, "create_service", lambda config: Service())
    probe.models(tmp_path, tmp_path)
    assert active == ["tcn"] and published == {"baseline", "tcn"}
    probe.models(tmp_path, tmp_path)
    assert active[0] == "tcn"
