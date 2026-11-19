"""The container smoke starts and cleans up its own deployment."""

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest
from support import compose_service_smoke


@pytest.mark.parametrize("build", [False, True])
def test_clean_engine_fetches_runtime_database_before_starting_services(
    monkeypatch: pytest.MonkeyPatch, build: bool
) -> None:
    available = {"application"} if not build else set()
    started: list[str] = []

    def engine(
        arguments: list[str], environment: dict[str, str], *, required: bool = True
    ) -> subprocess.CompletedProcess[bytes]:
        action = arguments[2]
        if action == "build":
            available.add("application")
        elif action == "pull":
            assert arguments[3:] == ["postgres"]
            available.add("postgres")
        elif action == "up":
            # A build-stage cache cannot supply an absent runtime image.
            assert available == {"application", "postgres"}
            assert arguments[arguments.index("--pull") + 1] == "never"
            assert "--no-build" in arguments
            assert "--wait" in arguments and "--wait-timeout" in arguments
            started.append(arguments[-1])
        return subprocess.CompletedProcess(arguments, 0, b"", b"")

    monkeypatch.setattr(compose_service_smoke, "command", engine)
    compose_service_smoke.start_services(["docker", "compose"], {}, build=build)
    assert started == ["postgres", "maintenance-worker"]


def test_failed_database_pull_prevents_service_start(monkeypatch: pytest.MonkeyPatch) -> None:
    started = False

    def unavailable_registry(
        arguments: list[str], environment: dict[str, str], *, required: bool = True
    ) -> subprocess.CompletedProcess[bytes]:
        nonlocal started
        if arguments[2] == "pull":
            raise RuntimeError("registry unavailable")
        if arguments[2] == "up":
            started = True
        return subprocess.CompletedProcess(arguments, 0, b"", b"")

    monkeypatch.setattr(compose_service_smoke, "command", unavailable_registry)
    with pytest.raises(RuntimeError, match="registry unavailable"):
        compose_service_smoke.start_services(["docker", "compose"], {}, build=False)
    assert not started


def test_bootstrap_with_existing_schema_is_rejected_before_migrations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def engine(
        arguments: list[str], environment: dict[str, str], *, required: bool = True
    ) -> subprocess.CompletedProcess[bytes]:
        assert "migrate" not in arguments
        inventory = b"5; 2615 2200 SCHEMA - public regimex\n" if "exec" in arguments else b""
        return subprocess.CompletedProcess(arguments, 0, inventory, b"")

    monkeypatch.setattr(compose_service_smoke, "command", engine)
    with pytest.raises(RuntimeError, match="before migration"):
        compose_service_smoke.start_services(["docker", "compose"], {}, build=False)


def test_smoke_environment_isolates_images_and_storage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REGIMEX_IMAGE_TAG", "production")
    monkeypatch.setenv("REGIMEX_S3_PREFIX", "test-bucket/base")
    monkeypatch.setenv("REGIMEX_S3_BUCKET", "test-bucket")
    monkeypatch.setenv("REGIMEX_ANALYSIS_REPLICAS", "32")
    monkeypatch.setenv("COMPOSE_PROFILES", "ops")
    monkeypatch.setenv("DOCKER_CONTEXT", "ci")

    environment = compose_service_smoke.smoke_environment("mine", build=True, s3=True)
    assert environment["REGIMEX_IMAGE_TAG"] == "mine"
    assert environment["REGIMEX_S3_PREFIX"] == "test-bucket/base/mine"
    assert environment["REGIMEX_S3_BUCKET"] == "test-bucket"
    assert environment["DOCKER_CONTEXT"] == "ci"
    assert "COMPOSE_PROFILES" not in environment
    assert "REGIMEX_ANALYSIS_REPLICAS" not in environment

    local = compose_service_smoke.smoke_environment("mine", build=False, s3=False)
    assert local["REGIMEX_IMAGE_TAG"] == "production"
    assert "REGIMEX_S3_BUCKET" not in local


def test_partial_build_removes_only_its_own_tags(monkeypatch: pytest.MonkeyPatch) -> None:
    removed = []

    def engine(
        arguments: list[str], environment: dict[str, str], *, required: bool = True
    ) -> subprocess.CompletedProcess[bytes]:
        if arguments[2] == "ls":
            return subprocess.CompletedProcess(
                arguments, 0, b"regimex-api:mine\nregimex-api:production\npostgres:16\n", b""
            )
        assert arguments[1:3] == ["image", "rm"]
        removed.extend(arguments[3:])
        return subprocess.CompletedProcess(arguments, 0, b"", b"")

    monkeypatch.setattr(compose_service_smoke, "command", engine)
    compose_service_smoke.remove_smoke_images("docker", "mine", {})
    assert removed == ["regimex-api:mine"]


def test_mounted_secrets_work_with_private_host_umask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if shutil.which("openssl") is None:
        pytest.skip("openssl is required to prepare the test certificates")
    mounts = tmp_path / "secrets"
    credentials = tmp_path / "provider.credentials"
    credentials.write_text("private fixture")
    provider_ca = tmp_path / "provider.crt"
    provider_ca.write_text("public fixture")
    ownership: dict[str, int] = {}

    def record_owner(path: Path, uid: int) -> None:
        ownership[path.name] = uid

    monkeypatch.setattr(compose_service_smoke, "owner", record_owner)
    previous = os.umask(0o077)
    try:
        ca = compose_service_smoke.prepare_secrets(
            mounts, s3_credentials=credentials, s3_ca=provider_ca
        )
    finally:
        os.umask(previous)

    assert ca.read_bytes() == (mounts / "api.crt").read_bytes()
    assert (mounts / "s3-ca.crt").read_text() == provider_ca.read_text()
    for name in ("api.crt", "api-ca.crt", "s3-ca.crt"):
        assert stat.S_IMODE((mounts / name).stat().st_mode) == 0o444
    assert ownership == {
        "s3-credentials": 10001,
        "database-password": 999,
        "application-password": 999,
        "database-dsn": 10001,
        "api.key": 10001,
    }
    for name in ownership:
        assert stat.S_IMODE((mounts / name).stat().st_mode) == 0o600


@pytest.mark.parametrize("timeout", [False, True])
def test_command_failure_keeps_credentials_and_paths_out_of_diagnostics(
    monkeypatch: pytest.MonkeyPatch, timeout: bool
) -> None:
    def fail(arguments: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        if timeout:
            raise subprocess.TimeoutExpired(arguments, 1800, stderr=b"private output")
        return subprocess.CompletedProcess(arguments, 1, b"private token", b"private output")

    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(RuntimeError) as caught:
        compose_service_smoke.command(
            ["docker", "compose", "run", "ops", "cat", "/private/credentials"],
            {"AWS_SECRET_ACCESS_KEY": "private fixture"},
        )
    assert str(caught.value) == "Compose command category=run service=ops " + (
        "timed_out" if timeout else "exit_code=1"
    )
