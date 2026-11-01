"""The release smoke can start with no cached PostgreSQL runtime image."""

import subprocess

import pytest
from support import compose_service_smoke


@pytest.mark.parametrize("build", [False, True])
def test_clean_engine_fetches_runtime_database_before_starting_services(
    monkeypatch: pytest.MonkeyPatch, build: bool
) -> None:
    available = {"application"} if not build else set()
    started = False

    def engine(
        arguments: list[str], environment: dict[str, str], *, required: bool = True
    ) -> subprocess.CompletedProcess[bytes]:
        nonlocal started
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
            started = True
        return subprocess.CompletedProcess(arguments, 0, b"", b"")

    monkeypatch.setattr(compose_service_smoke, "command", engine)
    compose_service_smoke.start_services(["docker", "compose"], {}, build=build)
    assert started


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
