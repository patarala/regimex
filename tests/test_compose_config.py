"""Check the storage and access paths after Docker Compose has merged the files."""

import json
import os
import runpy
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STORE_CLIENTS = ("api", "analysis-worker", "admin-worker", "maintenance-worker", "ops")


@pytest.fixture(scope="module")
def compose() -> list[str]:
    for command in (["docker", "compose"], ["docker-compose"]):
        if (
            shutil.which(command[0])
            and subprocess.run([*command, "version"], capture_output=True).returncode == 0
        ):
            return command
    pytest.skip("Docker Compose is required to resolve the deployment")


@pytest.mark.parametrize("s3", [False, True], ids=["local", "s3"])
def test_effective_compose_storage(compose: list[str], s3: bool) -> None:
    files = ["-f", "compose.yaml"]
    if s3:
        files.extend(["-f", "compose.s3.yaml"])
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("REGIMEX_", "COMPOSE_"))
    }
    env.update(
        REGIMEX_DATABASE_SCHEMA="lab",
        REGIMEX_HTTPS_PORT="12443",
        REGIMEX_HTTP_CONCURRENCY_LIMIT="6",
        REGIMEX_HTTP_CONNECTION_LIMIT="20",
        REGIMEX_S3_ENDPOINT_URL="https://store.example.org",
        REGIMEX_S3_BUCKET="experiments",
        REGIMEX_S3_PREFIX="test",
    )
    output = subprocess.check_output(
        [
            *compose,
            "--env-file",
            os.devnull,
            *files,
            "--profile",
            "*",
            "config",
            "--format",
            "json",
        ],
        cwd=ROOT,
        env=env,
    )
    services = json.loads(output)["services"]
    api = services["api"]
    assert api["environment"]["REGIMEX_PUBLIC_ORIGIN"] == "https://localhost:12443"
    assert api["environment"]["REGIMEX_HTTP_CONCURRENCY_LIMIT"] == "6"
    assert api["environment"]["REGIMEX_HTTP_CONNECTION_LIMIT"] == "20"
    assert str(api["ports"][0]["published"]) == "12443"
    migration = services["migrate"]
    assert migration["environment"]["REGIMEX_DATABASE_SCHEMA"] == "lab"
    assert {secret["source"] for secret in migration["secrets"]} == {"database_dsn"}
    assert not migration.get("volumes")
    assert set(migration["networks"]) == {"backend"}

    for name in STORE_CLIENTS:
        service = services[name]
        settings = service["environment"]
        assert settings["REGIMEX_DATABASE_SCHEMA"] == "lab"
        assert settings["REGIMEX_OBJECT_BACKEND"] == ("s3" if s3 else "local")
        mounts = {v["target"]: v["source"] for v in service["volumes"] if v["type"] == "volume"}
        secrets = {item["source"] for item in service["secrets"]}
        if s3:
            assert mounts[settings["REGIMEX_S3_LOCK_ROOT"]] == "s3-coordination"
            assert "objects" not in mounts.values()
            assert {"s3_credentials", "s3_ca"} <= secrets
        else:
            assert mounts["/data"] == "objects"
            assert {"s3_credentials", "s3_ca"}.isdisjoint(secrets)
        assert ("storage" in service["networks"]) is s3
        if name == "ops":
            assert mounts["/operator-output"] == "operator-output"
        else:
            assert "/operator-output" not in mounts


def test_update_builders_match_compose_image_tags(compose: list[str]) -> None:
    services = json.loads(
        subprocess.check_output(
            [*compose, "-f", "compose.yaml", "--profile", "service", "config", "--format", "json"],
            cwd=ROOT,
            env={k: v for k, v in os.environ.items() if not k.startswith(("REGIMEX_", "COMPOSE_"))},
        )
    )["services"]
    builders = runpy.run_path(str(ROOT / "deploy/update_components.py"))["BUILDERS"]
    assert set(builders) == {"api", "analysis-worker", "maintenance-worker", "admin-worker"}
    for role, builder in builders.items():
        assert services[builder]["build"]["target"] == ("api" if builder == "api" else "worker")
        assert services[role]["image"] == services[builder]["image"]
