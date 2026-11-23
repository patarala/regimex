"""Exercise the component-update entry point without a Docker daemon."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "deploy/update_components.py"
ROLES = ["api", "analysis-worker", "admin-worker", "maintenance-worker"]


def rollout(
    tmp_path: Path, roles: list[str], *, states: str, fail: str = "", next_states: str = ""
) -> tuple[subprocess.CompletedProcess[str], list[list[str]]]:
    docker = tmp_path / "docker"
    docker.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "args = sys.argv[1:]\n"
        "assert os.environ['COMPOSE_FILE'] == 'compose.yaml:compose.s3.yaml'\n"
        "assert os.environ['COMPOSE_PROJECT_NAME'] == 'lab'\n"
        "if args[0] == 'inspect':\n"
        "    prior = open('calls.jsonl').read()\n"
        "    changed = os.environ['NEXT_STATES'] if '\"inspect\"' in prior else ''\n"
        "    print(changed or os.environ['STATES'])\n"
        "else:\n"
        "    assert args[:3] == ['compose', '--profile', 'service']\n"
        "    args = args[3:]\n"
        "    if args[0] == 'ps': print('existing-container')\n"
        "with open('calls.jsonl', 'a') as out: print(json.dumps(args), file=out)\n"
        "if ' '.join(args).startswith(os.environ['FAIL'] or 'unused'):\n"
        "    sys.exit('compose failed')\n"
    )
    docker.chmod(0o700)
    env = dict(
        os.environ,
        PATH=f"{tmp_path}:{os.environ['PATH']}",
        COMPOSE_FILE="compose.yaml:compose.s3.yaml",
        COMPOSE_PROJECT_NAME="lab",
        STATES=states,
        NEXT_STATES=next_states,
        FAIL=fail,
    )
    result = subprocess.run(
        [sys.executable, str(SCRIPT), *roles], cwd=tmp_path, env=env, capture_output=True, text=True
    )
    log = tmp_path / "calls.jsonl"
    return result, [
        json.loads(line) for line in log.read_text().splitlines()
    ] if log.exists() else []


def test_update_checks_new_images_then_replaces_requested_roles_in_reader_order(
    tmp_path: Path,
) -> None:
    states = "\n".join(
        [
            *(f"{role} False running" for role in ROLES),
            "analysis-worker False running",
            "api True exited",
        ]
    )
    result, calls = rollout(tmp_path, [*reversed(ROLES), "api"], states=states)
    assert result.returncode == 0, result.stderr
    changes = [args for args in calls if args[0] not in {"ps", "inspect"}]
    ordered = ["api", "maintenance-worker", "admin-worker", "analysis-worker"]
    assert changes[0] == ["build", "api", "analysis-worker"]
    assert changes[1:5] == [
        ["run", "--rm", "--no-deps", "-T", role, "service", "health", "--component", "database"]
        for role in ordered
    ]
    assert [args[-1] for args in changes[5:]] == ordered
    for args in changes[5:]:
        assert args[:7] == [
            "up",
            "--no-deps",
            "--no-build",
            "--pull",
            "never",
            "--wait",
            "--wait-timeout",
        ]
        assert args[-3:-1] == ["--scale", f"{args[-1]}={2 if args[-1] == 'analysis-worker' else 1}"]


@pytest.mark.parametrize("fail", ["build", "run --rm"])
def test_failed_build_or_schema_probe_keeps_running_services(tmp_path: Path, fail: str) -> None:
    result, calls = rollout(
        tmp_path, ["analysis-worker"], states="analysis-worker False running", fail=fail
    )
    assert result.returncode and "compose failed" in result.stderr
    assert ["build", "analysis-worker"] in calls
    assert all(args[0] != "up" for args in calls)


def test_failed_api_update_stops_before_writer(tmp_path: Path) -> None:
    result, calls = rollout(
        tmp_path,
        ["analysis-worker", "api"],
        states="api False running\nanalysis-worker False running",
        fail="up",
    )
    assert result.returncode and "compose failed" in result.stderr
    assert [args[-1] for args in calls if args[0] == "up"] == ["api"]


@pytest.mark.parametrize("states", ["api False exited", "api True running"])
def test_offline_role_is_not_started(tmp_path: Path, states: str) -> None:
    result, calls = rollout(tmp_path, ["api"], states=states)
    assert result.returncode
    assert [args[0] for args in calls] == ["ps", "inspect"]


def test_stateful_services_and_one_offs_cannot_be_selected(tmp_path: Path) -> None:
    result, calls = rollout(tmp_path, ["postgres", "migrate"], states="")
    assert result.returncode and "invalid choice" in result.stderr
    assert calls == []


def test_concurrent_resize_stops_before_replacement(tmp_path: Path) -> None:
    worker = "analysis-worker False running"
    result, calls = rollout(
        tmp_path, ["analysis-worker"], states=worker, next_states=f"{worker}\n{worker}"
    )
    assert result.returncode and "Replica counts changed" in result.stderr
    assert all(args[0] != "up" for args in calls)


@pytest.mark.parametrize(
    ("role", "builder"), [("admin-worker", "analysis-worker"), ("maintenance-worker", "api")]
)
def test_shared_image_update_only_applies_to_the_selected_role(
    tmp_path: Path, role: str, builder: str
) -> None:
    result, calls = rollout(tmp_path, [role], states=f"{role} False running")
    assert result.returncode == 0, result.stderr
    assert [args for args in calls if args[0] == "build"] == [["build", builder]]
    updates = [args for args in calls if args[0] == "up"]
    assert len(updates) == 1 and updates[0][-1] == role
    assert "--force-recreate" not in updates[0]
