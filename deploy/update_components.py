"""Update selected processes in an existing Compose deployment."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from collections import Counter
from collections.abc import Sequence

# The API reads analysis reports. Analysis workers read models published by admin workers.
BUILDERS = {
    "api": "api",
    "maintenance-worker": "api",
    "analysis-worker": "analysis-worker",
    "admin-worker": "analysis-worker",
}
COMPOSE = ["docker", "compose", "--profile", "service"]


def run(*args: str) -> None:
    print(shlex.join([*COMPOSE, *args]), flush=True)
    subprocess.run([*COMPOSE, *args], check=True)


def replicas(roles: list[str], *, updated: Sequence[str] = ()) -> Counter[str]:
    ids = subprocess.check_output([*COMPOSE, "ps", "--all", "--quiet", *roles], text=True).split()
    counts: Counter[str] = Counter()
    if ids:
        fields = (
            '{{index .Config.Labels "com.docker.compose.service"}} '
            '{{index .Config.Labels "com.docker.compose.oneoff"}} {{.State.Status}} '
            "{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}"
        )
        states = subprocess.check_output(["docker", "inspect", "--format", fields, *ids], text=True)
        for state in states.splitlines():
            role, oneoff, status, health = state.split()
            if oneoff.lower() == "true":
                continue
            if role in updated and (status != "running" or health != "healthy"):
                raise ValueError(
                    f"Updated {role} is no longer healthy. Check the deployment first."
                )
            if status not in {"running", "restarting"}:
                raise ValueError(f"{role} has a {status} container. Check the deployment first.")
            counts[role] += 1
    for role in roles:
        if not counts[role]:
            raise ValueError(f"No running {role} in this Compose project.")
    return counts


def update(roles: list[str]) -> None:
    selected = [role for role in BUILDERS if role in roles]
    counts = replicas(selected)
    builders = dict.fromkeys(BUILDERS[role] for role in selected)
    run("build", *builders)
    for role in selected:
        run("run", "--rm", "--no-deps", "-T", role, "service", "health", "--component", "database")
    for index, role in enumerate(selected):
        if replicas(selected, updated=selected[:index]) != counts:
            raise ValueError(
                "Replica counts changed during the update. Repeat with the current scale."
            )
        run(
            "up",
            "--no-deps",
            "--no-build",
            "--pull",
            "never",
            "--wait",
            "--wait-timeout",
            "180",
            "--scale",
            f"{role}={counts[role]}",
            role,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roles", nargs="+", choices=BUILDERS, help="Service roles to update")
    try:
        update(parser.parse_args().roles)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        sys.exit(f"Component update stopped: {exc}")
