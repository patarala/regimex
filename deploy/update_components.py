"""Update selected processes in an existing Compose deployment."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from collections import Counter

# Readers must be ready before an analysis worker can publish a newer report.
BUILDERS = {
    "api": "api",
    "maintenance-worker": "api",
    "admin-worker": "analysis-worker",
    "analysis-worker": "analysis-worker",
}
COMPOSE = ["docker", "compose", "--profile", "service"]


def run(*args: str) -> None:
    print(shlex.join([*COMPOSE, *args]), flush=True)
    subprocess.run([*COMPOSE, *args], check=True)


def replicas(roles: list[str]) -> Counter[str]:
    ids = subprocess.check_output([*COMPOSE, "ps", "--all", "--quiet", *roles], text=True).split()
    counts: Counter[str] = Counter()
    if ids:
        fields = (
            '{{index .Config.Labels "com.docker.compose.service"}} '
            '{{index .Config.Labels "com.docker.compose.oneoff"}} {{.State.Status}}'
        )
        states = subprocess.check_output(["docker", "inspect", "--format", fields, *ids], text=True)
        for state in states.splitlines():
            role, oneoff, status = state.split()
            if oneoff.lower() == "true":
                continue
            if status != "running":
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
    if replicas(selected) != counts:
        raise ValueError("Replica counts changed during the build. Repeat with the current scale.")
    for role in selected:
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
    parser.add_argument("roles", nargs="+", choices=BUILDERS, help="Running services to replace")
    try:
        update(parser.parse_args().roles)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        sys.exit(f"Component update stopped: {exc}")
