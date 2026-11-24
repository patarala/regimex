"""Run a fresh loopback service in a delegated cgroup v2 for each comparison."""

import argparse
import hashlib
import json
import os
import re
import secrets
import selectors
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import tomllib
from collections import Counter
from collections.abc import Callable
from contextlib import ExitStack, contextmanager
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen
from uuid import UUID, uuid4

import psycopg

from regimex.data_catalog import content_hash
from regimex.model_registry import InferenceRuntime, read_inference_runtime
from regimex.storage import write_json

ROOT = Path(__file__).resolve().parents[1]
CLI = [sys.executable, "-m", "regimex"]
SEEDS = (42, 43, 44, 45)


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def clean_env():
    # A throwaway run must never inherit an installed service's store or DSN.
    return {
        "PATH": os.environ["PATH"],
        "HOME": os.environ["HOME"],
        "LANG": "C.UTF-8",
        "PYTHONPATH": str(ROOT / "src"),
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }


def once(command, env, log, timeout=120):
    with log.open("ab") as output:
        result = subprocess.run(command, env=env, stdout=output, stderr=output, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"{Path(command[0]).name} exited {result.returncode}. See {log}")


def stop(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(8)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(5)


def launch(stack, command, env, log):
    with log.open("ab") as output:
        process = subprocess.Popen(command, env=env, stdout=output, stderr=output)
    stack.callback(stop, process)
    return process


def wait_ready(processes, probe: Callable[[], bool], component):
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        if any(process.poll() is not None for process in processes):
            raise RuntimeError(f"{component} exited during startup. See the cell logs")
        try:
            if probe():
                return
        except OSError, psycopg.OperationalError:
            pass
        time.sleep(0.2)
    raise RuntimeError(f"{component} did not become ready within 45 seconds")


def port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def bin_path(name, directory=None):
    location = shutil.which(str(Path(directory) / name) if directory else name)
    if location is None:
        raise ValueError(f"Missing {name}. Check PATH or REGIMEX_PG_BINDIR for PostgreSQL.")
    return str(Path(location).absolute())


def inference_policy(path):
    runtime = InferenceRuntime() if path is None else read_inference_runtime(path)
    resolved = runtime.model_dump(mode="json")
    return {
        "inference_runtime": resolved,
        "inference_policy_sha256": content_hash(resolved),
        "runtime_override": path is not None,
    }


def versions(config):
    env = clean_env()
    recorded = {
        "regimex": tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"],
        "python": sys.version.split()[0],
        "k6": subprocess.check_output(
            [config["k6"], "version"], env=env, text=True, timeout=10
        ).strip(),
    }
    for name in ("postgres", "initdb"):
        output = subprocess.check_output(
            [config[name], "--version"], env=env, text=True, timeout=10
        ).strip()
        if not re.fullmatch(rf"{name} \(PostgreSQL\) 16\.\d+(?: \([^\r\n]+\))?", output):
            raise ValueError(f"PostgreSQL 16 is required; found {output!r} for {name}")
        recorded[name] = output
    if (ROOT / ".git").exists() and shutil.which("git"):
        recorded["git_commit"] = subprocess.check_output(
            ["git", "-C", ROOT, "rev-parse", "HEAD"], env=env, text=True, timeout=10
        ).strip()
        recorded["git_dirty"] = bool(
            subprocess.check_output(
                ["git", "-C", ROOT, "status", "--porcelain"], env=env, text=True, timeout=10
            )
        )
    return recorded


@contextmanager
def finish_with(action):
    """Keep the load failure and attach any secondary diagnostic or cleanup failure."""
    try:
        yield
    except BaseException as failure:
        try:
            action()
        except Exception as secondary:
            failure.add_note(f"Additional failure while finishing the phase: {secondary}")
        raise
    else:
        action()


CGROUP_ROOT = Path("/sys/fs/cgroup")


def delegated_parent(value):
    if not value:
        raise ValueError(
            "Pass --cgroup-parent or set REGIMEX_LOAD_CGROUP_PARENT to a delegated cgroup v2 "
            "directory with cpu and memory enabled for child groups."
        )
    parent = Path(value).resolve(strict=True)
    if not parent.is_relative_to(CGROUP_ROOT) or not (parent / "cgroup.controllers").is_file():
        raise ValueError(f"{parent} must be a cgroup v2 directory under {CGROUP_ROOT}")
    enabled = (parent / "cgroup.subtree_control").read_text().split()
    if not {"cpu", "memory"}.issubset(enabled) or not os.access(parent, os.W_OK):
        raise ValueError(
            f"{parent} needs write access and cpu and memory in cgroup.subtree_control"
        )
    return parent


def owned_directory(path, identity):
    try:
        current = path.lstat()
    except FileNotFoundError:
        return False
    if (current.st_dev, current.st_ino) != identity:
        raise RuntimeError(f"The directory at {path} changed. Inspect it before cleanup.")
    return True


def identity(stat):
    return stat.st_dev, stat.st_ino


def remove_group(path, fd):
    if owned_directory(path, identity(os.fstat(fd))):
        path.rmdir()


@contextmanager
def new_group(parent):
    path = parent / f"regimex-load-{uuid4().hex}"
    path.mkdir()
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        # Keep attach and kill bound to this group if an operator renames the directory.
        with finish_with(lambda: remove_group(path, fd)):
            yield Path(f"/proc/self/fd/{fd}"), fd
    finally:
        os.close(fd)


def stop_group(group):
    (group / "cgroup.kill").write_text("1")
    deadline = time.monotonic() + 10
    while "populated 0" not in (group / "cgroup.events").read_text().splitlines():
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Processes remain in {group} after cgroup.kill")
        time.sleep(0.05)


def cleanup_cell(group, scratch, scratch_id):
    # Both resources may already have been released by the other supervisor.
    if scratch.exists() or (group / "cgroup.kill").exists():
        try:
            stop_group(group)
        except (OSError, RuntimeError) as exc:
            raise RuntimeError(
                f"Preserved private data at {scratch}. Verify {group.resolve()} is empty first."
            ) from exc
    if owned_directory(scratch, scratch_id):
        # Cached model directories become read-only after validation.
        for _, _, _, fd in os.fwalk(scratch, follow_symlinks=False):
            os.fchmod(fd, 0o700)
        shutil.rmtree(scratch)


@contextmanager
def private_scratch(out, group):
    scratch = Path(tempfile.mkdtemp(prefix="runtime-", dir=out))
    scratch_id = identity(scratch.lstat())
    with finish_with(lambda: cleanup_cell(group, scratch, scratch_id)):
        yield scratch


def phase(config, cell, env, name):
    summary = cell / f"{name}.json"
    events = cell / f"{name}-points.jsonl"
    console = cell / f"{name}-console.log"
    env = {
        **env,
        "REGIMEX_LOAD_PHASE": name,
        "REGIMEX_LOAD_CLIENTS": "1" if name == "warmup" else "4",
        "REGIMEX_LOAD_JOBS": "1" if name == "warmup" else "4",
        "REGIMEX_LOAD_RATE_PER_MINUTE": str(config["rate_per_minute"]),
        "REGIMEX_LOAD_DURATION_SECONDS": str(config["duration_seconds"]),
        "REGIMEX_LOAD_MAX_VUS": str(config["max_vus"]),
    }

    def save_accepted():
        ids = set()
        if console.is_file():
            with (cell / "jobs.jsonl").open("a") as jobs:
                for line in console.read_text().splitlines():
                    if line.startswith("REGIMEX_ACCEPTED "):
                        record = json.loads(line.removeprefix("REGIMEX_ACCEPTED "))
                        if record["phase"] != name:
                            raise ValueError("k6 acceptance marker has the wrong phase")
                        identity = UUID(record["analysis_id"])
                        if identity in ids:
                            raise ValueError("k6 reused an analysis ID within a phase")
                        ids.add(identity)
                        jobs.write(json.dumps(record) + "\n")
        if summary.exists():
            accepted = json.loads(summary.read_bytes())["metrics"]["analyses_accepted"]["count"]
            if len(ids) != accepted:
                raise ValueError(f"Missing acceptance records for {name}. See {console}")

    with finish_with(save_accepted):
        once(
            [
                config["k6"],
                "run",
                "--quiet",
                "--no-usage-report",
                "--log-format=raw",
                "--console-output",
                console,
                "--summary-export",
                summary,
                "--out",
                f"json={events}",
                ROOT / "scripts/load_analysis.js",
            ],
            env,
            cell / f"{name}-k6.log",
            timeout=(config["duration_seconds"] if name == "sustained" else 300) + 300,
        )


def summarize(cell, name, peak_bytes, arrival_seconds):
    metrics = json.loads((cell / f"{name}.json").read_bytes())["metrics"]
    starts, finishes, accounted = [], [], []
    with (cell / f"{name}-points.jsonl").open() as stream:
        for line in stream:
            point = json.loads(line)
            if point["type"] != "Point" or point["data"]["value"] <= 0:
                continue
            if point["metric"] == "analyses_started":
                starts.append(datetime.fromisoformat(point["data"]["time"]))
            elif point["metric"] == "analyses_verified":
                finishes.append(datetime.fromisoformat(point["data"]["time"]))
            elif point["metric"] == "analyses_accounted":
                accounted.append(datetime.fromisoformat(point["data"]["time"]))
    completed = metrics["analyses_verified"]["count"]
    if not starts or not finishes or len(finishes) != completed:
        raise ValueError("k6 completion samples do not match the summary")
    seconds = (max(finishes) - min(starts)).total_seconds()
    throughput_window = seconds
    if name == "sustained":
        if len(accounted) != metrics["analyses_scheduled"]["count"]:
            raise ValueError("Missing sustained completion or quota-refusal samples")
        throughput_window = max(
            arrival_seconds, (max(accounted) - min(starts)).total_seconds(), seconds
        )
    return {
        **outcomes(metrics, name),
        "status": "passed",
        "observed_seconds": seconds,
        "throughput_window_seconds": throughput_window,
        "completed_per_minute": completed * 60 / throughput_window,
        "cgroup_peak_bytes": peak_bytes,
    }


def outcomes(metrics, name):
    counters = {
        key: metrics.get(metric, {}).get("count")
        for key, metric in {
            "requested": "analyses_scheduled",
            "started": "analyses_started",
            "accepted": "analyses_accepted",
            "completed": "analyses_verified",
            "quota_refused": "analyses_quota_refused",
            "failed": "analyses_failed",
            "submit_failed": "analyses_submit_failed",
            "accounted": "analyses_accounted",
            "dropped": "dropped_iterations",
        }.items()
    }
    queue = {}
    if name == "sustained":
        for half in ("first", "second"):
            trend = metrics.get(f"analysis_queued_{half}_half", {})
            queue[f"queued_{half}_half"] = {
                "med": trend.get("med"),
                "max": trend.get("max"),
                "samples": metrics.get(f"analysis_queue_{half}_half_samples", {}).get("count"),
            }
        queue["active_after_drain"] = metrics.get("analysis_active_after_drain", {}).get("max")
        queue["after_drain_samples"] = metrics.get("analysis_after_drain_samples", {}).get("count")
    result = metrics.get("analysis_to_result", {}) if counters["completed"] else {}
    intake = metrics.get("analysis_intake", {}) if counters["accepted"] else {}
    # k6's legacy Rate export counts true samples (unexpected HTTP responses) as passes.
    return {
        **counters,
        **queue,
        "http_failures": metrics.get("http_req_failed", {}).get("passes"),
        "result_median_ms": result.get("med"),
        "result_p95_ms": result.get("p(95)"),
        "intake_median_ms": intake.get("med"),
    }


def run_cell(config_file, scratch, group, steps, workers, name):
    config = json.loads(config_file.read_bytes())
    out = config_file.parent
    cell = out / f"{name}-{steps}-{workers}w"
    cell.mkdir()
    os.sched_setaffinity(0, config["cpu_ids"])
    quota, period = (group / "cpu.max").read_text().split()
    if (
        str(os.getpid()) not in (group / "cgroup.procs").read_text().splitlines()
        or quota == "max"
        or int(quota) != len(config["cpu_ids"]) * int(period)
        or set(config["cpu_ids"]) != os.sched_getaffinity(0)
        or int((group / "memory.max").read_text()) != config["memory_bytes"]
        or (group / "memory.swap.max").read_text().strip() != "0"
    ):
        raise RuntimeError("The requested cgroup CPU, memory or swap limit is not active")
    envelope = {
        "cgroup": "/" + str(group.resolve().relative_to(CGROUP_ROOT)),
        "cpu_ids": sorted(os.sched_getaffinity(0)),
        "cpu_max": [int(quota), int(period)],
        "memory_max": config["memory_bytes"],
        "swap_max": 0,
    }
    save(cell / "envelope.json", envelope)

    env = clean_env()
    pg_port = port()
    password = secrets.token_urlsafe(32)
    password_file = scratch / "pg-password"
    password_file.write_text(password)
    pgdata = scratch / "pg"
    once(
        [
            config["initdb"],
            "-D",
            pgdata,
            "-U",
            "loaduser",
            "-A",
            "scram-sha-256",
            "--pwfile",
            password_file,
            "--no-locale",
            "--encoding=UTF8",
        ],
        env,
        cell / "postgres.log",
    )
    dsn = f"postgresql://loaduser:{password}@127.0.0.1:{pg_port}/postgres"
    with ExitStack() as stack:
        pg = launch(
            stack,
            [config["postgres"], "-D", pgdata, "-h", "127.0.0.1", "-p", str(pg_port), "-k", ""],
            env,
            cell / "postgres.log",
        )

        def pg_ready():
            with psycopg.connect(dsn, connect_timeout=1):
                return True

        wait_ready([pg], pg_ready, "PostgreSQL")
        api_port = port()
        origin = f"http://127.0.0.1:{api_port}"
        env.update(
            REGIMEX_DATABASE_DSN=dsn,
            REGIMEX_OBJECT_ROOT=str(scratch / "objects"),
            REGIMEX_WORK_ROOT=str(scratch / "api-work"),
            REGIMEX_PUBLIC_ORIGIN=origin,
            REGIMEX_ALLOW_LOOPBACK_HTTP="true",
        )
        service = [*CLI, "service"]
        setup_log = cell / "setup.log"
        once([*service, "migrate"], env, setup_log)
        accounts = [("admin", "admin"), ("user", "user")]
        if name.startswith("journey-"):
            accounts.append(("second-user", "user"))
        for account, role in accounts:
            once(
                [
                    *service,
                    "token-create",
                    "--owner",
                    account,
                    "--role",
                    role,
                    "--output",
                    scratch / account,
                ],
                env,
                setup_log,
            )
        policy_options = []
        if config["runtime_override"]:
            snapshot = scratch / "inference-runtime.json"
            write_json(snapshot, config["inference_runtime"])
            policy_options = ["--runtime-file", snapshot]
        once(
            [
                *service,
                "register-model",
                *policy_options,
                "--bundle",
                config["bundle"],
                "--manifest-sha256",
                config["manifest_sha256"],
                "--token-file",
                scratch / "admin",
                "--activate",
            ],
            env,
            setup_log,
        )
        api_env = {**env, "REGIMEX_HTTP_CONCURRENCY_LIMIT": str(config["http_concurrency_limit"])}
        services = [
            pg,
            launch(stack, [*service, "serve", "--port", str(api_port)], api_env, cell / "api.log"),
        ]
        logs = []
        for index in range(1, workers + 1):
            log = cell / f"worker-{index}.log"
            logs.append(log)
            worker_env = {**env, "REGIMEX_WORK_ROOT": str(scratch / f"worker-{index}")}
            services.append(
                launch(stack, [*service, "worker", "--queue", "analysis"], worker_env, log)
            )

        def ready():
            with urlopen(origin + "/health/models", timeout=2) as response:
                if not json.load(response)["deployment_configured"]:
                    return False
            with psycopg.connect(dsn, connect_timeout=2) as connection:
                return (
                    connection.execute(
                        "SELECT count(*) FROM regimex_worker_queues WHERE queue_name='analysis'"
                    ).fetchone()[0]
                    == workers
                )

        wait_ready(services, ready, "API and analysis workers")
        env.update(
            REGIMEX_LOAD_TOKEN=(scratch / "user").read_text().strip(),
            REGIMEX_LOAD_PORT=str(api_port),
            REGIMEX_LOAD_MODEL=config["model"],
            REGIMEX_LOAD_EXPECTED_BUNDLE_SHA256=config["manifest_sha256"],
            REGIMEX_LOAD_EXPECTED_POLICY_SHA256=config["inference_policy_sha256"],
            REGIMEX_LOAD_DIR=str(out / "inputs" / str(steps)),
        )

        def collect_stages(jobs):
            once(
                [
                    sys.executable,
                    ROOT / "scripts/load_timings.py",
                    "--jobs",
                    jobs,
                    *(arg for log in logs for arg in ("--worker-log", log)),
                    "--output",
                    cell / "timings.json",
                ],
                env,
                cell / "timings.log",
            )

        if name.startswith("journey-"):
            from load_journeys import run_journey

            env["REGIMEX_LOAD_SECOND_TOKEN"] = (scratch / "second-user").read_text().strip()
            env["REGIMEX_LOAD_BURST_PER_OWNER"] = str(config["burst_per_owner"])
            try:
                outcome = run_journey(name.removeprefix("journey-"), cell, env, config["k6"])
            finally:
                envelope["peak_bytes"] = int((group / "memory.peak").read_text())
                save(cell / "envelope.json", envelope)
            collect_stages(cell / "verified.jsonl")
            phases = json.loads((cell / "timings.json").read_bytes())["phases"]
            save(
                cell / "result.json",
                {
                    **outcome,
                    "status": "passed",
                    "phase": name,
                    "steps": "mixed"
                    if name in ("journey-contention", "journey-cancel", "journey-uploads")
                    else 256,
                    "workers": workers,
                    "median_stage_ms": {},
                    "stage_groups": {key: value["median_ms"] for key, value in phases.items()},
                    "cgroup_peak_bytes": envelope["peak_bytes"],
                },
            )
            return

        def collect():
            envelope["peak_bytes"] = int((group / "memory.peak").read_text())
            save(cell / "envelope.json", envelope)
            jobs = cell / "jobs.jsonl"
            if not jobs.is_file() or not jobs.stat().st_size:
                return
            collect_stages(jobs)

        with finish_with(collect):
            phase(config, cell, env, "warmup")
            phase(config, cell, env, name)
            row = {
                "phase": name,
                "steps": steps,
                "workers": workers,
                **summarize(
                    cell, name, int((group / "memory.peak").read_text()), config["duration_seconds"]
                ),
            }
        timings = json.loads((cell / "timings.json").read_bytes())
        row["median_stage_ms"] = timings["phases"][name]["median_ms"]
        save(cell / "result.json", row)


def failed_cell(cell, planned_phase, steps, workers):
    unreadable = []

    def read_artifact(name):
        try:
            data = json.loads((cell / name).read_bytes())
            if isinstance(data, dict):
                return data
        except FileNotFoundError:
            return {}
        except OSError, ValueError:
            pass
        unreadable.append(name)
        return {}

    measured = read_artifact("result.json")
    if measured:
        return {**measured, "status": "failed", "failure_stage": "after measurement"}

    # A warm-up summary cannot describe a measured phase that started but never exported.
    phase = "setup"
    for candidate in (planned_phase, "warmup"):
        if (cell / f"{candidate}-k6.log").exists() or (cell / f"{candidate}.json").exists():
            phase = candidate
            break
    metrics = read_artifact(f"{phase}.json").get("metrics", {})
    row = {
        "phase": phase,
        "planned_phase": planned_phase,
        "steps": steps,
        "workers": workers,
        "status": "failed",
        **outcomes(metrics, phase),
    }
    row["cgroup_peak_bytes"] = read_artifact("envelope.json").get("peak_bytes")
    phases = read_artifact("timings.json").get("phases", {})
    row["median_stage_ms"] = phases.get(phase, {}).get("median_ms", {})
    points = cell / f"{phase}-points.jsonl"
    if planned_phase.startswith("journey-") and (cell / "k6.log").is_file():
        from load_journeys import journey_summary

        row.update(journey_summary(read_artifact("summary.json").get("metrics", {})))
        row.update(phase=planned_phase, failure_stage="journey")
        points = cell / "points.jsonl"
    if unreadable:
        row["unreadable_artifacts"] = unreadable
    if points.exists():
        failures = Counter()
        with points.open("rb") as stream:
            for line in stream:
                try:
                    point = json.loads(line)
                except json.JSONDecodeError, UnicodeDecodeError:
                    continue
                if (
                    point.get("type") == "Point"
                    and point.get("metric") == "http_req_failed"
                    and point["data"]["value"] == 1
                ):
                    tags = point["data"]["tags"]
                    failures[
                        (tags.get("name", "unknown request"), tags.get("status", "unknown status"))
                    ] += 1
        row["http_failure_samples"] = [
            {"request": request, "status": status, "count": count}
            for (request, status), count in sorted(failures.items())
        ]
    return row


@contextmanager
def record_cell(out, rows, total, name, steps, workers):
    cell = out / f"{name}-{steps}-{workers}w"
    try:
        yield
        row = json.loads((cell / "result.json").read_bytes())
    except (Exception, KeyboardInterrupt, SystemExit) as failure:
        try:
            row = failed_cell(cell, name, steps, workers)
            if cell.is_dir():
                save(cell / "result.json", row)
            report(out, [*rows, row], total)
        except Exception as diagnostic:
            failure.add_note(f"Could not summarize the failed cell: {diagnostic}")
        raise
    rows.append(row)
    report(out, rows, total)


def display(value, divisor=1, digits=None):
    if value is None:
        return "unavailable"
    return str(value) if digits is None else f"{value / divisor:.{digits}f}"


def report(out, rows, total):
    save(out / "comparison.json", rows)
    status = "Complete" if len(rows) == total else "Partial"
    if any(row.get("status") == "failed" for row in rows):
        status = "Failed"
    lines = [
        f"# {status} load comparison ({len(rows)} of {total} cells reported)",
        "",
        "| Phase | Status | Steps | Workers | Completed / requested | Jobs/min | Median result s | "
        "p95 result s | Peak MiB | HTTP failures |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['phase']} | {row.get('status', 'passed')} | "
            f"{row['steps']} | {row['workers']} | "
            f"{display(row.get('completed'))} / {display(row.get('requested'))} | "
            f"{display(row.get('completed_per_minute'), 1, 2)} | "
            f"{display(row.get('result_median_ms'), 1000, 2)} | "
            f"{display(row.get('result_p95_ms'), 1000, 2)} | "
            f"{display(row.get('cgroup_peak_bytes'), 2**20, 1)} | "
            f"{display(row.get('http_failures'))} |"
        )
    lines.extend(
        [
            "",
            "Matrix jobs/min uses the first measured submission through the last verified "
            "download. Sustained jobs/min includes the full arrival window and any further time "
            "through the final verification or quota refusal. "
            "Peak memory is the fresh cgroup's kernel high-water mark from bootstrap through that "
            "phase, including PostgreSQL, API, worker trees, k6 and charged file cache.",
        ]
    )
    lines.extend(
        [
            "",
            "## Median stage durations in milliseconds",
            "",
            "| Phase | Steps | Workers | Queue wait | Worker occupancy | Child process | "
            "Model load | Prediction | Backend predict | HTML report |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        fields = (
            "queue_wait",
            "worker_occupancy",
            "child_process",
            "model_load",
            "prediction",
            "backend_predict",
            "report",
        )
        groups = row.get("stage_groups", {row["phase"]: row["median_stage_ms"]})
        for phase_name, stages in groups.items():
            values = [display(stages.get(k), digits=1) for k in fields]
            lines.append(
                f"| {phase_name} | {row['steps']} | {row['workers']} | " + " | ".join(values) + " |"
            )
    for row in rows:
        if row.get("status") == "failed":
            fields = (
                "requested",
                "started",
                "accepted",
                "completed",
                "quota_refused",
                "submit_failed",
                "failed",
                "dropped",
                "http_failures",
            )
            lines.extend(
                [
                    "",
                    f"## Failed {row['phase']} cell, {row['steps']} steps "
                    f"and {row['workers']} workers",
                    "",
                    "| Requested | Started | Accepted | Verified | Quota refused | "
                    "Failed submissions | Failed iterations | Dropped | HTTP failures |",
                    "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
                    "| " + " | ".join(display(row.get(field)) for field in fields) + " |",
                    "",
                ]
            )
            for item in row.get("http_failure_samples", []):
                lines.append(f"- {item['request']}: HTTP {item['status']} (count {item['count']})")
            if row.get("unreadable_artifacts"):
                lines.extend(["", "Could not read " + ", ".join(row["unreadable_artifacts"]) + "."])
        if row["phase"] == "sustained" and row.get("status") != "failed":
            counts = [row[f"queued_{half}_half"].get("samples") for half in ("first", "second")]
            coverage = " and ".join("unavailable" if n is None else str(n) for n in counts)
            lines.extend(
                [
                    "",
                    f"Sustained intake started {row['started']} arrivals, accepted "
                    f"{row['accepted']}, refused {row['quota_refused']} at quota, "
                    f"failed {row['failed']} and dropped {row['dropped']}. "
                    f"Queued jobs peaked at {row['queued_first_half']['max']} in the "
                    f"first half and {row['queued_second_half']['max']} in the second. "
                    f"Completed samples by half: {coverage}. "
                    f"Active after drain: {row['active_after_drain']}.",
                ]
            )
    (out / "comparison.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def supervise_cell(config_file, scratch, scratch_id, group_name, fd, steps, workers, name):
    """Stay outside the measured process tree so either supervisor can release its resources."""
    config = json.loads(config_file.read_bytes())
    runtime = (config["duration_seconds"] if name == "sustained" else 300) + 600
    group = Path(f"/proc/self/fd/{fd}")
    with (
        finish_with(lambda: remove_group(group_name, fd)),
        finish_with(lambda: cleanup_cell(group, scratch, scratch_id)),
    ):
        # The shell joins first so the interpreter and all its children inherit the limit.
        command = [
            "/bin/sh",
            "-ec",
            'if ! echo 0 > "$1/cgroup.procs"; then '
            'echo "Run the driver inside the delegation with migration permission" >&2; '
            'exit 1; fi; shift; exec "$@"',
            "regimex-load",
            str(group),
            sys.executable,
            str(Path(__file__).resolve()),
            "--cell",
            str(config_file),
            str(scratch),
            str(group),
            steps,
            workers,
            name,
        ]
        with subprocess.Popen(
            command, cwd=ROOT, env=clean_env(), umask=0o077, pass_fds=(fd,)
        ) as cell:
            try:
                wait_for_cell(cell, runtime)
            finally:
                cell.kill()


def wait_for_cell(cell, timeout):
    # The driver owns the write end of stdin. Its death also wakes the supervisor.
    with selectors.DefaultSelector() as selector:
        selector.register(sys.stdin, selectors.EVENT_READ, "driver")
        with os.fdopen(os.pidfd_open(cell.pid), "rb") as pidfd:
            selector.register(pidfd, selectors.EVENT_READ, "cell")
            ready = selector.select(timeout)
            if not ready:
                raise subprocess.TimeoutExpired(cell.args, timeout)
            if any(key.data == "driver" for key, _ in ready):
                raise RuntimeError("Load driver disappeared during the cell")
            code = cell.wait()
            if code:
                raise subprocess.CalledProcessError(code, cell.args)


def start_supervisor(config_file, config, steps, workers, name):
    # The driver retains independent handles in case the supervisor itself dies.
    with new_group(Path(config["cgroup_parent"])) as (group, fd):
        (group / "cpu.max").write_text(f"{len(config['cpu_ids']) * 100000} 100000")
        (group / "memory.max").write_text(str(config["memory_bytes"]))
        (group / "memory.swap.max").write_text("0")
        (group / "memory.peak").read_text()
        with (group / "cgroup.kill").open("w"):
            pass
        with private_scratch(config_file.parent, group) as scratch:
            dev, ino = identity(scratch.lstat())
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--supervise",
                str(config_file),
                str(scratch),
                str(dev),
                str(ino),
                str(group.resolve()),
                str(fd),
                str(steps),
                str(workers),
                name,
            ]
            with subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                pass_fds=(fd,),
                start_new_session=True,
            ) as supervisor:
                try:
                    code = supervisor.wait()
                finally:
                    supervisor.stdin.close()
                    supervisor.wait()
                if code:
                    raise subprocess.CalledProcessError(code, command)


def main():
    from load_journeys import JOURNEYS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=("standard", "matrix", "sustained", "journeys", *JOURNEYS, "all"),
        default="standard",
        help="standard runs matrix and sustained. Choose a named journey, journeys or all for more",
    )
    parser.add_argument("--bundle", type=Path, default=ROOT / "release/models/tcn")
    parser.add_argument("--runtime-file", type=Path, help="inference runtime JSON for all cells")
    parser.add_argument("--output", type=Path, help="new directory (default var/load-test/run-*)")
    parser.add_argument(
        "--http-concurrency-limit", type=int, default=8, help="maximum active API requests"
    )
    parser.add_argument(
        "--cgroup-parent",
        default=os.environ.get("REGIMEX_LOAD_CGROUP_PARENT"),
        help="delegated cgroup v2 directory with cpu and memory enabled for children",
    )
    parser.add_argument("--cpus", type=int, default=2)
    parser.add_argument("--memory-mib", type=int, default=3072)
    parser.add_argument("--rate-per-minute", type=int, default=6)
    parser.add_argument("--duration-seconds", type=int, default=120)
    parser.add_argument("--max-vus", type=int, default=16)
    parser.add_argument("--burst-per-owner", type=int, choices=range(2, 13), default=4)
    options = parser.parse_args()
    if options.http_concurrency_limit < 2:
        parser.error("--http-concurrency-limit must be at least 2")
    if (
        min(
            options.cpus,
            options.memory_mib,
            options.rate_per_minute,
            options.duration_seconds,
            options.max_vus,
        )
        < 1
    ):
        parser.error("resource limits and arrival parameters must be positive")
    if (
        options.duration_seconds < 2
        or options.rate_per_minute > 120
        or options.duration_seconds > 600
        or options.max_vus > 32
        or options.rate_per_minute * options.duration_seconds > 6000
    ):
        parser.error("sustained limit: 120 arrivals/min, 2-600 seconds, 32 VUs and 100 arrivals")
    if options.rate_per_minute * options.duration_seconds % 60:
        parser.error("rate * duration / 60 must be a whole number of arrivals")
    try:
        parent_group = delegated_parent(options.cgroup_parent)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    cpus = sorted(os.sched_getaffinity(0))
    if options.cpus > len(cpus):
        parser.error(f"only {len(cpus)} CPUs are available")
    bundle = options.bundle.resolve()
    manifest = bundle / "manifest.json"
    if not manifest.is_file():
        parser.error("supply --bundle pointing to an unpacked Regimex model directory")
    if not (ROOT / "scripts/load_timings.py").is_file():
        parser.error("the disposable stand requires the scripts/load_timings.py stage collector")
    pgdir = os.environ.get("REGIMEX_PG_BINDIR")
    if not pgdir and not shutil.which("initdb") and shutil.which("pg_config"):
        pgdir = subprocess.check_output(["pg_config", "--bindir"], text=True).strip()
    config = {
        **inference_policy(options.runtime_file),
        "http_concurrency_limit": options.http_concurrency_limit,
        "cpu_ids": cpus[: options.cpus],
        "cgroup_parent": str(parent_group),
        "memory_bytes": options.memory_mib * 2**20,
        "bundle": str(bundle),
        "model": json.loads(manifest.read_bytes())["model_version_id"],
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "rate_per_minute": options.rate_per_minute,
        "duration_seconds": options.duration_seconds,
        "max_vus": options.max_vus,
        "burst_per_owner": options.burst_per_owner,
        "profile": options.profile,
        "k6": bin_path("k6"),
        "initdb": bin_path("initdb", pgdir),
        "postgres": bin_path("postgres", pgdir),
    }
    config["versions"] = versions(config)
    if options.output:
        out = options.output.resolve()
        out.mkdir(mode=0o700, parents=True)
    else:
        parent = ROOT / "var/load-test"
        parent.mkdir(parents=True, exist_ok=True)
        out = Path(tempfile.mkdtemp(prefix="run-", dir=parent))
    print(f"Load artifacts: {out}", flush=True)
    cells = []
    if options.profile in ("all", "standard", "matrix"):
        cells.extend(("parallel", steps, workers) for steps in (512, 4096) for workers in (1, 2))
    if options.profile in ("all", "standard", "sustained"):
        cells.append(("sustained", 512, 2))
    if options.profile in ("all", "journeys"):
        cells.extend((f"journey-{name}", 256, 2) for name in JOURNEYS)
    elif options.profile in JOURNEYS:
        cells.append((f"journey-{options.profile}", 256, 2))
    config_file = out / "run.json"
    config["planned_cells"] = [f"{name}-{steps}-{workers}w" for name, steps, workers in cells]
    rows = []
    report(out, rows, len(cells))
    config["inputs"] = {}
    for steps in sorted({steps for phase, steps, _ in cells if not phase.startswith("journey-")}):
        inputs = out / "inputs" / str(steps)
        inputs.mkdir(parents=True)
        for seed in SEEDS:
            market = inputs / f"market-{seed}.json"
            once(
                [
                    *CLI,
                    "generate",
                    "--seed",
                    str(seed),
                    "--steps",
                    str(steps),
                    "--instruments",
                    "2",
                    "--output",
                    market,
                ],
                clean_env(),
                out / "generate.log",
            )
            config["inputs"][str(market.relative_to(out))] = hashlib.sha256(
                market.read_bytes()
            ).hexdigest()
    save(config_file, config)
    for name, steps, workers in cells:
        print(f"Running {name}, {steps} steps, {workers} worker(s)", flush=True)
        with record_cell(out, rows, len(cells), name, steps, workers):
            start_supervisor(config_file, config, steps, workers, name)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--cell"]:
        run_cell(
            Path(sys.argv[2]),
            Path(sys.argv[3]),
            Path(sys.argv[4]),
            int(sys.argv[5]),
            int(sys.argv[6]),
            sys.argv[7],
        )
    elif sys.argv[1:2] == ["--supervise"]:
        supervise_cell(
            Path(sys.argv[2]),
            Path(sys.argv[3]),
            (int(sys.argv[4]), int(sys.argv[5])),
            Path(sys.argv[6]),
            int(sys.argv[7]),
            *sys.argv[8:],
        )
    else:
        signal.signal(signal.SIGTERM, lambda number, _frame: sys.exit(128 + number))
        main()
