"""Run a fresh loopback service in a resource-limited systemd unit per comparison."""

import argparse
import hashlib
import json
import os
import re
import secrets
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


def stop_unit(unit):
    stop = subprocess.run(
        ["systemctl", "--user", "stop", unit], capture_output=True, text=True, timeout=20
    )
    if stop.returncode == 0:
        return
    # Transient units are normally collected before the runner's wait returns.
    status = subprocess.run(
        ["systemctl", "--user", "show", unit, "-p", "LoadState", "-p", "ActiveState"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if status.returncode == 0 and {"LoadState=not-found", "ActiveState=inactive"} <= set(
        status.stdout.splitlines()
    ):
        return
    raise RuntimeError(f"Cannot verify {unit} has stopped: {stop.stderr.strip()}")


@contextmanager
def private_scratch(out, unit):
    # Keep ambiguous shutdowns for inspection; explicit cleanup handles read-only model caches.
    temporary = tempfile.TemporaryDirectory(prefix="runtime-", dir=out, delete=False)
    scratch = Path(temporary.name)
    created = scratch.lstat()

    def cleanup():
        try:
            stop_unit(unit)
        except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
            raise RuntimeError(
                f"Preserved private data at {scratch}. Verify {unit} has stopped before removal."
            ) from exc
        if not os.path.samestat(created, scratch.lstat()):
            raise RuntimeError(f"The private scratch path changed. Inspect {scratch}")
        temporary.cleanup()

    with finish_with(cleanup):
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


def run_cell(config_file, scratch, steps, workers, name):
    config = json.loads(config_file.read_bytes())
    out = config_file.parent
    cell = out / f"{name}-{steps}-{workers}w"
    cell.mkdir()
    os.sched_setaffinity(0, config["cpu_ids"])
    cgroup_id = Path("/proc/self/cgroup").read_text().split("0::")[1].strip()
    group = Path("/sys/fs/cgroup") / cgroup_id.lstrip("/")
    quota, period = (group / "cpu.max").read_text().split()
    if (
        quota == "max"
        or int(quota) != len(config["cpu_ids"]) * int(period)
        or set(config["cpu_ids"]) != os.sched_getaffinity(0)
        or int((group / "memory.max").read_text()) != config["memory_bytes"]
        or (group / "memory.swap.max").read_text().strip() != "0"
    ):
        raise RuntimeError("The requested cgroup CPU, memory or swap limit is not active")
    envelope = {
        "cgroup": cgroup_id,
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
        for role in ("admin", "user"):
            once(
                [
                    *service,
                    "token-create",
                    "--owner",
                    role,
                    "--role",
                    role,
                    "--output",
                    scratch / role,
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

        def collect():
            envelope["peak_bytes"] = int((group / "memory.peak").read_text())
            save(cell / "envelope.json", envelope)
            jobs = cell / "jobs.jsonl"
            if not jobs.is_file() or not jobs.stat().st_size:
                return
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
    # A warm-up summary cannot describe a measured phase that started but never exported.
    phase = "setup"
    for candidate in (planned_phase, "warmup"):
        if (cell / f"{candidate}-k6.log").exists() or (cell / f"{candidate}.json").exists():
            phase = candidate
            break
    summary = cell / f"{phase}.json"
    metrics = json.loads(summary.read_bytes())["metrics"] if summary.exists() else {}
    row = {
        "phase": phase,
        "planned_phase": planned_phase,
        "steps": steps,
        "workers": workers,
        "status": "failed",
        **outcomes(metrics, phase),
    }
    envelope = cell / "envelope.json"
    row["cgroup_peak_bytes"] = (
        json.loads(envelope.read_bytes()).get("peak_bytes") if envelope.exists() else None
    )
    timings = cell / "timings.json"
    row["median_stage_ms"] = {}
    if timings.exists():
        row["median_stage_ms"] = (
            json.loads(timings.read_bytes())["phases"].get(phase, {}).get("median_ms", {})
        )
    points = cell / f"{phase}-points.jsonl"
    if points.exists():
        failures = Counter()
        with points.open() as stream:
            for line in stream:
                try:
                    point = json.loads(line)
                except json.JSONDecodeError:
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
    except Exception as failure:
        try:
            result = cell / "result.json"
            if result.is_file():
                row = json.loads(result.read_bytes())
                row.update(status="failed", failure_stage="after measurement")
            else:
                row = failed_cell(cell, name, steps, workers)
            if cell.is_dir():
                save(cell / "result.json", row)
            report(out, [*rows, row], total)
        except Exception as diagnostic:
            failure.add_note(f"Could not summarize the failed cell: {diagnostic}")
        raise
    row = json.loads((cell / "result.json").read_bytes())
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
            "Peak memory is the fresh unit's kernel high-water mark from bootstrap through that "
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
        stages = row["median_stage_ms"]
        fields = (
            "queue_wait",
            "worker_occupancy",
            "child_process",
            "model_load",
            "prediction",
            "backend_predict",
            "report",
        )
        values = ["unavailable" if stages.get(k) is None else f"{stages[k]:.1f}" for k in fields]
        lines.append(
            f"| {row['phase']} | {row['steps']} | {row['workers']} | " + " | ".join(values) + " |"
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("matrix", "sustained", "all"), default="all")
    parser.add_argument("--bundle", type=Path, default=ROOT / "release/models/tcn")
    parser.add_argument("--runtime-file", type=Path, help="inference runtime JSON for all cells")
    parser.add_argument("--output", type=Path, help="new directory (default var/load-test/run-*)")
    parser.add_argument(
        "--http-concurrency-limit", type=int, default=8, help="API connection/task limit"
    )
    parser.add_argument("--cpus", type=int, default=2)
    parser.add_argument("--memory-mib", type=int, default=3072)
    parser.add_argument("--rate-per-minute", type=int, default=6)
    parser.add_argument("--duration-seconds", type=int, default=120)
    parser.add_argument("--max-vus", type=int, default=16)
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
        "memory_bytes": options.memory_mib * 2**20,
        "bundle": str(bundle),
        "model": json.loads(manifest.read_bytes())["model_version_id"],
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "rate_per_minute": options.rate_per_minute,
        "duration_seconds": options.duration_seconds,
        "max_vus": options.max_vus,
        "profile": options.profile,
        "k6": bin_path("k6"),
        "initdb": bin_path("initdb", pgdir),
        "postgres": bin_path("postgres", pgdir),
    }
    config["versions"] = versions(config)
    runner = bin_path("systemd-run")
    if options.output:
        out = options.output.resolve()
        out.mkdir(mode=0o700, parents=True)
    else:
        parent = ROOT / "var/load-test"
        parent.mkdir(parents=True, exist_ok=True)
        out = Path(tempfile.mkdtemp(prefix="run-", dir=parent))
    print(f"Load artifacts: {out}", flush=True)
    cells = []
    if options.profile in ("all", "matrix"):
        cells.extend(("parallel", steps, workers) for steps in (512, 4096) for workers in (1, 2))
    if options.profile in ("all", "sustained"):
        cells.append(("sustained", 512, 2))
    config_file = out / "run.json"
    config["planned_cells"] = [f"{name}-{steps}-{workers}w" for name, steps, workers in cells]
    rows = []
    report(out, rows, len(cells))
    config["inputs"] = {}
    for steps in sorted({steps for _, steps, _ in cells}):
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
        unit = f"regimex-load-{uuid4().hex}.service"
        runtime = (options.duration_seconds if name == "sustained" else 300) + 600
        with (
            record_cell(out, rows, len(cells), name, steps, workers),
            private_scratch(out, unit) as tmp,
        ):
            command = [
                runner,
                "--user",
                "--unit",
                unit,
                "--collect",
                "--quiet",
                "--wait",
                "--pipe",
                f"--working-directory={ROOT}",
                "-p",
                f"MemoryMax={config['memory_bytes']}",
                "-p",
                "MemorySwapMax=0",
                "-p",
                f"CPUQuota={options.cpus * 100}%",
                "-p",
                f"RuntimeMaxSec={runtime}",
                "-p",
                "TimeoutStopSec=10",
                "-p",
                "UMask=0077",
                sys.executable,
                str(Path(__file__).resolve()),
                "--cell",
                str(config_file),
                str(tmp),
                str(steps),
                str(workers),
                name,
            ]
            subprocess.run(command, check=True, timeout=runtime + 30)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--cell"]:
        run_cell(
            Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
        )
    else:
        signal.signal(signal.SIGTERM, lambda number, _frame: sys.exit(128 + number))
        main()
