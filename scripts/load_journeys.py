"""Run bounded two-owner k6 scenarios against a dedicated test installation."""

import argparse
import json
import os
import secrets
import shutil
import subprocess
import tempfile
from pathlib import Path
from uuid import UUID

from regimex.storage import write_json

ROOT = Path(__file__).resolve().parents[1]
JOURNEYS = ("contention", "burst", "replay", "browse", "cancel", "uploads")


def journey_summary(metrics):
    counts = {
        "requested": "journey_jobs_requested",
        "accepted": "journey_jobs_accepted",
        "completed": "journey_results_verified",
        "cancelled": "journey_jobs_cancelled",
        "capacity_refusals": "journey_capacity_refusals",
        "verified_iterations": "journeys_verified",
    }
    result = {key: metrics.get(name, {}).get("count") for key, name in counts.items()}
    times = metrics.get("journey_submit_to_result_ms", {}) if result["completed"] else {}
    return {
        **result,
        "result_median_ms": times.get("med"),
        "result_p95_ms": times.get("p(95)"),
        "http_failures": metrics.get("http_req_failed", {}).get("passes"),
    }


def collect_journey_evidence(out: Path, *, interrupted: bool = False):
    seen = set()
    console = out / "console.log"
    if console.is_file():
        with console.open("rb") as lines, (out / "verified.jsonl").open("x") as stream:
            for line in lines:
                if not line.startswith(b"REGIMEX_VERIFIED "):
                    continue
                # A killed k6 process may have written only part of its final record.
                if interrupted and not line.endswith(b"\n"):
                    break
                job = json.loads(line.removeprefix(b"REGIMEX_VERIFIED "))
                identity = UUID(job["analysis_id"])
                if identity in seen:
                    raise ValueError("Duplicate verified analysis in k6 output")
                seen.add(identity)
                stream.write(json.dumps(job) + "\n")
    summary = out / "summary.json"
    metrics = json.loads(summary.read_bytes())["metrics"] if summary.exists() else {}
    results = journey_summary(metrics)
    write_json(out / "journey.json", results)
    return len(seen), results


def run_journey(
    profile: str, out: Path, env: dict[str, str], k6: str
) -> dict[str, int | float | None]:
    from regimex.market_export import sample_snapshot

    inputs = out / "inputs"
    inputs.mkdir()
    required = {
        "contention": ("anchor", "short", "long"),
        "burst": ("anchor",),
        "replay": ("anchor", "short"),
        "browse": ("anchor",),
        "cancel": ("anchor", "long"),
        "uploads": ("anchor", "short", "long"),
    }[profile]
    seeds = {}
    unique_seeds = secrets.SystemRandom().sample(range(2**32), len(required))
    for kind, seed in zip(required, unique_seeds, strict=True):
        steps = 4096 if kind == "long" else 256
        instruments = 2 if kind == "long" else 1
        market = sample_snapshot(seed=seed, steps=steps, instruments=instruments)
        write_json(inputs / f"{kind}.json", market.model_dump(mode="json"))
        seeds[kind] = {
            "seed": seed,
            "steps": steps,
            "instruments": instruments,
            "content_sha256": market.content_sha256,
        }
    write_json(out / "inputs.json", seeds)
    limits = {**env, "REGIMEX_LOAD_JOURNEY": profile, "REGIMEX_LOAD_DIR": str(inputs.resolve())}
    try:
        with (out / "k6.log").open("wb") as log:
            result = subprocess.run(
                [
                    k6,
                    "run",
                    "--no-usage-report",
                    "--quiet",
                    "--log-format=raw",
                    "--console-output",
                    str(out / "console.log"),
                    "--summary-export",
                    str(out / "summary.json"),
                    "--out",
                    f"json={out / 'points.jsonl'}",
                    str(ROOT / "scripts/load_journeys.js"),
                ],
                env=limits,
                stdout=log,
                stderr=log,
                timeout=1000,
                check=False,
            )
        if result.returncode:
            raise RuntimeError(f"k6 {profile} exited {result.returncode}. See {out / 'k6.log'}")
    except (subprocess.TimeoutExpired, RuntimeError) as failure:
        try:
            collect_journey_evidence(out, interrupted=True)
        except Exception as error:
            failure.add_note(f"Could not finish collecting k6 evidence: {error}")
        raise
    verified, results = collect_journey_evidence(out)
    completed, cancelled = results["completed"], results["cancelled"]
    accepted, refusals = results["accepted"], results["capacity_refusals"]
    if (
        verified != completed
        or accepted != completed + cancelled
        or results["requested"] != accepted + refusals
    ):
        raise ValueError("Submitted jobs do not match verified, cancelled and refused outcomes")
    print(f"{profile}: {completed} verified, {cancelled} cancelled, {refusals} refused", flush=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=("all", *JOURNEYS), nargs="?", default="all")
    args = parser.parse_args()
    env = dict(os.environ)
    if not env.get("REGIMEX_LOAD_MODEL") or env["REGIMEX_LOAD_MODEL"] == "active":
        parser.error("Set REGIMEX_LOAD_MODEL to an immutable published version")
    for option in ("REGIMEX_LOAD_TOKEN_FILE", "REGIMEX_LOAD_SECOND_TOKEN_FILE"):
        if not env.get(option):
            parser.error(f"Set {option} to the dedicated test owner's token file")
        env[option.removesuffix("_FILE")] = Path(env[option]).read_text().strip()
    if not env["REGIMEX_LOAD_TOKEN"] or not env["REGIMEX_LOAD_SECOND_TOKEN"]:
        parser.error("Both test credentials must be nonempty")
    if env["REGIMEX_LOAD_TOKEN"] == env["REGIMEX_LOAD_SECOND_TOKEN"]:
        parser.error("Use two distinct test owners")
    k6 = shutil.which("k6")
    if not k6:
        parser.error("Install k6 and add it to PATH")
    parent = ROOT / "var/load-test"
    parent.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix="journeys-", dir=parent))
    print(f"Inputs, k6 summaries and samples: {out}", flush=True)
    for name in JOURNEYS if args.profile == "all" else (args.profile,):
        cell = out / name
        cell.mkdir()
        run_journey(name, cell, env, k6)


if __name__ == "__main__":
    main()
