"""Reconcile k6 acceptance markers with the jobs recorded for this load run."""

import argparse
import json
from pathlib import Path
from uuid import UUID


def recorded_phases(path: Path) -> dict[UUID, str]:
    phases: dict[UUID, str] = {}
    if path.is_file():
        with path.open() as lines:
            for line in lines:
                job = json.loads(line)
                identity = UUID(job["analysis_id"])
                if identity in phases:
                    raise ValueError(f"jobs.jsonl contains repeated analysis ID {identity}")
                phases[identity] = job["phase"]
    return phases


def collect_acceptances(directory: Path, phase: str) -> None:
    console = directory / f"{phase}-console.log"
    jobs_path = directory / "jobs.jsonl"
    previous = recorded_phases(jobs_path)
    seen: set[UUID] = set()
    recorded = 0
    marker_error = None
    if console.is_file():
        with console.open() as lines, jobs_path.open("a") as jobs:
            for line in lines:
                if not line.startswith("REGIMEX_ACCEPTED "):
                    continue
                try:
                    record = json.loads(line.removeprefix("REGIMEX_ACCEPTED "))
                    if record["phase"] != phase:
                        raise ValueError("k6 acceptance marker has the wrong phase")
                    if not isinstance(record["analysis_id"], str):
                        raise ValueError("analysis_id must be a UUID string")
                    identity = UUID(record["analysis_id"])
                except (ValueError, KeyError, TypeError) as exc:
                    marker_error = marker_error or f"{phase}: invalid acceptance marker: {exc}"
                    continue
                recorded += 1
                if identity in seen:
                    continue
                seen.add(identity)
                earlier_phase = previous.get(identity)
                if earlier_phase is None:
                    jobs.write(json.dumps(record) + "\n")
                elif earlier_phase != phase:
                    marker_error = marker_error or (
                        f"{phase}: analysis ID {identity} already accepted in {earlier_phase}"
                    )
    if marker_error:
        raise ValueError(marker_error)
    if recorded != len(seen):
        raise ValueError(f"{phase}: {recorded} acceptances reused {len(seen)} unique analysis IDs")
    if any(identity not in seen and owner == phase for identity, owner in previous.items()):
        raise ValueError(f"{phase}: current log is missing a previously recorded acceptance")
    summary = json.loads((directory / f"{phase}.json").read_bytes())
    expected = summary["metrics"]["analyses_accepted"]["count"]
    if recorded != expected:
        raise ValueError(f"{phase}: expected {expected} acceptance records, found {recorded}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("phase", choices=["warmup", "serial", "parallel", "sustained"])
    args = parser.parse_args()
    try:
        collect_acceptances(args.directory, args.phase)
    except (OSError, KeyError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")
