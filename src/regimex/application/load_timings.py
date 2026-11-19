"""Join accepted analysis IDs to operator-side queue and child timings."""

import json
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypedDict
from uuid import UUID

from regimex.infrastructure.analysis_timings import QueueActivity, queue_activity
from regimex.infrastructure.database import DatabaseRepository
from regimex.storage import write_json

CHILD_METRICS = ("child_process", "model_load", "prediction", "report")
METRICS = ("queue_wait", "worker_occupancy", *CHILD_METRICS, "backend_predict")


@dataclass
class AnalysisTiming:
    phase: str
    analysis_id: str
    state: str | None
    attempts: int | None
    deliveries: int | None
    events: list[str]
    issues: list[str]
    queue_wait_ms: float | None = None
    worker_occupancy_ms: float | None = None
    child_process_ms: float | None = None
    model_load_ms: float | None = None
    prediction_ms: float | None = None
    backend_predict_ms: float | None = None
    report_ms: float | None = None


def submissions(path: Path) -> dict[UUID, str]:
    jobs: dict[UUID, str] = {}
    with path.open() as stream:
        for line in stream:
            item = json.loads(line)
            job = UUID(item["analysis_id"])
            phase = item["phase"]
            if not isinstance(phase, str) or not phase or job in jobs:
                raise ValueError("Expected a named phase and a unique analysis ID per acceptance")
            jobs[job] = phase
    if not jobs:
        raise ValueError("No accepted analyses in the jobs manifest")
    return jobs


def child_records(paths: list[Path], jobs: dict[UUID, str]) -> dict[str, list[dict[str, object]]]:
    ids = {str(job) for job in jobs}
    records: dict[str, list[dict[str, object]]] = defaultdict(list)
    for path in paths:
        with path.open() as stream:
            for line in stream:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (
                    isinstance(record, dict)
                    and record.get("logger") == "regimex.analysis"
                    and record.get("action") == "analysis_child_finished"
                    and record.get("operation_id") in ids
                ):
                    records[record["operation_id"]].append(record)
    return records


def duration_ms(value: object) -> float | None:
    if (
        isinstance(value, (float, int))
        and not isinstance(value, bool)
        and 0 <= value <= sys.float_info.max
    ):
        return float(value)
    return None


def measure(
    phase: str, job: UUID, activity: QueueActivity | None, children: list[dict[str, object]]
) -> AnalysisTiming:
    if activity is None:
        return AnalysisTiming(phase, str(job), None, None, None, [], ["analysis_missing"])
    record = AnalysisTiming(
        phase, str(job), activity.state, activity.attempts, activity.deliveries, activity.events, []
    )
    if activity.state != "succeeded":
        record.issues.append("analysis_" + activity.state)
    if activity.attempts > 1:
        record.issues.append("analysis_retried")
    if activity.deliveries > 1:
        record.issues.append("multiple_deliveries")
    if activity.events.count("started") > 1:
        record.issues.append("multiple_starts")
    if activity.events != ["deferred", "started", "succeeded"]:
        record.issues.append("incomplete_or_irregular_queue_events")
    elif (
        activity.deferred is None
        or activity.started is None
        or activity.succeeded is None
        or not activity.deferred <= activity.started <= activity.succeeded
    ):
        record.issues.append("invalid_queue_timestamps")
    elif activity.deliveries == 1 and activity.attempts == 1:
        record.queue_wait_ms = (activity.started - activity.deferred).total_seconds() * 1000
        record.worker_occupancy_ms = (activity.succeeded - activity.started).total_seconds() * 1000
    else:
        record.issues.append("delivery_without_unique_attempt")

    if len(children) != 1:
        record.issues.append("child_event_missing" if not children else "multiple_child_events")
    elif (
        activity.attempts != 1
        or type(children[0].get("attempt")) is not int
        or children[0]["attempt"] != 1
    ):
        record.issues.append("child_attempt_not_unique")
    else:
        child, model, prediction, report = [
            duration_ms(children[0].get(metric + "_ms")) for metric in CHILD_METRICS
        ]
        if child is None or model is None or prediction is None or report is None:
            record.issues.append("invalid_child_timings")
        elif model + prediction + report > child:
            record.issues.append("stages_exceed_child_time")
        else:
            record.child_process_ms = child
            record.model_load_ms = model
            record.prediction_ms = prediction
            record.report_ms = report
            backend = duration_ms(children[0].get("backend_predict_ms"))
            if backend is None or backend > prediction:
                record.issues.append("invalid_backend_predict_timing")
            else:
                record.backend_predict_ms = backend
            if record.worker_occupancy_ms is not None and child > record.worker_occupancy_ms:
                record.issues.append("child_exceeds_worker_occupancy")
    return record


class PhaseSummary(TypedDict):
    submitted: int
    complete: int
    median_ms: dict[str, float | None]


def phase_summaries(records: list[AnalysisTiming]) -> dict[str, PhaseSummary]:
    phases: dict[str, list[AnalysisTiming]] = defaultdict(list)
    for record in records:
        phases[record.phase].append(record)
    summaries: dict[str, PhaseSummary] = {}
    for phase, submitted in phases.items():
        complete = [record for record in submitted if not record.issues]
        medians = {
            metric: statistics.median(getattr(record, metric + "_ms") for record in complete)
            if complete
            else None
            for metric in METRICS
        }
        summaries[phase] = {
            "submitted": len(submitted),
            "complete": len(complete),
            "median_ms": medians,
        }
    return summaries


def collect_timings(
    repository: DatabaseRepository, jobs_path: Path, worker_logs: list[Path], output: Path
) -> bool:
    """Write the report even when incomplete evidence disqualifies a measurement."""
    jobs = submissions(jobs_path)
    deadline = time.monotonic() + 5
    while True:
        activity = queue_activity(repository, list(jobs))
        # A client can download a result before the worker acknowledges its delivery.
        pending_ack = any(
            row.state == "succeeded" and row.events == ["deferred", "started"]
            for row in activity.values()
        )
        if not pending_ack or time.monotonic() >= deadline:
            break
        time.sleep(0.1)
    children = child_records(worker_logs, jobs)
    records = [
        measure(phase, job, activity.get(job), children.get(str(job), []))
        for job, phase in jobs.items()
    ]
    write_json(
        output, {"analyses": [asdict(r) for r in records], "phases": phase_summaries(records)}
    )
    return all(not record.issues for record in records)
