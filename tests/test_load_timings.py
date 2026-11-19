"""Ambiguous queue deliveries and absent child timings stay out of load-test medians."""

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest

pytest.importorskip("psycopg")
pytest.importorskip("procrastinate")
pytest.importorskip("alembic")


def test_retried_and_missing_events_are_reported_separately(tmp_path: Path) -> None:
    from regimex.application.load_timings import (
        child_records,
        measure,
        phase_summaries,
        submissions,
    )
    from regimex.infrastructure.analysis_timings import QueueActivity

    job = uuid4()
    start = datetime.now(UTC)
    done = QueueActivity(
        job,
        "succeeded",
        1,
        1,
        ["deferred", "started", "succeeded"],
        start,
        start + timedelta(seconds=1),
        start + timedelta(seconds=4),
    )
    child = {
        "logger": "regimex.analysis",
        "action": "analysis_child_finished",
        "operation_id": str(job),
        "attempt": 1,
        "child_process_ms": 2000,
        "model_load_ms": 1000,
        "prediction_ms": 200,
        "report_ms": 300,
    }
    log = tmp_path / "worker.log"
    unrelated = {**child, "operation_id": str(uuid4())}
    log.write_text("worker started\n" + json.dumps(unrelated) + "\n" + json.dumps(child))
    events = child_records([log], {job: "serial"})[str(job)]
    good = measure("serial", job, done, events)
    assert good.queue_wait_ms == 1000 and good.worker_occupancy_ms == 3000
    missing = measure("serial", job, done, [])
    assert missing.issues == ["child_event_missing"] and missing.model_load_ms is None
    retried = replace(done, attempts=2, deliveries=2, events=done.events + done.events)
    repeat = measure("parallel", job, retried, events * 2)
    assert {
        "analysis_retried",
        "multiple_deliveries",
        "multiple_starts",
        "multiple_child_events",
    } <= set(repeat.issues)
    assert repeat.worker_occupancy_ms is None and repeat.child_process_ms is None
    incomplete = measure("parallel", job, replace(done, events=done.events[:-1]), events)
    assert incomplete.queue_wait_ms is None and incomplete.issues
    assert measure(
        "parallel", job, replace(done, started=start - timedelta(seconds=1)), events
    ).issues
    assert measure("parallel", job, None, []).issues == ["analysis_missing"]
    assert measure("parallel", job, done, [{**child, "attempt": 2}]).issues == [
        "child_attempt_not_unique"
    ]
    assert measure("parallel", job, done, [{**child, "report_ms": None}]).issues == [
        "invalid_child_timings"
    ]
    summary = phase_summaries([good, missing, repeat, incomplete])
    assert summary["serial"]["submitted"] == 2 and summary["serial"]["complete"] == 1
    assert summary["serial"]["median_ms"]["worker_occupancy"] == 3000
    assert summary["parallel"]["median_ms"]["worker_occupancy"] is None

    jobs = tmp_path / "jobs.jsonl"
    jobs.write_text((json.dumps({"analysis_id": str(job), "phase": "serial"}) + "\n") * 2)
    with pytest.raises(ValueError, match="unique analysis ID"):
        submissions(jobs)
