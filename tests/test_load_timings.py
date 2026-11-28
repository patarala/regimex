"""Ambiguous queue deliveries and absent child timings stay out of load-test medians."""

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock
from uuid import uuid4

import pytest

pytest.importorskip("psycopg")
pytest.importorskip("procrastinate")
pytest.importorskip("alembic")


def test_retried_and_missing_events_are_reported_separately(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from regimex.application.load_timings import (
        collect_timings,
        measure,
        phase_summaries,
        submissions,
    )
    from regimex.infrastructure.analysis_timings import QueueActivity
    from regimex.infrastructure.database import DatabaseRepository

    job, operation_id = uuid4(), uuid4()
    start = datetime.now(UTC)
    done = QueueActivity(
        job,
        operation_id,
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
        "operation_id": str(operation_id),
        "attempt": 1,
        "child_process_ms": 2000,
        "model_load_ms": 1000,
        "prediction_ms": 200,
        "backend_predict_ms": 100,
        "report_ms": 300,
    }
    log = tmp_path / "worker.log"
    unrelated = {**child, "operation_id": str(job), "prediction_ms": 0}
    log.write_text("worker started\n" + json.dumps(unrelated) + "\n" + json.dumps(child))
    jobs = tmp_path / "jobs.jsonl"
    jobs.write_text(json.dumps({"analysis_id": str(job), "phase": "serial"}) + "\n")
    monkeypatch.setattr(
        "regimex.application.load_timings.queue_activity", lambda repo, ids: {job: done}
    )
    output = tmp_path / "timings.json"
    assert collect_timings(Mock(spec=DatabaseRepository), jobs, [log], output)
    (measured,) = json.loads(output.read_bytes())["analyses"]
    assert measured["analysis_id"] == str(job)
    assert measured["prediction_ms"] == 200
    events = [child]
    good = measure("serial", job, done, events)
    assert good.queue_wait_ms == 1000 and good.worker_occupancy_ms == 3000
    assert good.backend_predict_ms == 100
    invalid = measure("serial", job, done, [{**child, "backend_predict_ms": 201}])
    assert invalid.backend_predict_ms is None and invalid.prediction_ms == 200
    assert invalid.issues == ["invalid_backend_predict_timing"]
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
    assert summary["serial"]["median_ms"]["backend_predict"] == 100
    assert summary["parallel"]["median_ms"]["worker_occupancy"] is None

    jobs.write_text((json.dumps({"analysis_id": str(job), "phase": "serial"}) + "\n") * 2)
    with pytest.raises(ValueError, match="unique analysis ID"):
        submissions(jobs)
