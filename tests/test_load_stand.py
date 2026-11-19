"""Check the measured interval excludes setup markers and includes the final verification."""

import json
import runpy
from pathlib import Path

import pytest


def test_load_summary_uses_verified_points_and_preserves_quota_refusals(tmp_path: Path) -> None:
    pytest.importorskip("psycopg")
    stand = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/load_stand.py"))
    metrics = {
        "analyses_scheduled": {"count": 3},
        "analyses_started": {"count": 3},
        "analyses_accepted": {"count": 2},
        "analyses_verified": {"count": 2},
        "analyses_quota_refused": {"count": 1},
        "analysis_to_result": {"med": 4_000, "p(95)": 5_900},
        "analysis_intake": {"med": 150},
        "analysis_queued_first_half": {"med": 1, "max": 1},
        "analysis_queued_second_half": {"med": 2, "max": 2},
        "analysis_active_after_drain": {"max": 0},
    }
    (tmp_path / "sustained.json").write_text(json.dumps({"metrics": metrics}))
    (tmp_path / "memory.peak").write_text("123456")
    points = [
        (0, "analyses_started", 0),
        (0, "analyses_verified", 0),
        (3, "analyses_started", 1),
        (4, "analyses_started", 1),
        (5, "analyses_started", 1),
        (7, "analyses_verified", 1),
        (9, "analyses_verified", 1),
        (11, "analysis_active_after_drain", 0),
    ]
    with (tmp_path / "sustained-points.jsonl").open("w") as stream:
        for second, metric, value in points:
            timestamp = f"2026-01-01T00:00:{second:02}Z"
            stream.write(
                json.dumps(
                    {"type": "Point", "metric": metric, "data": {"time": timestamp, "value": value}}
                )
                + "\n"
            )
    summary = stand["summarize"](tmp_path, "sustained", tmp_path)
    assert summary["completed_per_minute"] == 20
    assert summary["observed_seconds"] == 6
    assert summary["requested"] == 3
    assert summary["quota_refused"] == 1
    assert summary["dropped"] == 0
    assert summary["queued_second_half"]["max"] == 2
    assert summary["cgroup_peak_bytes"] == 123456
