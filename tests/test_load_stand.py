"""Check the measured interval excludes setup markers and includes the final verification."""

import json
import runpy
from pathlib import Path

import pytest


@pytest.mark.parametrize(("last_accounted", "window"), [(9, 12), (21, 18)])
def test_load_summary_uses_arrival_window_and_drain(
    tmp_path: Path, last_accounted: int, window: int
) -> None:
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
        "analysis_queue_first_half_samples": {"count": 2},
        "analysis_queue_second_half_samples": {"count": 3},
        "analysis_after_drain_samples": {"count": 1},
    }
    (tmp_path / "sustained.json").write_text(json.dumps({"metrics": metrics}))
    points = [
        (0, "analyses_started", 0),
        (0, "analyses_verified", 0),
        (3, "analyses_started", 1),
        (4, "analyses_started", 1),
        (5, "analyses_started", 1),
        (7, "analyses_accounted", 1),
        (7, "analyses_verified", 1),
        (9, "analyses_verified", 1),
        (9, "analyses_accounted", 1),
        (last_accounted, "analyses_accounted", 1),
        (24, "analysis_active_after_drain", 0),
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
    summary = stand["summarize"](tmp_path, "sustained", 123456, 12)
    assert summary["completed_per_minute"] == 120 / window
    assert summary["throughput_window_seconds"] == window
    assert summary["observed_seconds"] == 6
    assert summary["requested"] == 3
    assert summary["quota_refused"] == 1
    assert summary["dropped"] == 0
    assert summary["queued_second_half"]["max"] == 2
    assert summary["cgroup_peak_bytes"] == 123456


def test_secondary_evidence_failure_preserves_the_load_error() -> None:
    pytest.importorskip("psycopg")
    stand = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/load_stand.py"))

    def unavailable() -> None:
        raise RuntimeError("queue evidence unavailable")

    with (
        pytest.raises(RuntimeError, match="k6 failed") as failure,
        stand["finish_with"](unavailable),
    ):
        raise RuntimeError("k6 failed")
    assert "queue evidence unavailable" in failure.value.__notes__[0]


def test_stand_freezes_the_normalized_runtime(tmp_path: Path) -> None:
    from regimex.data_catalog import content_hash
    from regimex.model_registry import InferenceRuntime

    pytest.importorskip("psycopg")
    stand = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/load_stand.py"))
    policy = tmp_path / "runtime.json"
    policy.write_text('{"threads":1}')
    frozen = stand["inference_policy"](policy)
    policy.write_text('{"threads":3}')
    expected = InferenceRuntime(threads=1).model_dump(mode="json")
    assert frozen == {
        "inference_runtime": expected,
        "inference_policy_sha256": content_hash(expected),
        "runtime_override": True,
    }
    default = stand["inference_policy"](None)
    assert default["inference_policy_sha256"] == content_hash(
        InferenceRuntime().model_dump(mode="json")
    )
    policy.write_bytes(b" " * 8193)
    with pytest.raises(ValueError, match="8 KiB"):
        stand["inference_policy"](policy)


def test_stop_failure_preserves_private_data_and_original_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("psycopg")
    stand = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/load_stand.py"))

    def unverifiable(unit: str) -> None:
        raise OSError(f"Cannot reach manager for {unit}")

    globals_ = stand["private_scratch"].__wrapped__.__globals__
    monkeypatch.setitem(globals_, "stop_unit", unverifiable)
    with (
        pytest.raises(ValueError, match="k6 failed") as failure,
        stand["private_scratch"](tmp_path, "regimex-load-test.service") as scratch,
    ):
        marker = scratch / "query-in-progress"
        marker.write_text("data")
        raise ValueError("k6 failed")
    assert marker.is_file()
    assert str(scratch) in failure.value.__notes__[0]


def test_stop_confirmation_only_removes_the_original_scratch_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("psycopg")
    stand = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/load_stand.py"))
    globals_ = stand["private_scratch"].__wrapped__.__globals__
    monkeypatch.setitem(globals_, "stop_unit", lambda _unit: None)
    with (
        pytest.raises(RuntimeError, match="scratch path changed"),
        stand["private_scratch"](tmp_path, "regimex-load-test.service") as scratch,
    ):
        scratch.rename(tmp_path / "moved")
        scratch.mkdir()
        marker = scratch / "unrelated"
        marker.touch()
    assert marker.is_file()
    with stand["private_scratch"](tmp_path, "regimex-load-second.service") as clean:
        assert clean.is_dir()
    assert not clean.exists()
