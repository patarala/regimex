"""Check the measured interval excludes setup markers and includes the final verification."""

import json
import os
import runpy
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from uuid import UUID

import pytest


def write_acceptances(path: Path, phase: str, identities: list[int], count: int) -> None:
    records = [{"phase": phase, "analysis_id": str(UUID(int=i))} for i in identities]
    (path / f"{phase}-console.log").write_text(
        "".join("REGIMEX_ACCEPTED " + json.dumps(record) + "\n" for record in records)
    )
    (path / f"{phase}.json").write_text(
        json.dumps({"metrics": {"analyses_accepted": {"count": count}}})
    )


@pytest.mark.parametrize(
    ("identities", "count", "error"),
    [
        ([1, 2], 2, None),
        ([1, 2, 1, 3], 4, "4 acceptances reused 3 unique analysis IDs"),
        ([1, 2], 3, "expected 3 acceptance records, found 2"),
    ],
)
def test_phase_checks_acceptance_marks_across_virtual_users(
    tmp_path: Path, identities: list[int], count: int, error: str | None
) -> None:
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    collect = runpy.run_path(str(scripts / "load_acceptances.py"))["collect_acceptances"]
    write_acceptances(tmp_path, "sustained", identities, count)
    if error:
        with pytest.raises(ValueError, match=error):
            collect(tmp_path, "sustained")
    else:
        collect(tmp_path, "sustained")
    jobs = [json.loads(line) for line in (tmp_path / "jobs.jsonl").read_text().splitlines()]
    assert [UUID(job["analysis_id"]).int for job in jobs] == list(dict.fromkeys(identities))
    assert all(job["phase"] == "sustained" for job in jobs)


def test_reconciling_a_phase_twice_preserves_unique_ids_across_phases(tmp_path: Path) -> None:
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    collect = runpy.run_path(str(scripts / "load_acceptances.py"))["collect_acceptances"]
    write_acceptances(tmp_path, "warmup", [1], 1)
    collect(tmp_path, "warmup")
    collect(tmp_path, "warmup")
    write_acceptances(tmp_path, "sustained", [2, 1, 3], 3)
    for _ in range(2):
        with pytest.raises(ValueError, match=f"{UUID(int=1)} already accepted in warmup"):
            collect(tmp_path, "sustained")
    jobs = [json.loads(line) for line in (tmp_path / "jobs.jsonl").read_text().splitlines()]
    assert [(job["phase"], UUID(job["analysis_id"]).int) for job in jobs] == [
        ("warmup", 1),
        ("sustained", 2),
        ("sustained", 3),
    ]
    write_acceptances(tmp_path, "sustained", [2], 1)
    with pytest.raises(ValueError, match="missing a previously recorded acceptance"):
        collect(tmp_path, "sustained")


def test_bad_acceptance_marker_still_records_later_valid_ids(tmp_path: Path) -> None:
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    collect = runpy.run_path(str(scripts / "load_acceptances.py"))["collect_acceptances"]
    write_acceptances(tmp_path, "parallel", [1, 2], 3)
    log = tmp_path / "parallel-console.log"
    log.write_text("REGIMEX_ACCEPTED {\n" + log.read_text())
    with pytest.raises(ValueError, match="invalid acceptance marker"):
        collect(tmp_path, "parallel")
    jobs = [json.loads(line) for line in (tmp_path / "jobs.jsonl").read_text().splitlines()]
    assert [UUID(job["analysis_id"]).int for job in jobs] == [1, 2]


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
        "dropped_iterations": {"count": 0},
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

    (tmp_path / "cell-group").mkdir()

    def unverifiable(group: Path) -> None:
        raise OSError(f"Cannot verify cgroup {group}")

    globals_ = stand["private_scratch"].__wrapped__.__globals__
    monkeypatch.setitem(globals_, "stop_group", unverifiable)
    with (
        pytest.raises(ValueError, match="k6 failed") as failure,
        stand["private_scratch"](tmp_path, tmp_path / "cell-group") as scratch,
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
    monkeypatch.setitem(globals_, "stop_group", lambda _group: None)
    (tmp_path / "cell-group").mkdir()
    (tmp_path / "second-group").mkdir()
    with (
        pytest.raises(RuntimeError, match=r"directory at .* changed"),
        stand["private_scratch"](tmp_path, tmp_path / "cell-group") as scratch,
    ):
        scratch.rename(tmp_path / "moved")
        scratch.mkdir()
        marker = scratch / "unrelated"
        marker.touch()
    assert marker.is_file()
    external = tmp_path / "external"
    external.mkdir()
    with stand["private_scratch"](tmp_path, tmp_path / "second-group") as clean:
        digest_dir = clean / "worker-1" / "cache" / ("a" * 64)
        digest_dir.mkdir(parents=True)
        (digest_dir / "weights.pt").touch(mode=0o400)
        (digest_dir / "external-link").symlink_to(external, target_is_directory=True)
        digest_dir.chmod(0o500)
    assert external.is_dir()
    assert not clean.exists()


def test_failed_cell_reports_partial_k6_outcomes_and_keeps_failure_status(tmp_path: Path) -> None:
    pytest.importorskip("psycopg")
    stand = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/load_stand.py"))
    cell = tmp_path / "sustained-512-2w"
    cell.mkdir()
    metrics = {
        "analyses_scheduled": {"count": 3},
        "analyses_started": {"count": 3},
        "analyses_accepted": {"count": 2},
        "analyses_verified": {"count": 2},
        "analyses_failed": {"count": 1},
        "analyses_submit_failed": {"count": 1},
        "http_req_failed": {"passes": 1, "fails": 8},
        "analysis_to_result": {"med": 3000, "p(95)": 3900},
    }
    (cell / "sustained.json").write_text(json.dumps({"metrics": metrics}))
    (cell / "sustained-points.jsonl").write_text(
        json.dumps(
            {
                "type": "Point",
                "metric": "http_req_failed",
                "data": {"value": 1, "tags": {"status": "503", "name": "submit analysis"}},
            }
        )
        + "\n"
    )
    with (
        pytest.raises(RuntimeError, match="exit 99"),
        stand["record_cell"](tmp_path, [], 1, "sustained", 512, 2),
    ):
        raise RuntimeError("k6 exit 99")
    rows = json.loads((tmp_path / "comparison.json").read_text())
    assert rows == [json.loads((cell / "result.json").read_text())]
    assert rows[0]["status"] == "failed"
    assert rows[0]["http_failures"] == 1
    assert rows[0]["http_failure_samples"] == [
        {"request": "submit analysis", "status": "503", "count": 1}
    ]
    assert rows[0]["completed"] == 2 and rows[0]["requested"] == 3
    assert rows[0]["dropped"] is None and rows[0]["cgroup_peak_bytes"] is None
    markdown = (tmp_path / "comparison.md").read_text()
    assert "# Failed load comparison (1 of 1 cells reported)" in markdown
    assert "2 / 3" in markdown and "HTTP 503 (count 1)" in markdown
    assert "3.00 | 3.90 | unavailable | 1 |" in markdown


def test_interrupted_cell_preserves_evidence_from_intact_files(tmp_path: Path) -> None:
    pytest.importorskip("psycopg")
    stand = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/load_stand.py"))
    cell = tmp_path / "sustained-512-2w"
    cell.mkdir()
    (cell / "warmup.json").write_text('{"metrics":{"analyses_verified":{"count":1}}}')
    for name in ("sustained.json", "result.json", "timings.json"):
        (cell / name).write_bytes(b'{"')
    (cell / "envelope.json").write_text('{"peak_bytes":1234}')
    (cell / "sustained-points.jsonl").write_bytes(
        b'{"type":"Point","metric":"http_req_failed",'
        b'"data":{"value":1,"tags":{"name":"poll analysis","status":"503"}}}\n{"\xc3'
    )
    with (
        pytest.raises(KeyboardInterrupt),
        stand["record_cell"](tmp_path, [], 1, "sustained", 512, 2),
    ):
        raise KeyboardInterrupt
    (row,) = json.loads((tmp_path / "comparison.json").read_bytes())
    assert (row["phase"], row["status"], row["completed"]) == ("sustained", "failed", None)
    assert row["cgroup_peak_bytes"] == 1234
    assert row["http_failure_samples"] == [
        {"request": "poll analysis", "status": "503", "count": 1}
    ]
    assert set(row["unreadable_artifacts"]) == {"result.json", "sustained.json", "timings.json"}
    markdown = (tmp_path / "comparison.md").read_text()
    assert "# Failed load comparison (1 of 1 cells reported)" in markdown
    assert "poll analysis: HTTP 503 (count 1)" in markdown
    assert "timings.json" in markdown


def test_cgroup_parent_requires_delegated_controllers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("psycopg")
    stand = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/load_stand.py"))
    check = stand["delegated_parent"]
    monkeypatch.setitem(check.__globals__, "CGROUP_ROOT", tmp_path)
    parent = tmp_path / "load-delegation"
    parent.mkdir()
    (parent / "cgroup.controllers").write_text("cpu memory")
    enabled = parent / "cgroup.subtree_control"
    enabled.write_text("cpu")
    with pytest.raises(ValueError, match="REGIMEX_LOAD_CGROUP_PARENT"):
        check(None)
    with pytest.raises(ValueError, match=r"cgroup\.subtree_control"):
        check(parent)
    assert enabled.read_text() == "cpu"
    enabled.write_text("cpu memory")
    assert check(parent) == parent


def test_stopping_a_cell_waits_until_descendants_are_gone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("psycopg")
    stand = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/load_stand.py"))
    events = tmp_path / "cgroup.events"
    events.write_text("populated 1\nfrozen 0\n")

    def finish_exit(_delay: float) -> None:
        assert (tmp_path / "cgroup.kill").read_text() == "1"
        events.write_text("populated 0\nfrozen 0\n")

    monkeypatch.setattr("time.sleep", finish_exit)
    stand["stop_group"](tmp_path)
    assert "populated 0" in events.read_text()


def test_moved_group_keeps_its_identity_during_cleanup(tmp_path: Path) -> None:
    pytest.importorskip("psycopg")
    stand = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/load_stand.py"))
    with (
        pytest.raises(RuntimeError, match=r"directory at .* changed"),
        stand["new_group"](tmp_path) as (handle, _),
    ):
        path = handle.resolve()
        path.rename(tmp_path / "moved")
        path.mkdir()
        (path / "cgroup.kill").write_text("unrelated")
        (handle / "cgroup.events").write_text("populated 0")
        stand["stop_group"](handle)
    assert (path / "cgroup.kill").read_text() == "unrelated"
    assert (tmp_path / "moved" / "cgroup.kill").read_text() == "1"


def test_cell_supervisor_observes_timeout_and_driver_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("psycopg")
    stand = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/load_stand.py"))
    reader, writer = os.pipe()
    with (
        os.fdopen(reader, "rb") as lifetime,
        os.fdopen(writer, "wb") as driver,
        subprocess.Popen(["sleep", "30"]) as cell,
    ):
        try:
            monkeypatch.setattr("sys.stdin", lifetime)
            with pytest.raises(subprocess.TimeoutExpired):
                stand["wait_for_cell"](cell, 0)
            driver.close()
            with pytest.raises(RuntimeError, match="driver disappeared"):
                stand["wait_for_cell"](cell, 1)
        finally:
            cell.kill()


def test_driver_cleans_owned_resources_after_supervisor_is_killed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("psycopg")
    stand = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/load_stand.py"))
    script = tmp_path / "killed-supervisor.py"
    script.write_text(
        "import os, signal, sys\nfrom pathlib import Path\n"
        "scratch = Path(sys.argv[3])\n"
        "(scratch / 'token').touch()\n"
        "(scratch.parent / 'started').write_text(str(scratch))\n"
        "os.kill(os.getpid(), signal.SIGKILL)\n"
    )
    parent = tmp_path / "cgroups"
    parent.mkdir()

    @contextmanager
    def group_fixture(parent: Path) -> Iterator[tuple[Path, int]]:
        with stand["new_group"](parent) as (group, fd):
            (group / "memory.peak").write_text("0")
            (group / "cgroup.events").write_text("populated 0\n")
            try:
                yield group, fd
            finally:
                assert (group / "cgroup.kill").read_text() == "1"
                for control in group.iterdir():
                    control.unlink()

    start = stand["start_supervisor"]
    monkeypatch.setitem(start.__globals__, "__file__", str(script))
    monkeypatch.setitem(start.__globals__, "new_group", group_fixture)
    config = {"cgroup_parent": str(parent), "cpu_ids": [1], "memory_bytes": 4096}
    with pytest.raises(subprocess.CalledProcessError) as failure:
        start(tmp_path / "run.json", config, 512, 2, "sustained")
    assert failure.value.returncode == -9
    assert not Path((tmp_path / "started").read_text()).exists()
    assert not list(parent.iterdir())


def test_failed_journey_preserves_partial_counts_and_http_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("psycopg")
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    stand = runpy.run_path(str(scripts / "load_stand.py"))
    (tmp_path / "k6.log").touch()
    metrics = {
        "journey_jobs_requested": {"count": 5},
        "journey_jobs_accepted": {"count": 4},
        "journey_results_verified": {"count": 1},
        "http_req_failed": {"passes": 1},
    }
    (tmp_path / "summary.json").write_text(json.dumps({"metrics": metrics}))
    stages = {"queue_wait_ms": 123.4}
    (tmp_path / "timings.json").write_text(
        json.dumps({"phases": {"contention-first-short": {"median_ms": stages}}})
    )
    point = {
        "type": "Point",
        "metric": "http_req_failed",
        "data": {"value": 1, "tags": {"name": "poll analysis", "status": "0"}},
    }
    (tmp_path / "points.jsonl").write_text(json.dumps(point) + "\n")
    report = stand["failed_cell"](tmp_path, "journey-contention", 256, 2)
    assert report["status"] == "failed" and report["phase"] == "journey-contention"
    assert report["requested"] == 5 and report["accepted"] == 4 and report["completed"] == 1
    assert report["stage_groups"] == {"contention-first-short": stages}
    assert report["result_median_ms"] is None and report["cancelled"] is None
    assert report["http_failure_samples"] == [
        {"request": "poll analysis", "status": "0", "count": 1}
    ]
