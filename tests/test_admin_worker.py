"""Computation limits cover metadata, late output and process descendants."""

import ctypes
import os
import signal
import subprocess
import sys
import time
from contextlib import suppress
from os import stat_result
from pathlib import Path
from subprocess import Popen
from types import SimpleNamespace
from typing import Any, cast

import pytest

import regimex.computation_worker as computation_worker
from regimex.computation_worker import (
    ChildLimits,
    ComputationCancelled,
    ComputationFailure,
    supervise_child,
)
from regimex.resource_limits import ResourceBudgetExceeded

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux child supervision")


@pytest.mark.parametrize("leader_outcome", ["success", "cancelled"])
@pytest.mark.parametrize("descendant_stop", ["graceful", "forced", "oversized"])
def test_supervisor_stops_its_descendants_after_leader_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    leader_outcome: str,
    descendant_stop: str,
) -> None:
    # Adopt this fixture's orphan so the test can reap it instead of leaving a
    # zombie for the host's init. Restore the original process setting afterward.
    libc = ctypes.CDLL(None, use_errno=True)
    previous_subreaper = ctypes.c_int()
    assert libc.prctl(37, ctypes.byref(previous_subreaper), 0, 0, 0) == 0
    assert libc.prctl(36, 1, 0, 0, 0) == 0
    script = """
import os
import signal
import sys
import time
from pathlib import Path
root = Path(sys.argv[1])
if os.fork() == 0:
    def graceful_stop(signum, frame):
        time.sleep(0.1)
        if sys.argv[3] == 'oversized':
            (root / 'late-output').write_bytes(b'x' * 8192)
        (root / 'graceful-stop').touch()
        sys.exit(0)
    signal.signal(signal.SIGTERM, signal.SIG_IGN if sys.argv[3] == 'forced' else graceful_stop)
    (root / 'descendant.tmp').write_text(str(os.getpid()))
    (root / 'descendant.tmp').rename(root / 'descendant.pid')
    while True:
        time.sleep(0.01)
while not (root / 'descendant.pid').exists():
    time.sleep(0.01)
if sys.argv[2] == 'success':
    (root / 'child-result.json').write_text('{}')
    sys.exit(0)
while True:
    time.sleep(0.01)
"""
    original_launch = subprocess.Popen
    children: list[Popen[bytes]] = []
    descendant: int | None = None
    unrelated = original_launch(
        [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
    )

    def launch(*args: Any, **kwargs: Any) -> Popen[bytes]:
        child = cast(
            Popen[bytes],
            original_launch(
                [sys.executable, "-c", script, str(tmp_path), leader_outcome, descendant_stop],
                **kwargs,
            ),
        )
        children.append(child)
        return child

    def heartbeat() -> None:
        nonlocal descendant
        ready = tmp_path / "descendant.pid"
        deadline = time.monotonic() + 5
        while not ready.exists():
            assert time.monotonic() < deadline, "fixture descendant did not start"
            time.sleep(0.01)
        descendant = int(ready.read_text())
        if leader_outcome == "cancelled":
            raise ComputationCancelled()

    try:
        monkeypatch.setattr(subprocess, "Popen", launch)
        limits = ChildLimits(10, 1024**3, 4096, 1)
        if leader_outcome == "cancelled":
            with pytest.raises(ComputationCancelled):
                supervise_child({}, tmp_path, limits=limits, heartbeat=heartbeat)
        elif descendant_stop == "oversized":
            with pytest.raises(ComputationFailure) as error:
                supervise_child({}, tmp_path, limits=limits, heartbeat=heartbeat)
            assert error.value.failure.code == "capacity_limit"
            assert "scratch" in str(error.value)
        else:
            assert supervise_child({}, tmp_path, limits=limits, heartbeat=heartbeat) == {}
        assert len(children) == 1 and children[0].poll() is not None
        assert descendant is not None
        # KILL delivery is asynchronous; wait only for this known descendant.
        deadline = time.monotonic() + 5
        while True:
            reaped, status = os.waitpid(descendant, os.WNOHANG)
            if reaped == descendant:
                descendant = None
                break
            assert time.monotonic() < deadline, "computation descendant survived supervision"
            time.sleep(0.01)
        if descendant_stop != "forced":
            assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0
            assert (tmp_path / "graceful-stop").exists()
        else:
            assert os.WIFSIGNALED(status) and os.WTERMSIG(status) == signal.SIGKILL
        assert unrelated.poll() is None
    finally:
        for child in (*children, unrelated):
            with suppress(ProcessLookupError):
                os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=5)
        if descendant is not None:
            os.waitpid(descendant, 0)
        assert libc.prctl(36, previous_subreaper.value, 0, 0, 0) == 0


@pytest.mark.parametrize(
    ("failure", "expected_code"),
    [
        (ResourceBudgetExceeded("private allocation details"), "capacity_limit"),
        (ValueError("private validation details"), "invalid_data"),
    ],
)
def test_budget_failures_remain_distinct_from_invalid_inputs(
    failure: Exception, expected_code: str
) -> None:
    result = computation_worker._failure_from_exception(failure)
    assert result.code == expected_code
    assert not result.retryable
    assert "private" not in result.message


@pytest.mark.parametrize("output_kind", ["file", "directory"])
def test_child_output_can_be_renamed_during_a_live_scratch_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, output_kind: str
) -> None:
    script = """
import sys
import time
from pathlib import Path
root = Path(sys.argv[1])
pending = root / 'pending-output'
if sys.argv[2] == 'directory':
    pending.mkdir()
    (pending / 'data').write_bytes(b'x' * 128)
else:
    pending.write_bytes(b'x' * 128)
while not (root / 'continue').exists():
    time.sleep(0.001)
pending.rename(root / 'output')
(root / 'child-result.json').write_text('{}')
"""
    original_launch = subprocess.Popen
    original_lstat = Path.lstat
    children: list[Popen[bytes]] = []
    renamed = False

    def launch(*args: Any, **kwargs: Any) -> Popen[bytes]:
        child = cast(
            Popen[bytes],
            original_launch([sys.executable, "-c", script, str(tmp_path), output_kind], **kwargs),
        )
        children.append(child)
        return child

    def rename_before_measurement(path: Path) -> stat_result:
        nonlocal renamed
        if path == tmp_path / "pending-output" and not renamed:
            renamed = True
            (tmp_path / "continue").touch()
            assert children[0].wait(timeout=5) == 0
        return original_lstat(path)

    monkeypatch.setattr(subprocess, "Popen", launch)
    monkeypatch.setattr(Path, "lstat", rename_before_measurement)
    assert (
        supervise_child(
            {},
            tmp_path,
            limits=ChildLimits(5, 1024**3, 4096, 1),
            heartbeat=lambda: None,
        )
        == {}
    )
    assert renamed and len(children) == 1 and children[0].returncode == 0
    output = tmp_path / "output"
    assert (output / "data" if output_kind == "directory" else output).read_bytes() == b"x" * 128


def test_request_metadata_cannot_start_a_child_over_the_scratch_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden_launch(*args: object, **kwargs: object) -> None:
        pytest.fail("over-budget request started a computation process")

    monkeypatch.setattr(subprocess, "Popen", forbidden_launch)
    with pytest.raises(ComputationFailure, match="request metadata") as error:
        supervise_child(
            {"payload": "x" * 4096},
            tmp_path,
            limits=ChildLimits(5, 1024**3, 4096, 1),
            heartbeat=lambda: None,
        )
    assert error.value.failure.code == "capacity_limit"
    assert not (tmp_path / "child-request.json").exists()


@pytest.mark.parametrize("limit", ["scratch", "wall_time", "within_budget"])
def test_completed_child_still_has_to_meet_aggregate_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, limit: str
) -> None:
    # Release a real subprocess after the supervisor's last running-child check,
    # then wait for its successful exit before the next poll. This makes the
    # between-polls completion deterministic without mocking process state.
    script = """
import sys
import time
from pathlib import Path
root = Path(sys.argv[1])
while not (root / 'continue').exists():
    time.sleep(0.001)
for index in range(2):
    (root / f'output-{index}').write_bytes(b'x' * int(sys.argv[2]))
(root / 'child-result.json').write_text('{}')
"""
    original_launch = subprocess.Popen
    original_monotonic = time.monotonic
    children: list[Popen[bytes]] = []
    elapsed = 0.0

    def launch(*args: Any, **kwargs: Any) -> Popen[bytes]:
        child = cast(
            Popen[bytes],
            original_launch(
                [
                    sys.executable,
                    "-c",
                    script,
                    str(tmp_path),
                    "2048" if limit == "scratch" else "0",
                ],
                **kwargs,
            ),
        )
        children.append(child)
        return child

    def finish_between_polls(seconds: float) -> None:
        nonlocal elapsed
        (tmp_path / "continue").touch()
        assert children[0].wait(timeout=5) == 0
        if limit == "wall_time":
            elapsed = 10.0

    monkeypatch.setattr(subprocess, "Popen", launch)
    monkeypatch.setattr(
        computation_worker,
        "time",
        SimpleNamespace(
            sleep=finish_between_polls,
            monotonic=lambda: original_monotonic() + elapsed,
        ),
    )
    limits = ChildLimits(5, 1024**3, 4096, 1)
    if limit == "within_budget":
        assert supervise_child({}, tmp_path, limits=limits, heartbeat=lambda: None) == {}
    else:
        with pytest.raises(ComputationFailure) as error:
            supervise_child({}, tmp_path, limits=limits, heartbeat=lambda: None)
        assert error.value.failure.code == "capacity_limit"
        assert ("scratch" if limit == "scratch" else "wall-time") in str(error.value)
    assert len(children) == 1 and children[0].returncode == 0
