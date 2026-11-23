"""Exercise the terminal journal through the CLI and real local child processes."""

import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from uuid import uuid4

import httpx2
import pytest
from typer.testing import CliRunner

from regimex.activity_client import ActivityReceipt, _exclusive
from regimex.cli import app
from regimex.terminal_activity import ActivityStart

ORIGIN = "http://127.0.0.1:8000"


def record_http(
    monkeypatch: pytest.MonkeyPatch,
    on_start: Callable[[], None] | None = None,
) -> list[dict[str, object]]:
    bodies: list[dict[str, object]] = []
    record: dict[str, object] = {}

    def handle(request: httpx2.Request) -> httpx2.Response:
        assert request.headers["Authorization"] == "Bearer example-secret"
        assert request.url.host == "127.0.0.1"
        body = json.loads(request.content)
        bodies.append(body)
        if request.url.path == "/api/v1/terminal-activities":
            if not record:
                record.update(
                    note=None, status="running", finished_at=None, exit_code=None, result_text=None
                )
                record.update(body)
                if on_start:
                    on_start()
            assert record["activity_id"] == body["activity_id"]
            code = 201
        else:
            assert request.url.path == f"/api/v1/terminal-activities/{record['activity_id']}/finish"
            record.update(body)
            code = 200
        payload = json.dumps(record).encode()
        return httpx2.Response(
            code, headers={"Content-Type": "application/json"}, stream=httpx2.ByteStream(payload)
        )

    transport = httpx2.MockTransport(handle)
    monkeypatch.setattr(httpx2, "AsyncClient", partial(httpx2.AsyncClient, transport=transport))
    return bodies


def run_command(tmp_path: Path, child: list[str], options: list[str] | None = None) -> list[str]:
    return [
        "activity",
        "run",
        "--title",
        "Score holdout",
        "--receipt-dir",
        str(tmp_path / "receipts"),
        "--token-file",
        str(tmp_path / "user.token"),
        *(options or []),
        "--",
        *child,
    ]


def saved(tmp_path: Path) -> tuple[Path, ActivityReceipt]:
    (path,) = (tmp_path / "receipts").glob("*.json")
    return path, ActivityReceipt.model_validate_json(path.read_bytes())


def test_process_finishes_and_receipt_is_committed_while_start_response_is_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "user.token").write_text("example-secret")
    observed = []

    def while_pending() -> None:
        deadline = time.monotonic() + 5
        while True:
            _, receipt = saved(tmp_path)
            if receipt.finish is not None:
                observed.append(receipt.finish.finished_at)
                break
            if time.monotonic() >= deadline:
                return
            time.sleep(0.01)

    sent = record_http(monkeypatch, on_start=while_pending)
    attachment = tmp_path / "score.txt"
    code = (
        "import sys; from pathlib import Path; "
        "assert sys.argv[1] == '--private-arg'; Path(sys.argv[2]).write_text('score 0.93'); "
        "raise SystemExit(7)"
    )
    result = CliRunner().invoke(
        app,
        run_command(
            tmp_path,
            [sys.executable, "-c", code, "--private-arg", str(attachment)],
            ["--result-file", str(attachment), "--note", "Offline fit"],
        ),
    )
    path, receipt = saved(tmp_path)
    assert result.exit_code == 7, result.output
    assert receipt.finish and receipt.finish.status == "failed" and receipt.finish.exit_code == 7
    assert receipt.finish.result_text == "score 0.93"
    assert observed == [receipt.finish.finished_at]
    assert f"{ORIGIN}/activity/{receipt.start.activity_id}" in result.output
    assert sent[-1]["exit_code"] == 7 and sent[-1]["result_text"] == "score 0.93"
    tracked = path.read_text() + json.dumps(sent) + result.output
    assert "--private-arg" not in tracked and "example-secret" not in tracked
    assert path.stat().st_mode & 0o077 == 0


def test_offline_result_replays_only_to_original_service_without_rerunning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    counter = tmp_path / "executions"
    code = "import sys; from pathlib import Path; Path(sys.argv[1]).open('a').write('1')"
    result = CliRunner().invoke(
        app, run_command(tmp_path, [sys.executable, "-c", code, str(counter)])
    )
    assert result.exit_code == 0 and "saved locally" in result.output
    path, receipt = saved(tmp_path)
    assert receipt.finish and receipt.finish.status == "succeeded"
    (tmp_path / "user.token").write_text("example-secret")
    sent = record_http(monkeypatch)
    command = [
        "activity",
        "sync",
        "--receipt",
        str(path),
        "--token-file",
        str(tmp_path / "user.token"),
    ]
    wrong = CliRunner().invoke(app, [*command, "--origin", "https://elsewhere.example"])
    assert wrong.exit_code == 1 and "different service" in wrong.output and not sent
    for _ in range(2):
        replay = CliRunner().invoke(app, command)
        assert replay.exit_code == 0, replay.output
    assert len(sent) == 4 and sent[0] == sent[2] and sent[1] == sent[3]
    assert counter.read_text() == "1"


@pytest.mark.parametrize("attachment", ["astral", "token"])
def test_maximum_result_text_and_credential_file_attachment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, attachment: str
) -> None:
    token = tmp_path / "user.token"
    token.write_text("example-secret")
    result_path = tmp_path / "result.txt"
    expected = None
    if attachment == "token":
        result_path.hardlink_to(token)
    else:
        expected = "🌲" * 16_000
        result_path.write_text(expected)
    sent = record_http(monkeypatch)
    result = CliRunner().invoke(
        app,
        run_command(tmp_path, [sys.executable, "-c", "pass"], ["--result-file", str(result_path)]),
    )
    receipt_path, receipt = saved(tmp_path)
    assert result.exit_code == 0 and "Saved activity:" in result.output
    assert receipt.finish and receipt.finish.result_text == expected
    assert sent[-1].get("result_text") == expected
    assert "example-secret" not in receipt_path.read_text()
    done = CliRunner().invoke(
        app, ["activity", "sync", "--receipt", str(receipt_path), "--token-file", str(token)]
    )
    assert done.exit_code == 0, done.output


def test_unfinished_receipt_requires_closed_local_lock_before_interrupted_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "activity.json"
    start = ActivityStart(activity_id=uuid4(), title="Interrupted", started_at=datetime.now(UTC))
    path.write_text(ActivityReceipt(origin=ORIGIN, start=start).model_dump_json())
    token = tmp_path / "user.token"
    token.write_text("example-secret")
    sent = record_http(monkeypatch)
    command = [
        "activity",
        "sync",
        "--receipt",
        str(path),
        "--interrupted",
        "--token-file",
        str(token),
    ]
    with _exclusive(path):
        busy = CliRunner().invoke(app, command)
        assert busy.exit_code == 1 and "still running" in busy.output and not sent
    done = CliRunner().invoke(app, command)
    assert done.exit_code == 0, done.output
    assert sent[-1]["status"] == "interrupted" and "exit_code" not in sent[-1]


@pytest.mark.parametrize("outcome", ["missing", "denied", "signal"])
def test_launch_failures_and_child_signals_preserve_shell_exit_codes(
    tmp_path: Path, outcome: str
) -> None:
    executable = tmp_path / "private-executable"
    if outcome == "denied":
        executable.write_text("unused")
    command = [str(executable)]
    status, expected = "failed", 127 if outcome == "missing" else 126
    if outcome == "signal":
        command = [sys.executable, "-c", "import os,signal; os.kill(os.getpid(), signal.SIGTERM)"]
        status, expected = "interrupted", 143
    result = CliRunner().invoke(app, run_command(tmp_path, command))
    _, receipt = saved(tmp_path)
    assert result.exit_code == expected, result.output
    assert receipt.finish and receipt.finish.status == status
    assert receipt.finish.exit_code == (-15 if outcome == "signal" else expected)
    assert executable.name not in result.output


@pytest.mark.parametrize("bad_text", [b"x" * 16_001, b"wrong\0text", b"\xff"])
def test_invalid_result_attachment_keeps_local_exit_status(tmp_path: Path, bad_text: bytes) -> None:
    attached = tmp_path / "result.txt"
    attached.write_bytes(bad_text)
    result = CliRunner().invoke(
        app, run_command(tmp_path, [sys.executable, "-c", "pass"], ["--result-file", str(attached)])
    )
    _, receipt = saved(tmp_path)
    assert result.exit_code == 0 and "Could not attach" in result.output
    assert receipt.finish and receipt.finish.result_text is None


def test_interrupt_during_start_sync_records_finish_and_reaps_child(tmp_path: Path) -> None:
    ready = tmp_path / "start-pending"
    child_pid = tmp_path / "child"
    driver = """
import asyncio, sys
from pathlib import Path
from regimex import activity_client
from regimex.cli import app
async def slow_start(receipt, token_file, origin):
    if receipt.finish is None:
        Path(sys.argv[1]).touch()
        await asyncio.sleep(0.5)
    return origin
activity_client._sync = slow_start
app(args=sys.argv[2:])
"""
    child = (
        "import os,sys,time; from pathlib import Path; "
        "Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(60)"
    )
    command = [
        sys.executable,
        "-c",
        driver,
        str(ready),
        *run_command(tmp_path, [sys.executable, "-c", child, str(child_pid)]),
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        deadline = time.monotonic() + 10
        while not (ready.exists() and child_pid.exists()):
            assert process.poll() is None and time.monotonic() < deadline
            time.sleep(0.01)
        process.send_signal(signal.SIGINT)
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 130, (stdout, stderr)
        _, receipt = saved(tmp_path)
        assert receipt.finish and receipt.finish.status == "interrupted"
        with pytest.raises(ProcessLookupError):
            os.kill(int(child_pid.read_text()), 0)
    finally:
        process.kill()
        process.wait()
