"""Public CLI behavior against bounded, recorded service responses."""

import json
from collections.abc import Callable
from functools import partial
from pathlib import Path
from uuid import uuid4

import httpx2
import pytest
from support.report_fixtures import prediction_for
from typer.testing import CliRunner

from regimex import analysis_client
from regimex.cli import app
from regimex.exports import load_export_v2

EXPORT = Path(__file__).parent / "fixtures" / "experiment-v2.json"
ANALYSIS_ID = str(uuid4())
SNAPSHOT_ID = str(uuid4())
HTML = b"<!doctype html><title>Analysis</title>"


def reply(status: int, value: object) -> httpx2.Response:
    body = value if isinstance(value, bytes) else json.dumps(value).encode()
    content_type = "text/html" if isinstance(value, bytes) else "application/json"
    return httpx2.Response(
        status,
        headers={"Content-Type": content_type, "Content-Length": str(len(body))},
        stream=httpx2.ByteStream(body),
    )


def status_for(state: str, **changes: object) -> dict[str, object]:
    return {
        "analysis_id": ANALYSIS_ID,
        "snapshot_id": SNAPSHOT_ID,
        "job": {"state": state, "failure": None},
        "result_available": state == "succeeded",
        "report_available": state == "succeeded",
        **changes,
    }


@pytest.fixture
def token_file(tmp_path: Path) -> Path:
    token = tmp_path / "user.token"
    token.write_text("test-client-token\n")
    return token


@pytest.fixture
def command(tmp_path: Path, token_file: Path) -> list[str]:
    return ["analyze", "--output", str(tmp_path / "report.html"), "--token-file", str(token_file)]


def mock_http(
    monkeypatch: pytest.MonkeyPatch, handler: Callable[[httpx2.Request], httpx2.Response]
) -> None:
    transport = httpx2.MockTransport(handler)
    monkeypatch.setattr(httpx2, "AsyncClient", partial(httpx2.AsyncClient, transport=transport))


def test_submit_wait_and_save(
    tmp_path: Path, command: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    key = str(uuid4())

    def service(request: httpx2.Request) -> httpx2.Response:
        path = request.url.path
        calls.append(path)
        assert request.url.host == "127.0.0.1"
        assert request.headers["Authorization"] == "Bearer test-client-token"
        if path == "/api/v1/snapshots":
            assert request.method == "POST" and request.content == EXPORT.read_bytes()
            digest = json.loads(request.content)["content_sha256"]
            return reply(201, {"snapshot_id": SNAPSHOT_ID, "content_sha256": digest})
        if path == "/api/v1/analyses":
            assert request.headers["Idempotency-Key"] == key
            assert json.loads(request.content) == {
                "snapshot_id": SNAPSHOT_ID,
                "model_version_id": "chosen",
            }
            return reply(202, status_for("queued", polling_url="https://invalid.example/"))
        assert request.method == "GET"
        if path.endswith("/report"):
            return reply(200, HTML)
        return reply(200, status_for("succeeded", report_url="https://invalid.example/"))

    mock_http(monkeypatch, service)
    result = CliRunner().invoke(
        app, [*command, "--input", str(EXPORT), "--model", "chosen", "--idempotency-key", key]
    )
    assert result.exit_code == 0, result.output
    assert key in result.output and ANALYSIS_ID in result.output
    assert calls == [
        "/api/v1/snapshots",
        "/api/v1/analyses",
        f"/api/v1/analyses/{ANALYSIS_ID}",
        f"/api/v1/analyses/{ANALYSIS_ID}/report",
    ]
    assert (tmp_path / "report.html").read_bytes() == HTML


@pytest.mark.parametrize("resume", [False, True])
def test_save_json_predictions(
    tmp_path: Path, token_file: Path, monkeypatch: pytest.MonkeyPatch, resume: bool
) -> None:
    export = load_export_v2(EXPORT)
    payload = prediction_for(export).model_dump(mode="json")
    payload.update(analysis_id=ANALYSIS_ID, snapshot_id=SNAPSHOT_ID)
    # Large results use the artifact limit, while polling responses have a smaller limit.
    body = b" " * (64 * 1024) + json.dumps(payload).encode()
    calls = []

    def service(request: httpx2.Request) -> httpx2.Response:
        path = request.url.path
        calls.append((request.method, path))
        if path.endswith("/snapshots"):
            return reply(201, {"snapshot_id": SNAPSHOT_ID, "content_sha256": export.content_sha256})
        if path.endswith("/analyses"):
            return reply(202, status_for("succeeded", report_available=False))
        if path == f"/api/v1/analyses/{ANALYSIS_ID}":
            return reply(200, status_for("succeeded", report_available=False))
        assert path == f"/api/v1/analyses/{ANALYSIS_ID}/result"
        assert request.headers["Accept"] == "application/json"
        return httpx2.Response(
            200, headers={"Content-Type": "application/json"}, stream=httpx2.ByteStream(body)
        )

    output = tmp_path / "predictions.json"
    source = ["--analysis-id", ANALYSIS_ID] if resume else ["--input", str(EXPORT)]
    mock_http(monkeypatch, service)
    result = CliRunner().invoke(
        app,
        [
            "analyze",
            "--output",
            str(output),
            "--token-file",
            str(token_file),
            "--format",
            "json",
            *source,
        ],
    )
    assert result.exit_code == 0, result.output
    assert output.read_bytes() == body
    expected = (
        [("GET", f"/api/v1/analyses/{ANALYSIS_ID}")]
        if resume
        else [("POST", "/api/v1/snapshots"), ("POST", "/api/v1/analyses")]
    )
    assert calls == [*expected, ("GET", f"/api/v1/analyses/{ANALYSIS_ID}/result")]


@pytest.mark.parametrize(
    ("issue", "message"),
    [
        ("missing", "without a saved JSON result"),
        ("invalid", "schema"),
        ("wrong_id", "different analysis"),
    ],
)
def test_rejects_unusable_json_result(
    tmp_path: Path, command: list[str], monkeypatch: pytest.MonkeyPatch, issue: str, message: str
) -> None:
    def service(request: httpx2.Request) -> httpx2.Response:
        if request.url.path.endswith("/result"):
            assert issue != "missing"
            if issue == "invalid":
                return reply(200, {"invalid": True})
            return reply(200, prediction_for(load_export_v2(EXPORT)).model_dump(mode="json"))
        return reply(200, status_for("succeeded", result_available=issue != "missing"))

    mock_http(monkeypatch, service)
    result = CliRunner().invoke(app, [*command, "--format", "json", "--analysis-id", ANALYSIS_ID])
    assert result.exit_code == 1 and message in result.output
    assert not (tmp_path / "report.html").exists()


@pytest.mark.parametrize(
    ("value", "error"),
    [
        (status_for("cancelled"), "cancelled"),
        (status_for("succeeded", report_available=False), "without a saved HTML report"),
        (status_for("running", analysis_id=str(uuid4())), "different analysis ID"),
        (
            status_for(
                "failed",
                job={
                    "state": "failed",
                    "failure": {
                        "code": "capacity_limit",
                        "message": "The model exceeds its memory budget.",
                    },
                },
            ),
            "exceeds its memory budget",
        ),
    ],
)
def test_resume_reports_job_state_without_submission(
    tmp_path: Path, command: list[str], monkeypatch: pytest.MonkeyPatch, value: object, error: str
) -> None:
    def service(request: httpx2.Request) -> httpx2.Response:
        assert request.method == "GET" and request.url.path == f"/api/v1/analyses/{ANALYSIS_ID}"
        return reply(200, value)

    mock_http(monkeypatch, service)
    result = CliRunner().invoke(app, [*command, "--analysis-id", ANALYSIS_ID])
    assert result.exit_code == 1 and error in result.output
    assert not (tmp_path / "report.html").exists()


def test_resumed_download_preserves_a_concurrent_output(
    tmp_path: Path, command: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "report.html"

    def service(request: httpx2.Request) -> httpx2.Response:
        assert request.method == "GET"
        if request.url.path.endswith("/report"):
            output.write_bytes(b"other download")
            return reply(200, HTML)
        return reply(200, status_for("succeeded"))

    mock_http(monkeypatch, service)
    result = CliRunner().invoke(app, [*command, "--analysis-id", ANALYSIS_ID])
    assert result.exit_code == 1
    assert output.read_bytes() == b"other download"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["report.html", "user.token"]


def test_rejects_redirect_and_cleartext_remote_service(
    command: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def service(request: httpx2.Request) -> httpx2.Response:
        calls.append(str(request.url))
        return httpx2.Response(307, headers={"Location": "https://other.example/token"})

    mock_http(monkeypatch, service)
    result = CliRunner().invoke(app, [*command, "--analysis-id", ANALYSIS_ID])
    assert result.exit_code == 1 and "HTTP 307" in result.output
    assert len(calls) == 1 and "test-client-token" not in result.output
    result = CliRunner().invoke(
        app, [*command, "--analysis-id", ANALYSIS_ID, "--origin", "http://other.example"]
    )
    assert result.exit_code == 1 and "HTTPS" in result.output
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("length", "limit", "error"),
    [(len(HTML) + 1, 1000, "incomplete"), (len(HTML), len(HTML) - 1, "oversized")],
)
def test_truncated_or_oversized_report_leaves_no_file(
    tmp_path: Path,
    command: list[str],
    monkeypatch: pytest.MonkeyPatch,
    length: int,
    limit: int,
    error: str,
) -> None:
    monkeypatch.setattr(analysis_client, "MAX_REPORT_BYTES", limit)

    def service(request: httpx2.Request) -> httpx2.Response:
        if request.url.path.endswith("/report"):
            response = reply(200, HTML)
            response.headers["Content-Length"] = str(length)
            return response
        return reply(200, status_for("succeeded"))

    mock_http(monkeypatch, service)
    result = CliRunner().invoke(app, [*command, "--analysis-id", ANALYSIS_ID])
    assert result.exit_code == 1 and error in result.output
    assert not (tmp_path / "report.html").exists()


def test_timeout_prints_resumable_id(command: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    mock_http(monkeypatch, lambda request: reply(200, status_for("running")))
    result = CliRunner().invoke(app, [*command, "--analysis-id", ANALYSIS_ID, "--timeout", "1"])
    assert result.exit_code == 1 and ANALYSIS_ID in result.output and "Timed out" in result.output


def test_submission_replay_after_lost_response(
    command: list[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    keys: list[str] = []

    def service(request: httpx2.Request) -> httpx2.Response:
        if request.url.path.endswith("/snapshots"):
            digest = json.loads(request.content)["content_sha256"]
            return reply(201, {"snapshot_id": SNAPSHOT_ID, "content_sha256": digest})
        if request.url.path.endswith("/report"):
            return reply(200, HTML)
        assert json.loads(request.content)["model_version_id"] == "active"
        keys.append(request.headers["Idempotency-Key"])
        if len(keys) == 1:
            raise httpx2.ReadError("Submission outcome unknown.")
        return reply(202, status_for("succeeded"))

    mock_http(monkeypatch, service)
    submit = [*command, "--input", str(EXPORT)]
    result = CliRunner().invoke(app, submit)
    assert result.exit_code == 1 and keys[0] in result.output
    result = CliRunner().invoke(app, [*submit, "--idempotency-key", keys[0]])
    assert result.exit_code == 0, result.output
    assert keys[0] == keys[1] and (tmp_path / "report.html").read_bytes() == HTML
