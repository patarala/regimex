"""Public CLI behavior against bounded, recorded service responses."""

import asyncio
import errno
import json
import os
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from uuid import uuid4

import httpx2
import pytest
from support.report_fixtures import prediction_for
from typer.testing import CliRunner

from regimex import analysis_client, service_client
from regimex.cli import app
from regimex.exports import load_export_v2

EXPORT = Path(__file__).parent / "fixtures" / "experiment-v2.json"
PREDICTION = prediction_for(load_export_v2(EXPORT))
PINNED_MODEL = PREDICTION.model.model_dump(mode="json")
ANALYSIS_ID = str(uuid4())
OPERATION_ID = str(uuid4())
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


def status_for(
    state: str, *, job: dict[str, object] | None = None, **changes: object
) -> dict[str, object]:
    return {
        "analysis_id": ANALYSIS_ID,
        "snapshot_id": SNAPSHOT_ID,
        "snapshot_sha256": PREDICTION.snapshot_sha256,
        "source_id": PREDICTION.source.source_id,
        "run_id": PREDICTION.source.run_id,
        "source_snapshot_id": PREDICTION.source.snapshot_id,
        "instrument_ids": [item.instrument_id for item in PREDICTION.instruments],
        "step_count": PREDICTION.time_grid.step_count,
        "job": {
            "kind": "analysis",
            "operation_id": OPERATION_ID,
            "resource_id": changes.get("analysis_id", ANALYSIS_ID),
            "state": state,
            "failure": None,
            "model": PINNED_MODEL,
            **(job or {}),
        },
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


def test_directory_input_closes_the_file_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    descriptor = os.open(tmp_path, os.O_RDONLY)
    monkeypatch.setattr(os, "open", lambda *args, **kwargs: descriptor)
    with pytest.raises(IsADirectoryError):
        service_client.read_file(tmp_path, 1024)
    with pytest.raises(OSError) as closed:
        os.fstat(descriptor)
    assert closed.value.errno == errno.EBADF


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
            assert "X-Regimex-Command" not in request.headers
            digest = json.loads(request.content)["content_sha256"]
            return reply(201, {"snapshot_id": SNAPSHOT_ID, "content_sha256": digest})
        if path == "/api/v1/analyses":
            assert request.headers["Idempotency-Key"] == key
            assert request.headers["X-Regimex-Command"] == "analyze"
            assert json.loads(request.content) == {
                "snapshot_id": SNAPSHOT_ID,
                "model_version_id": PREDICTION.model.model_version_id,
            }
            return reply(202, status_for("queued", polling_url="https://invalid.example/"))
        assert request.method == "GET"
        assert "X-Regimex-Command" not in request.headers
        if path.endswith("/report"):
            return reply(200, HTML)
        return reply(200, status_for("succeeded", report_url="https://invalid.example/"))

    mock_http(monkeypatch, service)
    result = CliRunner().invoke(
        app,
        [
            *command,
            "--input",
            str(EXPORT),
            "--model",
            PREDICTION.model.model_version_id,
            "--idempotency-key",
            key,
        ],
    )
    assert result.exit_code == 0, result.output
    assert key in result.output and ANALYSIS_ID in result.output
    assert f"http://127.0.0.1:8000/?analysis={ANALYSIS_ID}" in result.output
    assert calls == [
        "/api/v1/snapshots",
        "/api/v1/analyses",
        f"/api/v1/analyses/{ANALYSIS_ID}",
        f"/api/v1/analyses/{ANALYSIS_ID}/report",
    ]
    assert (tmp_path / "report.html").read_bytes() == HTML


@pytest.mark.parametrize(
    ("accepted", "message"),
    [
        (status_for("queued", snapshot_sha256="0" * 64), "different snapshot"),
        (status_for("queued"), "different model version"),
    ],
)
def test_rejects_receipt_for_a_different_input(
    command: list[str], monkeypatch: pytest.MonkeyPatch, accepted: dict[str, object], message: str
) -> None:
    def service(request: httpx2.Request) -> httpx2.Response:
        if request.url.path.endswith("/snapshots"):
            return reply(
                201, {"snapshot_id": SNAPSHOT_ID, "content_sha256": PREDICTION.snapshot_sha256}
            )
        assert request.url.path == "/api/v1/analyses"
        return reply(202, accepted)

    mock_http(monkeypatch, service)
    result = CliRunner().invoke(app, [*command, "--input", str(EXPORT), "--model", "other-model"])
    assert result.exit_code == 1 and message in result.output


@pytest.mark.parametrize(
    "job_change",
    [{"model": {**PINNED_MODEL, "bundle_sha256": "0" * 64}}, {"operation_id": str(uuid4())}],
)
def test_polling_preserves_the_first_observed_model_and_job(
    command: list[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    job_change: dict[str, object],
) -> None:
    responses = iter(
        [
            status_for("running"),
            status_for("succeeded", job=job_change),
        ]
    )

    async def skip_delay(_seconds: float) -> None:
        pass

    def service(request: httpx2.Request) -> httpx2.Response:
        assert request.url.path == f"/api/v1/analyses/{ANALYSIS_ID}"
        return reply(200, next(responses))

    mock_http(monkeypatch, service)
    monkeypatch.setattr(asyncio, "sleep", skip_delay)
    result = CliRunner().invoke(app, [*command, "--analysis-id", ANALYSIS_ID])
    assert result.exit_code == 1 and "model changed while waiting" in result.output
    assert not (tmp_path / "report.html").exists()


def test_submission_checks_the_source_against_the_uploaded_export(
    command: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def service(request: httpx2.Request) -> httpx2.Response:
        if request.url.path.endswith("/snapshots"):
            return reply(
                201, {"snapshot_id": SNAPSHOT_ID, "content_sha256": PREDICTION.snapshot_sha256}
            )
        assert request.url.path == "/api/v1/analyses"
        return reply(202, status_for("succeeded", source_snapshot_id="different-source-snapshot"))

    mock_http(monkeypatch, service)
    result = CliRunner().invoke(app, [*command, "--input", str(EXPORT)])
    assert result.exit_code == 1 and "snapshot metadata" in result.output


def test_submit_training_links_the_accepted_run_and_preserves_request_bytes(
    tmp_path: Path, token_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "training.json"
    path.write_bytes(b'{"display_name":"Forest", "parameters":{"threshold":1.0}}')
    key, run_id = uuid4(), uuid4()

    def service(request: httpx2.Request) -> httpx2.Response:
        assert str(request.url) == "https://regimex.example/api/v1/training-runs"
        assert request.method == "POST" and request.content == path.read_bytes()
        assert request.headers["Idempotency-Key"] == str(key)
        assert request.headers["X-Regimex-Command"] == "submit-training"
        assert request.headers["Authorization"] == "Bearer test-client-token"
        return reply(
            202,
            {
                "run_id": str(run_id),
                "job": {"kind": "training", "operation_id": str(run_id)},
                "polling_url": "https://unrelated.example/",
            },
        )

    mock_http(monkeypatch, service)
    monkeypatch.setenv("REGIMEX_CLI_ORIGIN", "https://unrelated.example")
    monkeypatch.setenv("REGIMEX_CLI_TOKEN_FILE", str(token_file))
    result = CliRunner().invoke(
        app,
        [
            "submit-training",
            "--input",
            str(path),
            "--origin",
            "https://regimex.example",
            "--idempotency-key",
            str(key),
        ],
    )
    assert result.exit_code == 0, result.output
    assert str(key) in result.output
    assert f"https://regimex.example/training/{run_id}" in result.output
    assert "unrelated" not in result.output


@pytest.mark.parametrize("issue", ["evaluation_job", "different_operation"])
def test_submit_training_rejects_a_receipt_for_another_job(
    tmp_path: Path, token_file: Path, monkeypatch: pytest.MonkeyPatch, issue: str
) -> None:
    run_id = str(uuid4())
    job = {
        "kind": "evaluation" if issue == "evaluation_job" else "training",
        "operation_id": str(uuid4()) if issue == "different_operation" else run_id,
    }
    mock_http(monkeypatch, lambda request: reply(202, {"run_id": run_id, "job": job}))
    path = tmp_path / "training.json"
    path.write_text("{}")
    result = CliRunner().invoke(
        app, ["submit-training", "--input", str(path), "--token-file", str(token_file)]
    )
    assert result.exit_code == 1 and "Open in browser" not in result.output


@pytest.mark.parametrize(("resume", "recorded_grid"), [(False, True), (True, True), (True, False)])
def test_save_json_predictions(
    tmp_path: Path,
    token_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    resume: bool,
    recorded_grid: bool,
) -> None:
    export = load_export_v2(EXPORT)
    payload = prediction_for(export).model_dump(mode="json")
    payload.update(analysis_id=ANALYSIS_ID, snapshot_id=SNAPSHOT_ID)
    # Large results use the artifact limit, while polling responses have a smaller limit.
    body = b" " * (64 * 1024) + json.dumps(payload).encode()
    calls = []
    completed = status_for("succeeded", report_available=False)
    if not recorded_grid:
        completed.update(instrument_ids=[], step_count=None)

    def service(request: httpx2.Request) -> httpx2.Response:
        path = request.url.path
        calls.append((request.method, path))
        if path.endswith("/snapshots"):
            return reply(201, {"snapshot_id": SNAPSHOT_ID, "content_sha256": export.content_sha256})
        if path.endswith("/analyses"):
            return reply(202, completed)
        if path == f"/api/v1/analyses/{ANALYSIS_ID}":
            return reply(200, completed)
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
        ("wrong_snapshot_hash", "pinned at intake"),
        ("wrong_weights", "pinned at intake"),
        ("wrong_inference_policy", "pinned at intake"),
        ("wrong_runtime", "inference settings"),
        ("wrong_source", "snapshot metadata"),
        ("wrong_instruments", "snapshot metadata"),
        ("wrong_steps", "snapshot metadata"),
        ("duplicate_instrument", "schema"),
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
            data = PREDICTION.model_dump(mode="json")
            if issue != "wrong_id":
                data.update(analysis_id=ANALYSIS_ID, snapshot_id=SNAPSHOT_ID)
            if issue == "wrong_snapshot_hash":
                data["snapshot_sha256"] = "0" * 64
            if issue == "wrong_weights":
                data["model"]["bundle_sha256"] = "0" * 64
            if issue == "wrong_inference_policy":
                data["model"]["inference_policy_sha256"] = "0" * 64
            if issue == "wrong_runtime":
                data["inference_runtime"]["threads"] += 1
            if issue == "wrong_source":
                data["source"]["run_id"] = "different-source-run"
            if issue == "duplicate_instrument":
                data["instruments"] = [data["instruments"][0]] * 2
                data["sequences"] = [data["sequences"][0]] * 2
            return reply(200, data)
        if issue == "wrong_steps":
            return reply(
                200, status_for("succeeded", step_count=PREDICTION.time_grid.step_count + 1)
            )
        if issue == "wrong_instruments":
            return reply(200, status_for("succeeded", instrument_ids=["another-instrument"]))
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
        (status_for("succeeded", job={"resource_id": str(uuid4())}), "schema"),
        (status_for("succeeded", job={"kind": "training"}), "schema"),
        (
            status_for(
                "failed",
                job={
                    "state": "failed",
                    "model": PINNED_MODEL,
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


def test_transient_reads_resume_status_and_download_without_resubmitting(
    tmp_path: Path, command: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    delays: list[float] = []
    calls: list[str] = []

    async def record_delay(seconds: float) -> None:
        delays.append(seconds)

    def service(request: httpx2.Request) -> httpx2.Response:
        assert request.method == "GET"
        path = request.url.path
        calls.append(path)
        if path.endswith("/report"):
            if calls.count(path) == 1:
                raise httpx2.ReadError("Connection closed during download.")
            return reply(200, HTML)
        if calls.count(path) == 1:
            return httpx2.Response(503, headers={"Retry-After": "4"})
        return reply(200, status_for("succeeded"))

    mock_http(monkeypatch, service)
    monkeypatch.setattr(asyncio, "sleep", record_delay)
    result = CliRunner().invoke(app, [*command, "--analysis-id", ANALYSIS_ID])
    assert result.exit_code == 0, result.output
    path = f"/api/v1/analyses/{ANALYSIS_ID}"
    assert calls == [path, path, path + "/report", path + "/report"]
    assert delays == [4, 1]
    assert (tmp_path / "report.html").read_bytes() == HTML


@pytest.mark.parametrize(
    ("value", "delay"),
    [
        ("Wed, 21 Oct 2015 07:28:00 GMT", 5),
        ("Wed, 21 Oct 2015 07:27:00 GMT", 2),
        ("invalid", 2),
    ],
)
def test_retry_after_date_uses_the_remaining_delay(
    value: str, delay: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(time, "time", lambda: 1445412475)
    assert service_client._retry_delay(value, minimum=2) == delay


@pytest.mark.parametrize(("status", "expected_calls"), [(404, 1), (503, 3)])
def test_download_preserves_final_error_and_leaves_no_partial_file(
    tmp_path: Path,
    command: list[str],
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    expected_calls: int,
) -> None:
    calls = 0

    async def skip_delay(_seconds: float) -> None:
        pass

    def service(request: httpx2.Request) -> httpx2.Response:
        nonlocal calls
        if request.url.path.endswith("/report"):
            calls += 1
            return reply(status, {"error": {"message": "Report unavailable."}})
        return reply(200, status_for("succeeded"))

    mock_http(monkeypatch, service)
    monkeypatch.setattr(asyncio, "sleep", skip_delay)
    result = CliRunner().invoke(app, [*command, "--analysis-id", ANALYSIS_ID])
    assert result.exit_code == 1 and f"HTTP {status}. Report unavailable." in result.output
    assert calls == expected_calls
    assert not (tmp_path / "report.html").exists()


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
