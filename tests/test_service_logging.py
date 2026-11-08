"""Structured logging through the standard handler and exception boundary."""

import io
import json
import logging

import pytest

from regimex.service_logging import RedactingJSONFormatter


@pytest.mark.parametrize("exc_info", [False, None, True])
def test_successful_job_logging_without_an_exception(
    exc_info: bool | None, capsys: pytest.CaptureFixture[str]
) -> None:
    output = io.StringIO()
    handler = logging.StreamHandler(output)
    handler.setFormatter(RedactingJSONFormatter(("private-worker-identity",)))
    logger = logging.Logger("procrastinate.worker.maintenance", logging.INFO)
    logger.addHandler(handler)

    # Procrastinate supplies exc_info=False when a job completes successfully.
    logger.log(
        logging.INFO,
        "Job %s ended with status: Success; token=%s",
        "regimex.recover[1]",
        "private-job-token",
        exc_info=exc_info,
        extra={
            "action": "job_success",
            "worker_id": "private-worker-identity",
            "task_kwargs": {"payload": "private-job-payload"},
        },
    )

    captured = capsys.readouterr()
    assert captured.err == ""
    lines = output.getvalue().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["message"] == (
        "Job regimex.recover[1] ended with status: Success; token=[redacted]"
    )
    assert record["level"] == "INFO"
    assert record["action"] == "job_success"
    assert record["worker_id"] == "[redacted]"
    assert "exception_type" not in record
    assert "task_kwargs" not in record
    assert "private-" not in lines[0]


def test_failed_job_logging_preserves_exception_type_without_payload(
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = io.StringIO()
    handler = logging.StreamHandler(output)
    handler.setFormatter(RedactingJSONFormatter(()))
    logger = logging.Logger("procrastinate.worker.maintenance", logging.INFO)
    logger.addHandler(handler)

    try:
        raise ValueError("private-exception-payload")
    except ValueError:
        logger.exception(
            "Job failed: Authorization: Bearer %s",
            "private-job-token",
            extra={"action": "job_error", "payload": "private-job-payload"},
        )

    captured = capsys.readouterr()
    assert captured.err == ""
    lines = output.getvalue().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["message"] == "Job failed: Authorization: [redacted]"
    assert record["exception_type"] == "ValueError"
    assert record["action"] == "job_error"
    assert record["level"] == "ERROR"
    assert "private-" not in lines[0]
    assert "Traceback" not in lines[0]
