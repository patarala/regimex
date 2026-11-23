"""Submit an export and retrieve the saved HTML report or JSON predictions."""

import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

import httpx2
from pydantic import BaseModel, ValidationError

from regimex.contracts import MAX_INPUT_BYTES
from regimex.exports import parse_export_v2
from regimex.operations import OperationFailure, OperationState
from regimex.report_contracts import MAX_REPORT_BYTES
from regimex.service_client import connect, display_text, read_file, request
from regimex.storage import write_new_file


class _Snapshot(BaseModel):
    snapshot_id: UUID
    content_sha256: str


class _Job(BaseModel):
    state: OperationState
    failure: OperationFailure | None


class _Analysis(BaseModel):
    analysis_id: UUID
    snapshot_id: UUID
    job: _Job
    result_available: bool
    report_available: bool


async def _submit(
    client: httpx2.AsyncClient, input_path: Path, model_version: str, key: UUID
) -> _Analysis:
    payload = read_file(input_path, MAX_INPUT_BYTES)
    try:
        export = parse_export_v2(payload)
    except ValidationError as error:
        detail = error.errors(include_context=False, include_input=False)[0]
        raise ValueError(f"Invalid export: {display_text(detail['msg'])}") from None
    saved = _Snapshot.model_validate_json(
        await request(client, "snapshots", content=payload, expected_status=201)
    )
    if saved.content_sha256 != export.content_sha256:
        raise ValueError("The saved snapshot does not match the input export.")
    body = {"snapshot_id": str(saved.snapshot_id), "model_version_id": model_version}
    analysis = _Analysis.model_validate_json(
        await request(
            client,
            "analyses",
            content=json.dumps(body).encode(),
            key=key,
            expected_status=202,
            command="analyze",
        )
    )
    if analysis.snapshot_id != saved.snapshot_id:
        raise ValueError("The service returned an analysis for a different snapshot.")
    return analysis


async def _wait_for_success(
    client: httpx2.AsyncClient, analysis_id: UUID, status: _Analysis | None
) -> _Analysis:
    path = f"analyses/{analysis_id}"
    while True:
        if status is None:
            status = _Analysis.model_validate_json(await request(client, path))
        if status.analysis_id != analysis_id:
            raise ValueError("The service returned a different analysis ID.")
        if status.job.state == "succeeded":
            return status
        if status.job.state == "failed":
            failure = status.job.failure
            detail = f" ({failure.code}): {display_text(failure.message)}" if failure else "."
            raise ValueError(f"Analysis failed{detail}")
        if status.job.state == "cancelled":
            raise ValueError("The analysis was cancelled.")
        await asyncio.sleep(1.5)
        status = None


async def _download(
    client: httpx2.AsyncClient, analysis: _Analysis, output_format: Literal["html", "json"]
) -> bytes:
    path = f"analyses/{analysis.analysis_id}"
    if output_format == "html":
        if not analysis.report_available:
            raise ValueError("The analysis succeeded without a saved HTML report.")
        html = await request(
            client, path + "/report", media_type="text/html", limit=MAX_REPORT_BYTES
        )
        if not html:
            raise ValueError("The service returned an empty report.")
        return html

    from regimex.application.service import AnalysisResult

    if not analysis.result_available:
        raise ValueError("The analysis succeeded without a saved JSON result.")
    data = await request(client, path + "/result", limit=MAX_INPUT_BYTES)
    result = AnalysisResult.model_validate_json(data)
    if (result.analysis_id, result.snapshot_id) != (analysis.analysis_id, analysis.snapshot_id):
        raise ValueError("The saved result belongs to a different analysis or snapshot.")
    return data


async def analyze(
    *,
    input_path: Path | None,
    analysis_id: UUID | None,
    model_version: str | None,
    idempotency_key: UUID | None,
    origin: str,
    token_file: Path,
    output: Path,
    timeout: int,
    progress: Callable[[str], None],
    output_format: Literal["html", "json"] = "html",
) -> None:
    if (input_path is None) == (analysis_id is None):
        raise ValueError("Supply exactly one of --input and --analysis-id.")
    if analysis_id is not None and (model_version is not None or idempotency_key is not None):
        raise ValueError("--model and --idempotency-key apply only to a new submission.")
    if output.exists() or output.is_symlink():
        raise ValueError(f"Output already exists: {output}. Choose a new filename.")
    output.parent.mkdir(parents=True, exist_ok=True)
    if model_version is not None and not 1 <= len(model_version) <= 128:
        raise ValueError("The model version must contain between 1 and 128 characters.")
    async with connect(origin, token_file, timeout=timeout) as client:
        status = None
        if input_path is not None:
            key = idempotency_key or uuid4()
            progress(f"Idempotency key: {key}")
            status = await _submit(client, input_path, model_version or "active", key)
            analysis_id = status.analysis_id
        progress(f"Analysis ID: {analysis_id}")
        assert analysis_id is not None
        progress(f"Open in browser: {client.base_url.join(f'/?analysis={analysis_id}')}")
        completed = await _wait_for_success(client, analysis_id, status)
        write_new_file(output, await _download(client, completed, output_format))
