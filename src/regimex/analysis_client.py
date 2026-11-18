"""Submit an export and retrieve the saved HTML report or JSON predictions."""

import asyncio
import ipaddress
import json
import os
import ssl
import stat
from collections.abc import Callable
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit
from uuid import UUID, uuid4

import httpx2
from pydantic import BaseModel, ValidationError

from regimex.auth_contracts import canonical_origin
from regimex.contracts import MAX_INPUT_BYTES
from regimex.exports import parse_export_v2
from regimex.operations import OperationFailure, OperationState
from regimex.report_contracts import MAX_REPORT_BYTES
from regimex.storage import write_new_file

_JSON_LIMIT = 64 * 1024


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


def _service_origin(value: str) -> str:
    origin = canonical_origin(value)
    parsed = urlsplit(origin)
    if parsed.scheme == "http":
        try:
            if ipaddress.ip_address(parsed.hostname or "").is_loopback:
                return origin
        except ValueError:
            pass
        raise ValueError("Use HTTPS for remote services or a literal loopback address for HTTP.")
    return origin


def _read_file(path: Path, limit: int) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"Expected a regular file: {path}")
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"{path} exceeds the {limit}-byte limit.")
    return data


def _read_token(path: Path) -> str:
    token = _read_file(path, 4096).strip()
    if not token or not all(33 <= byte <= 126 for byte in token):
        raise ValueError("The token file must contain a single ASCII bearer token.")
    return token.decode("ascii")


def _display(text: str) -> str:
    return "".join(char if char.isprintable() else " " for char in text[:1024])


async def _response_bytes(response: httpx2.Response, limit: int) -> bytes:
    if response.headers.get("content-encoding", "identity").lower() != "identity":
        raise ValueError("The service returned an unsupported content encoding.")
    length = response.headers.get("content-length")
    if length is not None and (
        not length.isascii() or not length.isdecimal() or int(length) > limit
    ):
        raise ValueError("The response has an invalid or oversized Content-Length.")
    data = bytearray()
    async for chunk in response.aiter_raw():
        if len(data) + len(chunk) > limit:
            raise ValueError("The response exceeded its size limit.")
        data.extend(chunk)
    if length is not None and len(data) != int(length):
        raise ValueError("The service returned an incomplete response.")
    return bytes(data)


async def _request(
    client: httpx2.AsyncClient,
    path: str,
    *,
    expected_status: int = 200,
    content: bytes | None = None,
    key: UUID | None = None,
    media_type: str = "application/json",
    limit: int = _JSON_LIMIT,
) -> bytes:
    headers = {"Accept": media_type}
    if content is not None:
        headers["Content-Type"] = "application/json"
    if key is not None:
        headers["Idempotency-Key"] = str(key)
    method = "GET" if content is None else "POST"
    async with client.stream(method, path, headers=headers, content=content) as response:
        actual_type = response.headers.get("content-type", "").partition(";")[0].lower().strip()
        if response.status_code != expected_status:
            message = f"Service returned HTTP {response.status_code}."
            if actual_type == "application/json":
                body = await _response_bytes(response, _JSON_LIMIT)
                try:
                    error = json.loads(body)["error"]["message"]
                except ValueError, KeyError, TypeError:
                    pass
                else:
                    if isinstance(error, str):
                        message += " " + _display(error)
            raise RuntimeError(message)
        if actual_type != media_type:
            raise ValueError(f"Expected a {media_type} response from the service.")
        return await _response_bytes(response, limit)


async def _submit(
    client: httpx2.AsyncClient, input_path: Path, model_version: str, key: UUID
) -> _Analysis:
    payload = _read_file(input_path, MAX_INPUT_BYTES)
    try:
        export = parse_export_v2(payload)
    except ValidationError as error:
        detail = error.errors(include_context=False, include_input=False)[0]
        raise ValueError(f"Invalid export: {_display(detail['msg'])}") from None
    saved = _Snapshot.model_validate_json(
        await _request(client, "snapshots", content=payload, expected_status=201)
    )
    if saved.content_sha256 != export.content_sha256:
        raise ValueError("The saved snapshot does not match the input export.")
    request = {"snapshot_id": str(saved.snapshot_id), "model_version_id": model_version}
    analysis = _Analysis.model_validate_json(
        await _request(
            client, "analyses", content=json.dumps(request).encode(), key=key, expected_status=202
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
            status = _Analysis.model_validate_json(await _request(client, path))
        if status.analysis_id != analysis_id:
            raise ValueError("The service returned a different analysis ID.")
        if status.job.state == "succeeded":
            return status
        if status.job.state == "failed":
            failure = status.job.failure
            detail = f" ({failure.code}): {_display(failure.message)}" if failure else "."
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
        html = await _request(
            client, path + "/report", media_type="text/html", limit=MAX_REPORT_BYTES
        )
        if not html:
            raise ValueError("The service returned an empty report.")
        return html

    from regimex.application.service import AnalysisResult

    if not analysis.result_available:
        raise ValueError("The analysis succeeded without a saved JSON result.")
    data = await _request(client, path + "/result", limit=MAX_INPUT_BYTES)
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
    base = _service_origin(origin)
    token = _read_token(token_file)
    if model_version is not None and not 1 <= len(model_version) <= 128:
        raise ValueError("The model version must contain between 1 and 128 characters.")
    try:
        async with (
            asyncio.timeout(timeout),
            httpx2.AsyncClient(
                base_url=f"{base}/api/v1/",
                headers={"Authorization": "Bearer " + token, "Accept-Encoding": "identity"},
                verify=ssl.create_default_context(),
                trust_env=False,
                follow_redirects=False,
                timeout=30,
            ) as client,
        ):
            status = None
            if input_path is not None:
                key = idempotency_key or uuid4()
                progress(f"Idempotency key: {key}")
                status = await _submit(client, input_path, model_version or "active", key)
                analysis_id = status.analysis_id
            progress(f"Analysis ID: {analysis_id}")
            assert analysis_id is not None
            completed = await _wait_for_success(client, analysis_id, status)
            write_new_file(output, await _download(client, completed, output_format))
    except TimeoutError, httpx2.TimeoutException:
        raise RuntimeError("Timed out waiting for the service.") from None
    except httpx2.RequestError as error:
        raise RuntimeError(
            f"Service request failed ({type(error).__name__}). Check the connection."
        ) from None
    except ValidationError:
        raise ValueError("The service response did not match its schema.") from None
