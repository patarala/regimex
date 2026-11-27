"""Bounded, authenticated HTTP transport for the terminal clients."""

import asyncio
import ipaddress
import json
import os
import ssl
import stat
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit
from uuid import UUID

import httpx2
from pydantic import ValidationError

from regimex.auth_contracts import canonical_origin

_JSON_LIMIT = 64 * 1024
_RETRYABLE_READ_STATUSES = {408, 429, 502, 503, 504}


def service_origin(value: str) -> str:
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


def read_file(path: Path, limit: int) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"Expected a regular file: {path}")
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"{path} exceeds the {limit}-byte limit.")
    return data


def read_token(path: Path) -> str:
    token = read_file(path, 4096).strip()
    if not token or not all(33 <= byte <= 126 for byte in token):
        raise ValueError("The token file must contain a single ASCII bearer token.")
    return token.decode("ascii")


def display_text(text: str) -> str:
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


def _retry_delay(value: str | None, minimum: float) -> float:
    if value is None:
        return minimum
    if value.isascii() and value.isdecimal():
        return max(minimum, float(value))
    try:
        date = parsedate_to_datetime(value)
        if date.tzinfo is not None:
            return max(minimum, date.timestamp() - time.time())
    except ValueError, OverflowError:
        pass
    return minimum


async def _checked_body(
    response: httpx2.Response, expected_status: int, media_type: str, limit: int
) -> bytes:
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
                    message += " " + display_text(error)
        raise RuntimeError(message)
    if actual_type != media_type:
        raise ValueError(f"Expected a {media_type} response from the service.")
    return await _response_bytes(response, limit)


async def request(
    client: httpx2.AsyncClient,
    path: str,
    *,
    expected_status: int = 200,
    content: bytes | None = None,
    key: UUID | None = None,
    command: Literal["analyze", "submit-training"] | None = None,
    media_type: str = "application/json",
    limit: int = _JSON_LIMIT,
) -> bytes:
    headers = {"Accept": media_type}
    if content is not None:
        headers["Content-Type"] = "application/json"
    if key is not None:
        headers["Idempotency-Key"] = str(key)
    if command is not None:
        headers["X-Regimex-Command"] = command
    method = "GET" if content is None else "POST"
    attempt = 0
    while True:
        attempt += 1
        may_retry = method == "GET" and attempt < 3
        delay = float(attempt)
        try:
            async with client.stream(method, path, headers=headers, content=content) as response:
                if not may_retry or response.status_code not in _RETRYABLE_READ_STATUSES:
                    return await _checked_body(response, expected_status, media_type, limit)
                delay = _retry_delay(response.headers.get("retry-after"), delay)
        except httpx2.NetworkError, httpx2.TimeoutException, httpx2.RemoteProtocolError:
            if not may_retry:
                raise
        await asyncio.sleep(delay)


@asynccontextmanager
async def connect(
    origin: str, token_file: Path, *, timeout: int
) -> AsyncIterator[httpx2.AsyncClient]:
    base = service_origin(origin)
    token = read_token(token_file)
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
            yield client
    except TimeoutError, httpx2.TimeoutException:
        raise RuntimeError("Timed out waiting for the service.") from None
    except httpx2.RequestError as error:
        raise RuntimeError(
            f"Service request failed ({type(error).__name__}). Check the connection."
        ) from None
    except ValidationError:
        raise ValueError("The service response did not match its schema.") from None
