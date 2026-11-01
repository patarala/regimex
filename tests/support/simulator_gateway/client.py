"""Bounded, configured-origin HTTP client; never follows upstream redirects."""

import http.client
import ipaddress
import queue
import re
import socket
import ssl
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from typing import Annotated, Self
from urllib.parse import urlsplit

from pydantic import ConfigDict, Field, SecretStr, model_validator

from regimex.contracts import MAX_INPUT_BYTES, StrictModel

from .jwt import IDENTITY_PATTERN, json_bytes, strict_json

_RESOLVERS = threading.BoundedSemaphore(16)


def _addresses(
    host: str, port: int, deadline: float
) -> list[tuple[int, int, int, tuple[object, ...]]]:
    """OS DNS calls have no timeout; cap outstanding daemon lookups and waiting."""
    if not _RESOLVERS.acquire(blocking=False):
        raise TimeoutError
    result: queue.Queue[list[tuple[int, int, int, tuple[object, ...]]] | OSError] = queue.Queue(1)

    def resolve() -> None:
        try:
            addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
            result.put(
                [
                    (family, kind, protocol, address)
                    for family, kind, protocol, _, address in addresses[:32]
                ]
            )
        except OSError, UnicodeError:
            result.put(OSError("Cannot resolve the configured classification service."))
        finally:
            _RESOLVERS.release()

    threading.Thread(target=resolve, daemon=True, name="fixture-gateway-dns").start()
    try:
        answer = result.get(timeout=max(0.001, deadline - time.monotonic()))
    except queue.Empty:
        raise TimeoutError from None
    if isinstance(answer, OSError):
        raise OSError("Cannot resolve the configured classification service.") from None
    return answer


class GatewaySettings(StrictModel):
    model_config = ConfigDict(hide_input_in_errors=True)

    source_namespace: Annotated[str, Field(pattern=f"^{IDENTITY_PATTERN}$")]
    regimex_origin: str
    integration_token: SecretStr
    allow_loopback_http: bool = False
    connect_timeout: Annotated[float, Field(gt=0, le=10)] = 3.0
    read_timeout: Annotated[float, Field(gt=0, le=15)] = 10.0
    total_timeout: Annotated[float, Field(gt=0, le=60)] = 30.0
    response_bytes: Annotated[int, Field(ge=1024, le=MAX_INPUT_BYTES)] = MAX_INPUT_BYTES

    @model_validator(mode="after")
    def safe_configuration(self) -> Self:
        try:
            parsed = urlsplit(self.regimex_origin)
            port = parsed.port
            host = parsed.hostname
            if (
                parsed.username is not None
                or parsed.password is not None
                or parsed.path not in {"", "/"}
                or parsed.query
                or parsed.fragment
                or not host
                or "\\" in self.regimex_origin
                or any(ord(char) <= 32 or ord(char) == 127 for char in self.regimex_origin)
            ):
                raise ValueError
            if parsed.scheme != "https" and not (
                parsed.scheme == "http"
                and self.allow_loopback_http
                and ipaddress.ip_address(host).is_loopback
            ):
                raise ValueError
            if port is not None and not 1 <= port <= 65535:
                raise ValueError
        except ValueError:
            raise ValueError("A safe configured Regimex origin is required.") from None
        if not re.fullmatch(r"[A-Za-z0-9._~-]{16,512}", self.integration_token.get_secret_value()):
            raise ValueError("Invalid backend integration credential.")
        return self


class GatewayError(Exception):
    def __init__(self, code: str, message: str, status: int, *, retryable: bool) -> None:
        super().__init__(message)
        self.code, self.message, self.status, self.retryable = code, message, status, retryable


@dataclass(frozen=True)
class UpstreamResponse:
    status: int
    content_type: str
    body: bytes


class RegimexClient:
    def __init__(self, settings: GatewaySettings) -> None:
        self.settings = settings

    def request(
        self,
        method: str,
        path: str,
        *,
        subject: str,
        run_id: str,
        body: bytes | None = None,
        idempotency_key: str | None = None,
    ) -> UpstreamResponse:
        if (
            method not in {"GET", "POST"}
            or not re.fullmatch(r"/api/v1/integration/[A-Za-z0-9/-]+", path)
            or not re.fullmatch(IDENTITY_PATTERN, subject)
            or not re.fullmatch(IDENTITY_PATTERN, run_id)
            or (body is not None and len(body) > MAX_INPUT_BYTES)
        ):
            raise ValueError("Invalid internal gateway request.")
        headers = {
            "Authorization": "Bearer " + self.settings.integration_token.get_secret_value(),
            "X-Regimex-Subject": subject,
            "X-Regimex-Run-ID": run_id,
            "Accept": "application/json, text/html",
            "Accept-Encoding": "identity",
        }
        if body is not None:
            headers["Content-Type"] = "application/json"
        if idempotency_key is not None:
            if not re.fullmatch(r"gateway-[a-f0-9]{64}", idempotency_key):
                raise ValueError("Invalid internal gateway idempotency key.")
            headers["Idempotency-Key"] = idempotency_key
        parsed = urlsplit(self.settings.regimex_origin)
        assert parsed.hostname is not None
        connection: http.client.HTTPConnection
        if parsed.scheme == "https":
            connection = http.client.HTTPSConnection(
                parsed.hostname,
                parsed.port,
                timeout=self.settings.connect_timeout,
                context=ssl.create_default_context(),
            )
        else:
            connection = http.client.HTTPConnection(
                parsed.hostname, parsed.port, timeout=self.settings.connect_timeout
            )
        deadline = time.monotonic() + self.settings.total_timeout
        transport: socket.socket | None = None
        response: http.client.HTTPResponse | None = None

        def abort_request() -> None:
            if transport is not None:
                with suppress(OSError):
                    transport.shutdown(socket.SHUT_RDWR)

        watchdog = threading.Timer(self.settings.total_timeout, abort_request)
        watchdog.daemon = True
        watchdog.start()
        try:
            addresses = _addresses(
                parsed.hostname,
                parsed.port or (443 if parsed.scheme == "https" else 80),
                min(deadline, time.monotonic() + self.settings.connect_timeout),
            )
            for family, kind, protocol, address in addresses:
                transport = socket.socket(family, kind, protocol)
                try:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError
                    transport.settimeout(min(self.settings.connect_timeout, remaining))
                    transport.connect(address)
                    if parsed.scheme == "https":
                        transport = ssl.create_default_context().wrap_socket(
                            transport,
                            server_hostname=parsed.hostname,
                        )
                    connection.sock = transport
                    break
                except OSError:
                    transport.close()
            else:
                raise OSError("Cannot connect to the configured classification service.")
            self._timeout(transport, deadline)
            connection.request(method, path, body=body, headers=headers)
            self._timeout(transport, deadline)
            response = connection.getresponse()
            if response.status not in {200, 201, 202}:
                # Untrusted error bodies and Location URLs are deliberately unused.
                self._status_error(response.status)
            if response.getheader("Content-Encoding", "identity").lower() != "identity":
                raise ValueError("Encoded upstream response.")
            lengths = response.headers.get_all("Content-Length", [])
            if len(lengths) > 1:
                raise ValueError("Ambiguous upstream size.")
            expected = int(lengths[0]) if lengths else None
            if expected is not None and not 0 <= expected <= self.settings.response_bytes:
                raise ValueError("Upstream response exceeds limit.")
            data = bytearray()
            while True:
                self._timeout(transport, deadline)
                block = response.read1(min(65536, self.settings.response_bytes + 1 - len(data)))
                if not block:
                    break
                data.extend(block)
                if len(data) > self.settings.response_bytes:
                    raise ValueError("Upstream response exceeds limit.")
            if expected is not None and expected != len(data):
                raise ValueError("Incomplete upstream response.")
            return UpstreamResponse(
                response.status, response.getheader("Content-Type", ""), bytes(data)
            )
        except GatewayError:
            raise
        except OSError, http.client.HTTPException, ValueError:
            raise GatewayError(
                "classification_unavailable",
                "Classification is temporarily unavailable. "
                "The simulator results remain available.",
                503,
                retryable=True,
            ) from None
        finally:
            watchdog.cancel()
            if response is not None:
                response.close()
            connection.close()

    def _timeout(self, transport: socket.socket | None, deadline: float) -> None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError
        # Keep the original transport: HTTPConnection may detach it on close headers.
        if transport is not None:
            transport.settimeout(min(self.settings.read_timeout, remaining))

    @staticmethod
    def _status_error(status: int) -> None:
        if status == 404:
            raise GatewayError(
                "not_found", "Analysis resource was not found.", 404, retryable=False
            )
        if status == 409:
            raise GatewayError(
                "classification_conflict",
                "Classification is not available in its current state.",
                409,
                retryable=False,
            )
        raise GatewayError(
            "classification_unavailable",
            "Classification is temporarily unavailable. The simulator results remain available.",
            503,
            retryable=True,
        )

    def json(
        self,
        method: str,
        path: str,
        *,
        subject: str,
        run_id: str,
        payload: object | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, object]:
        response = self.request(
            method,
            path,
            subject=subject,
            run_id=run_id,
            body=None if payload is None else json_bytes(payload),
            idempotency_key=idempotency_key,
        )
        try:
            if response.content_type.split(";", 1)[0].strip().lower() != "application/json":
                raise ValueError
            value = strict_json(response.body)
            if not isinstance(value, dict):
                raise ValueError
            return dict(value)
        except ValueError, UnicodeError, RecursionError:
            raise GatewayError(
                "invalid_classification_response",
                "The classification response could not be verified.",
                502,
                retryable=True,
            ) from None
