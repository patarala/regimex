"""Structured operational logs with no request headers, bodies or query strings."""

import json
import logging
import os
import re
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from urllib.parse import unquote, urlsplit

if TYPE_CHECKING:
    from fastapi import FastAPI, Request, Response

    from regimex.service_settings import ServiceSettings

_URL_CREDENTIALS = re.compile(r"(?i)([a-z][a-z0-9+.-]*://)[^\s/@]+@")
_QUOTED_VALUE = r"(?:\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*')"
_AUTHORIZATION = re.compile(
    r"(?i)(\bauthorization[\"']?\s*[:=]\s*)(" + _QUOTED_VALUE + r"|[^\r\n,;]+)"
)
_ASSIGNED_SECRET = re.compile(
    r"(?i)(\b(?:password|(?:aws_)?secret_access_key|(?:aws_)?access_key_id|"
    r"(?:aws_)?session_token|token)[\"']?\s*[:=]\s*)(" + _QUOTED_VALUE + r"|[^\s,;]+)"
)
_BEARER = re.compile(r"(?i)(\b(?:bearer|basic)\s+)[^\s,;'\"]+")


class RedactingJSONFormatter(logging.Formatter):
    def __init__(self, secrets: tuple[str, ...]) -> None:
        super().__init__()
        variants = set(secrets)
        for value in secrets:
            variants.add(unquote(value))
            if "://" in value:
                try:
                    password = urlsplit(value).password
                except ValueError:
                    password = None
                if password:
                    variants.update((password, unquote(password)))
            for match in _ASSIGNED_SECRET.finditer(value):
                assigned = match[2]
                if len(assigned) > 1 and assigned[0] in {"'", '"'}:
                    assigned = re.sub(r"\\(.)", r"\1", assigned[1:-1])
                variants.add(assigned)
        self.secrets = tuple(
            sorted((value for value in variants if len(value) >= 4), key=len, reverse=True)
        )

    def _redact(self, value: str) -> str:
        for secret in self.secrets:
            value = value.replace(secret, "[redacted]")
        value = _URL_CREDENTIALS.sub(r"\1[redacted]@", value)
        value = _AUTHORIZATION.sub(r"\1[redacted]", value)
        value = _ASSIGNED_SECRET.sub(r"\1[redacted]", value)
        return _BEARER.sub(r"\1[redacted]", value)

    def format(self, record: logging.LogRecord) -> str:
        message = self._redact(record.getMessage())
        item: dict[str, object] = {
            "time": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": self._redact(record.name)[:128],
            "message": message[:4096],
        }
        # Arbitrary LogRecord extras and tracebacks can contain connection
        # arguments or request payloads. Preserve only implemented public fields.
        for name in (
            "action",
            "queue",
            "worker_id",
            "operation_id",
            "method",
            "route",
            "status",
            "duration_ms",
        ):
            value = getattr(record, name, None)
            if isinstance(value, (str, int, float, bool)):
                item[name] = self._redact(value) if isinstance(value, str) else value
        if record.exc_info is not None and record.exc_info[0] is not None:
            item["exception_type"] = record.exc_info[0].__name__
        return json.dumps(item, ensure_ascii=True, allow_nan=False, separators=(",", ":"))


def configure_service_logging(settings: ServiceSettings) -> None:
    secrets = [settings.database_dsn.get_secret_value()]
    for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
        value = os.environ.get(name)
        if value:
            secrets.append(value)
    handler = logging.StreamHandler()
    handler.setFormatter(RedactingJSONFormatter(tuple(secrets)))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(logging.INFO)
    logging.captureWarnings(True)
    # Raw Uvicorn access logs include arbitrary URL queries. Request metadata is
    # recorded by our application middleware using the matched route template.
    logging.getLogger("uvicorn.access").disabled = True


def install_http_logging(app: FastAPI) -> None:
    """Log route templates and timing, never caller-supplied URLs or identities."""
    logger = logging.getLogger("regimex.http")

    @app.middleware("http")
    async def request_log(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        start = time.monotonic()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            route = getattr(request.scope.get("route"), "path", "[unmatched]")
            method = (
                request.method
                if request.method in {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}
                else "OTHER"
            )
            logger.info(
                "HTTP request completed",
                extra={
                    "action": "request",
                    "method": method,
                    "route": route,
                    "status": status,
                    "duration_ms": round((time.monotonic() - start) * 1000, 3),
                },
            )
