"""Bound HTTP intake before ASGI, including incomplete request headers.

Uvicorn's keep-alive timer bounds idle time between requests, not the time
allowed to send a request. These absolute deadlines are never refreshed by
individual bytes. Response generation has its own application limits.
"""

import asyncio
from typing import Literal

import h11
from uvicorn.protocols.http.h11_impl import H11Protocol

_REQUEST_TIMEOUT = b"HTTP/1.1 408 Request Timeout\r\nConnection: close\r\nContent-Length: 0\r\n\r\n"


def bounded_http_protocol(
    *, header_timeout_seconds: float, body_timeout_seconds: float
) -> type[H11Protocol]:
    """Build the service protocol without changing Uvicorn's global defaults."""
    if not 0 < header_timeout_seconds < float("inf"):
        raise ValueError("HTTP header timeout must be finite and positive")
    if not 0 < body_timeout_seconds < float("inf"):
        raise ValueError("HTTP body timeout must be finite and positive")

    class BoundedHTTPProtocol(H11Protocol):
        _intake_timer: asyncio.TimerHandle | None = None
        _intake_phase: Literal["headers", "body"] | None = None

        def connection_made(self, transport: asyncio.BaseTransport) -> None:
            if not isinstance(transport, asyncio.Transport):
                raise TypeError("HTTP requires a stream transport")
            super().connection_made(transport)
            self._arm_intake("headers")

        def connection_lost(self, exc: Exception | None) -> None:
            self._clear_intake()
            super().connection_lost(exc)

        def handle_events(self) -> None:
            super().handle_events()
            # Inspect h11's parsed request state, including pipelined requests
            # resumed by Uvicorn after the preceding response completes.
            if self.transport.is_closing():
                self._clear_intake()
            elif self.conn.their_state is h11.IDLE:
                self._arm_intake("headers")
            elif self.conn.their_state is h11.SEND_BODY:
                self._arm_intake("body")
            else:
                self._clear_intake()

        def _arm_intake(self, phase: Literal["headers", "body"]) -> None:
            if self._intake_phase == phase:
                return
            self._clear_intake()
            self._intake_phase = phase
            seconds = header_timeout_seconds if phase == "headers" else body_timeout_seconds
            self._intake_timer = self.loop.call_later(seconds, self._intake_expired)

        def _clear_intake(self) -> None:
            if self._intake_timer is not None:
                self._intake_timer.cancel()
            self._intake_timer = None
            self._intake_phase = None

        def _intake_expired(self) -> None:
            self._clear_intake()
            if self.transport.is_closing():
                return
            cycle = self.cycle
            # A response may already be streaming while its unread request body
            # is incomplete. Close it without injecting a second HTTP response.
            if (
                cycle is None or cycle.response_complete or not cycle.response_started
            ) and self.conn.our_state is not h11.DONE:
                self.transport.write(_REQUEST_TIMEOUT)
            if cycle is not None:
                cycle.disconnected = True
            self.transport.close()

    return BoundedHTTPProtocol
