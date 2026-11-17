"""Client-side deadlines for short lease transactions, including network stalls."""

import time

import psycopg
from psycopg.abc import PQGen


class HeartbeatConnection(psycopg.Connection[tuple[object, ...]]):
    """Bound the complete transaction without duplicating repository SQL.

    Server statement timeouts cannot bound a connection that stops delivering
    server responses. Psycopg's wait timeout covers that client-side wait too.
    A timed-out connection is discarded before context-manager rollback, which
    would otherwise make another round trip through the stalled transport.
    """

    timeout_seconds: float = 15.0
    _deadline: float | None = None

    def wait[T](self, gen: PQGen[T], interval: float = 0.1, timeout: float | None = None) -> T:
        if self._deadline is None:
            self._deadline = time.monotonic() + self.timeout_seconds
        remaining = self._deadline - time.monotonic()
        if remaining <= 0:
            self.close()
            raise psycopg.OperationalError("Heartbeat database transaction timed out.")
        limit = remaining if timeout is None else min(remaining, timeout)
        try:
            return super().wait(gen, interval=interval, timeout=limit)
        except psycopg.OperationalError:
            self.close()
            if time.monotonic() >= self._deadline:
                raise psycopg.OperationalError(
                    "Heartbeat database transaction timed out."
                ) from None
            raise
