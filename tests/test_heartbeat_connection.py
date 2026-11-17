"""A heartbeat's client deadline survives a responsive TCP socket with no replies."""

import os
import select
import socket
import time
from collections.abc import Iterator
from contextlib import contextmanager
from threading import Event, Thread

import pytest

psycopg = pytest.importorskip("psycopg")

from psycopg.conninfo import conninfo_to_dict  # noqa: E402

from regimex.infrastructure.heartbeat_connection import HeartbeatConnection  # noqa: E402


@pytest.fixture
def database_dsn() -> str:
    value = os.environ.get("REGIMEX_TEST_DATABASE_DSN")
    if not value:
        pytest.skip("set REGIMEX_TEST_DATABASE_DSN to a disposable PostgreSQL database")
    return value


@contextmanager
def dropping_proxy(dsn: str) -> Iterator[tuple[int, Event]]:
    parameters = conninfo_to_dict(dsn)
    host = str(parameters.get("host") or "127.0.0.1")
    port = int(str(parameters.get("port") or "5432"))
    if host.startswith("/") or "," in host:
        pytest.skip("the stalled-transport fixture requires one TCP database endpoint")
    stop, drop = Event(), Event()
    errors: list[Exception] = []

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        listener.settimeout(5)

        def relay() -> None:
            try:
                with (
                    listener.accept()[0] as client,
                    socket.create_connection((host, port), 5) as upstream,
                ):
                    peers = {client: upstream, upstream: client}
                    while not stop.is_set():
                        readable, _, _ = select.select(list(peers), [], [], 0.05)
                        for incoming in readable:
                            block = incoming.recv(65536)
                            if not block:
                                return
                            # Consume packets so TCP remains healthy. Socket
                            # keepalives alone cannot detect this missing reply.
                            if not drop.is_set():
                                peers[incoming].sendall(block)
            except Exception as exc:
                errors.append(exc)

        worker = Thread(target=relay, name="heartbeat-test-proxy", daemon=True)
        worker.start()
        try:
            yield listener.getsockname()[1], drop
        finally:
            stop.set()
            worker.join(timeout=6)
        assert not worker.is_alive()
        assert not errors


def test_missing_server_reply_does_not_leave_a_blocked_heartbeat(
    database_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(HeartbeatConnection, "timeout_seconds", 0.3)
    with dropping_proxy(database_dsn) as (port, drop):
        connection = HeartbeatConnection.connect(
            database_dsn,
            host="127.0.0.1",
            port=port,
            sslmode="disable",
            connect_timeout=2,
            autocommit=True,
        )
        started = time.monotonic()
        with pytest.raises(psycopg.OperationalError, match="timed out"), connection:
            assert connection.execute("SELECT 1").fetchone() == (1,)
            drop.set()
            connection.execute("SELECT 1")
        assert time.monotonic() - started < 3
        assert connection.closed


def test_deadline_covers_commit_and_discards_expired_transaction(
    database_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(HeartbeatConnection, "timeout_seconds", 0.1)
    connection = HeartbeatConnection.connect(database_dsn, connect_timeout=2)
    with pytest.raises(psycopg.OperationalError, match="timed out"), connection:
        assert connection.execute("SELECT 1").fetchone() == (1,)
        time.sleep(0.15)
    assert connection.closed
