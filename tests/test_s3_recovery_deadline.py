"""Exercise offline recovery against an S3 peer that stops replying mid-request."""

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Event, Thread
from urllib.parse import parse_qs, urlsplit

import pytest
from pydantic import SecretStr
from test_recovery import DatabaseFactory, MinimalBackup, stopped
from test_recovery import databases as databases
from test_recovery import minimal as minimal
from test_recovery import tools as tools

from regimex.application.recovery import backup_service, restore_service
from regimex.infrastructure.maintenance import maintenance_lock
from regimex.infrastructure.s3_config import S3Config
from regimex.infrastructure.s3_object_store import S3ObjectStore
from regimex.recovery_contracts import PostgresTools, RecoveryLimits


@dataclass
class StalledRequest:
    entered: Event = field(default_factory=Event)
    disconnected: Event = field(default_factory=Event)


@contextmanager
def unresponsive_s3(stage: str, payload: bytes) -> Iterator[tuple[str, StalledRequest]]:
    request = StalledRequest()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def stall(self) -> None:
            request.entered.set()
            self.connection.settimeout(12)
            try:
                if self.connection.recv(1) == b"":
                    request.disconnected.set()
            except ConnectionResetError:
                request.disconnected.set()
            except TimeoutError:
                pass
            self.close_connection = True

        def reply(self, body: bytes) -> None:
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def object_headers(self) -> None:
            self.send_response(200)
            self.send_header("x-amz-version-id", "fixture-version")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()

        def do_HEAD(self) -> None:
            if stage == "download":
                self.object_headers()
            else:
                self.send_error(404)

        def do_GET(self) -> None:
            query = parse_qs(urlsplit(self.path).query, keep_blank_values=True)
            if "versioning" in query:
                self.reply(
                    b"<VersioningConfiguration><Status>Enabled</Status></VersioningConfiguration>"
                )
            elif "versions" in query and stage == "inventory":
                self.stall()
            elif "versions" in query:
                self.reply(
                    b"<ListVersionsResult><IsTruncated>false</IsTruncated></ListVersionsResult>"
                )
            elif "uploads" in query:
                self.reply(
                    b"<ListMultipartUploadsResult><IsTruncated>false</IsTruncated>"
                    b"</ListMultipartUploadsResult>"
                )
            elif stage == "download" and "versionId" in query:
                self.object_headers()
                self.wfile.write(payload[:1])
                self.stall()
            else:
                self.send_error(404)

        def do_PUT(self) -> None:
            # The server has accepted every byte. The client still has to wait for
            # the version ID before it can verify the object or publish the restore.
            if stage != "upload":
                self.send_error(405)
                return
            if self.rfile.read(int(self.headers["Content-Length"])) != payload:
                self.send_error(400)
                return
            self.stall()

        def log_message(self, format: str, *args: object) -> None:
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05})
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}", request
        finally:
            server.shutdown()
            thread.join()


@pytest.mark.parametrize("stage", ["download", "inventory", "upload"])
def test_offline_deadline_closes_stalled_s3_request(
    stage: str,
    minimal: MinimalBackup,
    databases: DatabaseFactory,
    tools: PostgresTools,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import psycopg

    pytest.importorskip("boto3")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "deadline-test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "deadline-test")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    (info,) = minimal.backup.manifest.objects
    payload = (minimal.directory / "objects" / info.sha256).read_bytes()
    database = minimal.source if stage == "download" else databases(minimal.source.schema)
    destination = tmp_path / "interrupted-backup"
    limits = RecoveryLimits(timeout_seconds=3)
    with unresponsive_s3(stage, payload) as (endpoint, request):
        config = S3Config(
            endpoint_url=endpoint,
            allow_loopback_http=True,
            region="us-east-1",
            bucket="test-bucket",
            prefix="data",
            lock_root=tmp_path / "coordination",
            read_timeout_seconds=10,
            max_attempts=1,
        )
        store = S3ObjectStore(config, max_object_bytes=len(payload), max_total_bytes=len(payload))
        started = time.monotonic()
        with pytest.raises(TimeoutError, match="wall-time"):
            if stage == "download":
                backup_service(
                    SecretStr(database._dsn),
                    database.schema,
                    store,
                    destination,
                    tools=tools,
                    attestation=stopped(),
                    limits=limits,
                )
            else:
                restore_service(
                    SecretStr(database._dsn),
                    store,
                    minimal.directory,
                    expected_manifest_sha256=minimal.backup.manifest_sha256,
                    tools=tools,
                    attestation=stopped(),
                    limits=limits,
                )
        assert time.monotonic() - started < limits.timeout_seconds + 1.5
        assert request.entered.is_set()
        assert request.disconnected.wait(1)
        # A subsequent maintenance command must be able to acquire both locks.
        with psycopg.connect(database._dsn) as connection:
            maintenance_lock(connection, database.schema)
        S3ObjectStore(config, max_object_bytes=len(payload), max_total_bytes=len(payload))

    if stage == "download":
        assert database.readiness()
        assert not (destination / "manifest.json").exists()
        assert not (destination / "objects" / info.sha256).exists()
    else:
        assert not database.readiness()
