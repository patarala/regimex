"""Two nodes with independent staging directories, one PostgreSQL schema and a real S3 prefix."""

import io
import os
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Barrier, Event, Thread
from typing import TYPE_CHECKING, BinaryIO
from uuid import uuid4

import pytest
from test_object_store_contract import cleanup_owned_test_prefix, digest, s3_configuration, source

from regimex.infrastructure.object_store import ObjectCapacityError, ObjectConflictError
from regimex.infrastructure.s3_object_store import S3ObjectStore, WriteReconciliation, _Client

if TYPE_CHECKING:
    from regimex.infrastructure.database import DatabaseRepository
    from regimex.infrastructure.maintenance import Connection


@dataclass
class Nodes:
    first: S3ObjectStore
    second: S3ObjectStore
    database: DatabaseRepository


@pytest.fixture
def nodes(tmp_path: Path) -> Iterator[Nodes]:
    psycopg = pytest.importorskip("psycopg")
    dsn = os.environ.get("REGIMEX_TEST_DATABASE_DSN")
    if not dsn:
        pytest.skip("set REGIMEX_TEST_DATABASE_DSN to a disposable PostgreSQL database")
    from regimex.infrastructure.database import DatabaseRepository
    from regimex.infrastructure.s3_coordination import S3Coordinator

    config = s3_configuration(tmp_path / "host-a").model_copy(update={"coordination": "postgres"})
    database = DatabaseRepository(dsn, schema="s3_cluster_" + uuid4().hex)
    database.migrate()
    first, second = (
        S3ObjectStore(
            config.model_copy(update={"lock_root": tmp_path / host}),
            coordinator=S3Coordinator(DatabaseRepository(dsn, schema=database.schema)),
            max_object_bytes=4,
            max_total_bytes=4,
        )
        for host in ("host-a", "host-b")
    )
    try:
        assert first.root != second.root and first.scope_id == second.scope_id
        yield Nodes(first, second, database)
    finally:
        cleanup_owned_test_prefix(first)
        with psycopg.connect(dsn, autocommit=True) as db:
            db.execute(
                psycopg.sql.SQL("DROP SCHEMA {} CASCADE").format(
                    psycopg.sql.Identifier(database.schema)
                )
            )


def test_two_uploads_compete_for_the_last_bytes(nodes: Nodes, tmp_path: Path) -> None:
    file = source(tmp_path, b"abcd")
    start = Barrier(2)

    def put(store: S3ObjectStore, key: str) -> str:
        start.wait(10)
        try:
            return store.put(key, file, expected_sha256=digest(b"abcd")).key
        except ObjectCapacityError:
            return "full"

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(put, store, key)
            for store, key in [(nodes.first, "left"), (nodes.second, "right")]
        ]
        results = [f.result(20) for f in futures]
    assert results.count("full") == 1
    versions = nodes.first.inventory()
    assert len(versions) == 1 and versions[0].size_bytes == 4
    known = nodes.second.stat(versions[0].key)
    assert nodes.first.put(known.key, file, expected_sha256=known.sha256) == known


def test_independent_uploads_can_fill_reserved_capacity_concurrently(
    nodes: Nodes, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checksum = digest(b"ab")
    file = source(tmp_path, b"ab")
    inside, release = Event(), Event()
    send = nodes.first._send_put

    def delayed(
        client: _Client,
        key: str,
        upload: BinaryIO,
        size: int,
        checksum: str,
        *,
        metadata: dict[str, str] | None = None,
    ) -> dict[str, object]:
        inside.set()
        assert release.wait(20)
        return send(client, key, upload, size, checksum, metadata=metadata)

    monkeypatch.setattr(nodes.first, "_send_put", delayed)
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(nodes.first.put, "first", file, expected_sha256=checksum)
        try:
            assert inside.wait(15)
            assert nodes.second.put("second", file, expected_sha256=checksum).size_bytes == 2
            with pytest.raises(ObjectCapacityError):
                nodes.second.put("third", file, expected_sha256=checksum)
        finally:
            release.set()
        assert first.result(10).size_bytes == 2
    assert nodes.second.capacity_report() == {
        "retained_versions": 2,
        "retained_payload_bytes": 4,
        "pending_writes": 0,
        "reserved_payload_bytes": 0,
    }


def test_lost_database_lock_cannot_overbook_an_inflight_upload(
    nodes: Nodes,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Remove the writer's database session while its provider request is outstanding."""
    pytest.importorskip("psycopg")
    from regimex.infrastructure.object_store import ObjectInfo

    first, second = nodes.first, nodes.second
    file = source(tmp_path, b"abcd")
    checksum = digest(b"abcd")
    starting, resume = Event(), Event()
    backend_pid = []
    original_put, original_send = first._put_reserved, first._send_put

    def remember(
        db: Connection,
        client: _Client,
        key: str,
        temporary: int,
        size: int,
        checksum: str,
        checkpoint: Callable[[], None] | None,
    ) -> ObjectInfo:
        backend_pid.append(db.info.backend_pid)
        return original_put(db, client, key, temporary, size, checksum, checkpoint)

    def paused(
        client: _Client,
        key: str,
        upload: BinaryIO,
        size: int,
        checksum: str,
        *,
        metadata: dict[str, str] | None = None,
    ) -> dict[str, object]:
        starting.set()
        assert resume.wait(20)
        return original_send(client, key, upload, size, checksum, metadata=metadata)

    monkeypatch.setattr(first, "_put_reserved", remember)
    monkeypatch.setattr(first, "_send_put", paused)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(first.put, "first", file, expected_sha256=checksum)
        try:
            assert starting.wait(15)
            with nodes.database.connection() as db:
                assert db.execute(
                    "SELECT pg_terminate_backend(%s, 5000)", (backend_pid[0],)
                ).fetchone() == (True,)
            with pytest.raises(ObjectCapacityError):
                second.put("competes", file, expected_sha256=checksum)
            with pytest.raises(ObjectConflictError, match="unconfirmed"):
                second.put("first", file, expected_sha256=checksum)
        finally:
            resume.set()
        with pytest.raises(OSError, match="coordination database"):
            pending.result(10)

    # The late request committed to S3. Its original owner cannot publish it.
    observed = second.stat("first")
    with pytest.raises(ObjectConflictError, match="Reconcile"):
        second.delete(observed)
    assert second.put("first", file, expected_sha256=checksum) == observed
    assert len(second.inventory()) == 1
    first.delete(observed)
    assert second.put("competes", file, expected_sha256=checksum).key == "competes"


def test_missing_uncertain_write_requires_provider_drain_to_release(
    nodes: Nodes,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from regimex.infrastructure.maintenance import ServicesStoppedAttestation

    file = source(tmp_path, b"abcd")
    checksum = digest(b"abcd")

    def disconnected(*args: object, **kwargs: object) -> dict[str, object]:
        raise OSError("connection lost before any response")

    monkeypatch.setattr(nodes.first, "_send_put", disconnected)
    with pytest.raises(OSError, match="connection lost"):
        nodes.first.put("first", file, expected_sha256=checksum)
    assert nodes.second.capacity_report() == {
        "retained_versions": 0,
        "retained_payload_bytes": 0,
        "pending_writes": 1,
        "reserved_payload_bytes": 4,
    }
    with pytest.raises(OSError, match="connection lost"):
        nodes.first.put("zero", source(tmp_path, b"", name="empty"), expected_sha256=digest(b""))
    monkeypatch.setattr("regimex.infrastructure.s3_object_store._RECONCILE_BATCH_SIZE", 1)
    attestation = ServicesStoppedAttestation(
        stopped_at=datetime.now(UTC),
        statement="Dedicated test",
        api_stopped=True,
        analysis_workers_stopped=True,
        administrative_workers_stopped=True,
        maintenance_stopped=True,
        garbage_collection_stopped=True,
        exclusive_object_access=True,
    )
    assert nodes.second.reconcile_pending_writes(
        attestation, provider_drained=False
    ) == WriteReconciliation(0, 0, 2, "first")
    with pytest.raises(ObjectCapacityError):
        nodes.second.put("second", file, expected_sha256=checksum)
    assert nodes.second.reconcile_pending_writes(
        attestation, provider_drained=True, after_key="first"
    ) == WriteReconciliation(0, 1, 1, None)
    assert nodes.second.reconcile_pending_writes(
        attestation, provider_drained=True
    ) == WriteReconciliation(0, 1, 0, None)
    assert nodes.second.put("second", file, expected_sha256=checksum).size_bytes == 4


def test_retryable_provider_response_does_not_transmit_a_write_token_twice(
    nodes: Nodes, tmp_path: Path
) -> None:
    tokens: list[str] = []

    class BusyProvider(BaseHTTPRequestHandler):
        def do_PUT(self) -> None:
            tokens.append(self.headers["x-amz-meta-regimex-write-id"])
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(503)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, format: str, *args: object) -> None:
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), BusyProvider) as server:
        thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05})
        thread.start()
        try:
            config = nodes.first.config.model_copy(
                update={
                    "endpoint_url": f"http://127.0.0.1:{server.server_port}",
                    "ca_bundle": None,
                    "allow_loopback_http": True,
                    "max_attempts": 3,
                    "lock_root": tmp_path / "retry-client",
                }
            )
            store = S3ObjectStore(
                config, max_object_bytes=4, max_total_bytes=4, coordinator=nodes.first.coordinator
            )
            token = str(uuid4())
            with pytest.raises(OSError, match="transmitted twice"):
                store._send_put(
                    store._sdk().client,
                    "payload",
                    io.BytesIO(b"ab"),
                    2,
                    digest(b"ab"),
                    metadata={"regimex-write-id": token},
                )
            assert tokens == [token]
        finally:
            server.shutdown()
            thread.join()


def test_namespace_and_limit_mismatch_fail_before_provider_access(
    nodes: Nodes, tmp_path: Path
) -> None:
    from regimex.infrastructure.object_store import ObjectIntegrityError

    file = source(tmp_path, b"abcd")
    nodes.first.put("bound", file, expected_sha256=digest(b"abcd"))
    first = nodes.first
    for path, prefix, limit in (
        ("different-limit", first.config.prefix, 8),
        ("different-prefix", first.config.prefix + "-other", 4),
    ):
        other = S3ObjectStore(
            first.config.model_copy(update={"prefix": prefix, "lock_root": tmp_path / path}),
            max_object_bytes=4,
            max_total_bytes=limit,
            coordinator=first.coordinator,
        )
        with pytest.raises(ObjectIntegrityError, match="namespaces or capacity"):
            other.check()


def test_materialization_and_exact_deletion_on_different_nodes(
    nodes: Nodes,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from regimex.infrastructure.object_store import ObjectInfo

    first, second = nodes.first, nodes.second
    file = source(tmp_path, b"abcd")
    known = first.put("read-me", file, expected_sha256=digest(b"abcd"))
    inside, deleting, release = Event(), Event(), Event()
    old_read = first._read

    def slow(
        client: _Client,
        key: str,
        *,
        destination: int | None = None,
        version_id: str | None = None,
        checkpoint: Callable[[], None] | None = None,
    ) -> ObjectInfo:
        if destination is not None:
            inside.set()
            assert release.wait(10)
        return old_read(
            client, key, destination=destination, version_id=version_id, checkpoint=checkpoint
        )

    monkeypatch.setattr(first, "_read", slow)

    def remove() -> None:
        deleting.set()
        second.delete(known)

    saved = tmp_path / "download"
    with ThreadPoolExecutor(max_workers=2) as pool:
        reader = pool.submit(
            first.materialize, known.key, saved, expected_size_bytes=4, expected_sha256=known.sha256
        )
        try:
            assert inside.wait(10)
            deletion = pool.submit(remove)
            assert deleting.wait(10)
            with pytest.raises(TimeoutError):
                deletion.result(0.15)
        finally:
            release.set()
        assert reader.result(10) == known
        deletion.result(10)
    assert saved.read_bytes() == b"abcd"
    with pytest.raises(FileNotFoundError):
        second.stat(known.key)
