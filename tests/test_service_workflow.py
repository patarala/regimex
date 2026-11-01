"""Real saved baseline analysis, durable results and repeated worker delivery."""

import asyncio
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from subprocess import Popen
from threading import Event
from typing import Any, cast
from uuid import uuid4

import pytest
from pydantic import SecretStr

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")
pytest.importorskip("plotly")
psycopg = pytest.importorskip("psycopg")
pytest.importorskip("procrastinate")
pytest.importorskip("alembic")

from psycopg import Connection
from test_application_training import source
from test_baseline_backend import bundle, rehash

from regimex.application import analysis as analysis_worker
from regimex.application.analysis_computation import snapshot_inputs
from regimex.application.service import AnalysisService
from regimex.computation_worker import supervise_child
from regimex.exports import ExportPayloadV2, seal_export
from regimex.infrastructure.database import DatabaseRepository
from regimex.infrastructure.object_store import LocalObjectStore, ObjectInfo
from regimex.model_registry import ArchitectureRegistry, InferenceRuntime, baseline_descriptor
from regimex.operations import Operation
from regimex.service_contracts import (
    Forbidden,
    NotFound,
    Principal,
    Role,
    StaleAttempt,
)
from regimex.service_settings import ServiceSettings
from regimex.storage import json_bytes


@pytest.fixture
def repository() -> Iterator[DatabaseRepository]:
    dsn = os.environ.get("REGIMEX_TEST_DATABASE_DSN")
    if not dsn:
        pytest.skip("set REGIMEX_TEST_DATABASE_DSN to a disposable PostgreSQL database")
    schema = "regimex_workflow_" + uuid4().hex
    repository = DatabaseRepository(dsn, schema=schema)
    repository.migrate()
    try:
        with repository.queue_app.open():
            yield repository
    finally:
        with psycopg.connect(dsn, autocommit=True) as connection:
            connection.execute(
                psycopg.sql.SQL("DROP SCHEMA {} CASCADE").format(psycopg.sql.Identifier(schema))
            )


@pytest.fixture
def service(repository: DatabaseRepository, tmp_path: Path) -> Iterator[AnalysisService]:
    value = AnalysisService(
        repository,
        LocalObjectStore(
            tmp_path / "objects", max_object_bytes=16 * 1024**2, max_total_bytes=128 * 1024**2
        ),
        ArchitectureRegistry((baseline_descriptor(),)),
        tmp_path / "work",
        heartbeat_seconds=0.02,
    )
    try:
        yield value
    finally:
        # Cache entries are private and read-only during use. Make this disposable
        # test directory removable without changing production retention policy.
        for root, directories, files in os.walk(tmp_path / "work"):
            Path(root).chmod(0o700)
            for name in directories:
                (Path(root) / name).chmod(0o700)
            for name in files:
                (Path(root) / name).chmod(0o600)


def principal(
    repository: DatabaseRepository, owner: str = "fixture-owner", role: Role = "user"
) -> Principal:
    token = repository.create_token(owner, role)
    return repository.authenticate(token)


def published_model(service: AnalysisService, tmp_path: Path) -> None:
    _, _, _, checksum = bundle(tmp_path)
    service.register_bundle(
        principal(service.repository, "administrator", "admin"),
        tmp_path / "bundle",
        manifest_sha256=checksum,
        inference_runtime=InferenceRuntime(threads=1),
        activate=True,
    )


def test_signed_zero_and_exponent_result_report_and_preview_survive_real_worker(
    service: AnalysisService, tmp_path: Path
) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx2")
    from fastapi.testclient import TestClient

    from regimex.api import create_app

    published_model(service, tmp_path)
    token = service.repository.create_token("numeric-owner", "user")
    owner = service.repository.authenticate(token)
    value = source(2000).export.model_dump(mode="json", exclude={"content_sha256"})
    value["time_grid"]["origin"] = -0.0
    value["instruments"][0]["initial_price"] = 1e20
    export = seal_export(ExportPayloadV2.model_validate_json(json.dumps(value)))
    snapshot = service.create_snapshot(owner, export)
    queued = service.submit_analysis(owner, snapshot.snapshot_id, idempotency_key="numeric-result")
    completed = service.run_analysis(queued.analysis_id)
    assert completed is not None and completed.state == "succeeded"
    result = service.get_result(owner, queued.analysis_id)
    assert hashlib.sha256(json_bytes(result)).hexdigest() == completed.result_sha256
    report = service.read_report(owner, queued.analysis_id)
    headers = {"Authorization": "Bearer " + token}
    with TestClient(create_app(service)) as client:
        for suffix in ("", "/result", "/report", "/report/preview"):
            response = client.get(
                f"/api/v1/analyses/{queued.analysis_id}" + suffix, headers=headers
            )
            assert response.status_code == 200, response.text[:500]
            if suffix == "/result":
                value = response.json()
                assert json_bytes(value) == json_bytes(result)
                assert repr(value["time_grid"]["origin"]) == "-0.0"
                assert type(value["instruments"][0]["initial_price"]) is float
                assert value["instruments"][0]["initial_price"] == 1e20
            elif suffix.startswith("/report"):
                assert response.content == report


def test_real_baseline_result_survives_restart_and_redelivery(
    service: AnalysisService, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    published_model(service, tmp_path)
    owner = principal(service.repository)
    export = source(2000).export
    snapshot = service.create_snapshot(owner, export)
    assert service.create_snapshot(owner, export) == snapshot
    queued = service.submit_analysis(owner, snapshot.snapshot_id, idempotency_key="first")
    completed = service.run_analysis(queued.operation.operation_id)
    assert completed is not None and completed.state == "succeeded" and completed.attempt == 1
    result = service.get_result(owner, queued.analysis_id)
    summary = service.get_analysis_summary(owner, queued.analysis_id)
    assert summary.report_available and summary.result_available
    report = service.get_report_metadata(owner, queued.analysis_id)
    assert report is not None and report.result_sha256 == completed.result_sha256
    report_bytes = service.read_report(owner, queued.analysis_id)
    assert (
        report_bytes.startswith(b"<!doctype html>")
        and b"whole requested completed snapshot" in report_bytes
    )
    assert result["source"] == export.source.model_dump(mode="json")
    assert result["time_grid"] == export.time_grid.model_dump(mode="json")
    assert result["class_names"] == ["sideways", "growth", "decline"]
    assert "reference_kind" not in result and "seed" not in result
    sequence = cast(list[dict[str, object]], result["sequences"])[0]
    assert len(cast(list[object], sequence["probabilities"])) == 192
    assert cast(list[dict[str, int]], sequence["intervals"])[0]["start"] == 0
    assert cast(list[dict[str, int]], sequence["intervals"])[-1]["stop"] == 192
    predictor, _ = analysis_worker.load_registered_predictor(
        service, service.repository.get_model("active"), worker_id="test-reload"
    )
    np.testing.assert_array_equal(
        np.asarray(sequence["probabilities"]),
        predictor.predict(snapshot_inputs(export), runtime=InferenceRuntime(threads=1)),
    )
    restarted = AnalysisService(
        service.repository, service.store, service.registry, service.work_root
    )
    assert restarted.get_result(owner, queued.analysis_id) == result
    assert restarted.read_report(owner, queued.analysis_id) == report_bytes
    assert restarted.run_analysis(queued.operation.operation_id) is None
    assert restarted.submit_analysis(
        owner, snapshot.snapshot_id, idempotency_key="first"
    ) == restarted.get_analysis(owner, queued.analysis_id)
    scoped = Principal(
        owner_id=owner.owner_id,
        role="integration",
        token_id=uuid4(),
        source_namespace=export.source.source_id,
        run_scope=export.source.run_id,
    )
    assert restarted.read_report(scoped, queued.analysis_id) == report_bytes

    def denied_materialize(*args: object, **kwargs: object) -> ObjectInfo:
        raise AssertionError("denied scoped report must not access storage")

    monkeypatch.setattr(service.store, "materialize", denied_materialize)
    for denied in (
        scoped.model_copy(update={"run_scope": "wrong-run"}),
        scoped.model_copy(update={"source_namespace": "wrong-source"}),
        scoped.model_copy(update={"owner_id": "wrong-subject"}),
    ):
        with pytest.raises(NotFound):
            restarted.get_report_metadata(denied, queued.analysis_id)
        with pytest.raises(NotFound):
            restarted.read_report(denied, queued.analysis_id)


def test_cached_bundle_is_not_downloaded_for_every_prediction(
    service: AnalysisService, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    published_model(service, tmp_path)
    owner = principal(service.repository)
    snapshot = service.create_snapshot(owner, source(2001).export)
    seen: list[str] = []
    original = service.store.materialize

    def observe(
        key: str, destination: Path, *, expected_sha256: str, expected_size_bytes: int
    ) -> ObjectInfo:
        seen.append(key)
        return original(
            key,
            destination,
            expected_sha256=expected_sha256,
            expected_size_bytes=expected_size_bytes,
        )

    monkeypatch.setattr(service.store, "materialize", observe)
    for key in ("one", "two"):
        queued = service.submit_analysis(owner, snapshot.snapshot_id, idempotency_key=key)
        result = service.run_analysis(queued.operation.operation_id)
        assert result is not None and result.state == "succeeded"
    assert seen == [snapshot.object.key, snapshot.object.key]


@dataclass
class _ChildProbe:
    processes: list[Popen[bytes]] = field(default_factory=list)
    roots: list[Path] = field(default_factory=list)
    launched: Event = field(default_factory=Event)

    def wait_until_ready(self) -> None:
        assert self.launched.wait(5), "computation child did not launch"
        deadline = time.monotonic() + 5
        while not (self.roots[0] / "test-ready").exists():
            assert self.processes[0].poll() is None, "child exited before reaching its blocked call"
            assert time.monotonic() < deadline, "computation child did not reach its blocked call"
            time.sleep(0.01)

    def cleanup(self) -> None:
        for child in self.processes:
            if child.poll() is None:
                with suppress(ProcessLookupError):
                    os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=5)


def _probe_child(monkeypatch: pytest.MonkeyPatch, behavior: str) -> _ChildProbe:
    # Only the computation body is substituted. PostgreSQL claims, heartbeats,
    # process supervision, resource accounting and completion remain real.
    script = """
import signal
import sys
import time
from pathlib import Path
root = Path(sys.argv[1])
signal.signal(signal.SIGTERM, signal.SIG_IGN)
if sys.argv[2] == 'scratch':
    for index in range(2):
        (root / f'large-output-{index}').write_bytes(b'x' * (600 * 1024))
(root / 'test-ready').touch()
while True:
    time.sleep(0.01)
"""
    original = subprocess.Popen
    probe = _ChildProbe()

    def launch(command: list[str], **kwargs: Any) -> Popen[bytes]:
        assert command[command.index("--task") + 1] == "analysis"
        assert "REGIMEX_TEST_DATABASE_DSN" not in kwargs["env"]
        root = Path(command[command.index("--root") + 1])
        child = cast(
            Popen[bytes],
            original([sys.executable, "-c", script, str(root), behavior], **kwargs),
        )
        probe.roots.append(root)
        probe.processes.append(child)
        probe.launched.set()
        return child

    monkeypatch.setattr(subprocess, "Popen", launch)
    return probe


@pytest.mark.parametrize("stop_reason", ["cancel", "timeout"])
def test_blocking_child_is_stopped_and_the_worker_can_process_the_next_analysis(
    service: AnalysisService,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stop_reason: str,
) -> None:
    published_model(service, tmp_path)
    owner = principal(service.repository)
    snapshot = service.create_snapshot(owner, source(2002).export)
    queued = service.submit_analysis(owner, snapshot.snapshot_id, idempotency_key="cancel-running")
    service.time_budget_seconds = 1 if stop_reason == "timeout" else 5
    with ThreadPoolExecutor(max_workers=1) as worker:
        with monkeypatch.context() as patch:
            probe = _probe_child(patch, "block")
            try:
                future = worker.submit(service.run_analysis, queued.operation.operation_id)
                probe.wait_until_ready()
                running = service.get_operation(owner, queued.operation.operation_id)
                assert running.attempt_token is not None
                if stop_reason == "cancel":
                    service.cancel_operation(owner, queued.operation.operation_id)
                result = future.result(timeout=10)
                assert result is not None
                if stop_reason == "cancel":
                    assert result.state == "cancelled" and result.failure is None
                else:
                    assert result.state == "failed" and result.failure is not None
                    assert result.failure.code == "capacity_limit" and not result.failure.retryable
                assert probe.processes[0].returncode == -signal.SIGKILL
                assert not probe.roots[0].exists()
            finally:
                probe.cleanup()
        summary = service.get_analysis_summary(owner, queued.analysis_id)
        assert not summary.result_available and not summary.report_available
        assert service.get_analysis(owner, queued.analysis_id).result is None
        assert service.get_report_metadata(owner, queued.analysis_id) is None
        service.time_budget_seconds = 120
        following = service.submit_analysis(
            owner, snapshot.snapshot_id, idempotency_key="after-blocked-child"
        )
        completed = worker.submit(service.run_analysis, following.operation.operation_id).result(
            timeout=30
        )
        assert completed is not None and completed.state == "succeeded"
        report = service.get_report_metadata(owner, following.analysis_id)
        assert report is not None
        with pytest.raises(StaleAttempt):
            service.repository.complete_analysis(
                queued.operation.operation_id,
                running.attempt_token,
                service.get_result(owner, following.analysis_id),
                report=report,
            )
        assert service.get_analysis(owner, queued.analysis_id).result is None
        assert service.run_analysis(queued.operation.operation_id) is None


@pytest.mark.parametrize("limit", ["memory", "scratch", "input_scratch"])
def test_analysis_resource_exhaustion_has_no_committed_output_or_retry(
    service: AnalysisService,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    limit: str,
) -> None:
    published_model(service, tmp_path)
    owner = principal(service.repository)
    snapshot = service.create_snapshot(owner, source(2017).export)
    queued = service.submit_analysis(owner, snapshot.snapshot_id, idempotency_key=limit)
    probe = None
    if limit == "memory":
        # The actual child installs RLIMIT_AS before importing NumPy or a backend.
        service.memory_budget_bytes = 32 * 1024**2
    elif limit == "scratch":
        service.scratch_budget_bytes = 1024**2
        probe = _probe_child(monkeypatch, "scratch")
    else:
        service.scratch_budget_bytes = 1

        def forbidden_launch(*args: object, **kwargs: object) -> None:
            pytest.fail("over-budget inputs launched an analysis child")

        monkeypatch.setattr(subprocess, "Popen", forbidden_launch)
    try:
        result = service.run_analysis(queued.operation.operation_id)
        assert result is not None and result.state == "failed" and result.failure is not None
        assert result.failure.code == "capacity_limit" and not result.failure.retryable
        summary = service.get_analysis_summary(owner, queued.analysis_id)
        assert not summary.result_available and not summary.report_available
        assert service.get_analysis(owner, queued.analysis_id).result is None
        assert service.get_report_metadata(owner, queued.analysis_id) is None
        assert not list(service.work_root.glob("analysis-*"))
        if probe is not None:
            assert len(probe.processes) == 1
            assert probe.processes[0].returncode == -signal.SIGKILL
        assert service.run_analysis(queued.operation.operation_id) is None
    finally:
        if probe is not None:
            probe.cleanup()


@pytest.mark.parametrize(
    "transient_error", [OSError, psycopg.OperationalError, psycopg.errors.SerializationFailure]
)
def test_transient_failure_requeues_then_exact_model_succeeds(
    service: AnalysisService,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    transient_error: type[Exception],
) -> None:
    published_model(service, tmp_path)
    owner = principal(service.repository)
    snapshot = service.create_snapshot(owner, source(2003).export)
    queued = service.submit_analysis(owner, snapshot.snapshot_id, idempotency_key="retry")
    original = supervise_child

    def unavailable(*args: object, **kwargs: object) -> object:
        raise transient_error("private-provider-credential-must-not-be-persisted")

    monkeypatch.setattr(analysis_worker, "supervise_child", unavailable)
    first = service.run_analysis(queued.operation.operation_id)
    assert first is not None and first.state == "queued" and first.attempt == 1
    assert first.failure is None
    monkeypatch.setattr(analysis_worker, "supervise_child", original)
    second = service.run_analysis(queued.operation.operation_id)
    assert second is not None and second.state == "succeeded" and second.attempt == 2
    assert second.model == queued.operation.model
    assert "private-provider" not in second.model_dump_json()


def test_corrupt_cached_model_fails_without_a_committed_result(
    service: AnalysisService, tmp_path: Path
) -> None:
    published_model(service, tmp_path)
    owner = principal(service.repository)
    snapshot = service.create_snapshot(owner, source(2004).export)
    queued = service.submit_analysis(owner, snapshot.snapshot_id, idempotency_key="corruption")
    record = service.repository.get_model("active")
    cache = (
        service.work_root
        / "model-cache"
        / f"{record.bundle_sha256}-{record.inference_policy_sha256}"
    )
    weights = next(path for path in cache.iterdir() if path.suffix == ".npz")
    weights.chmod(0o600)
    weights.write_bytes(b"corrupt")
    failed = service.run_analysis(queued.operation.operation_id)
    assert failed is not None and failed.state == "failed" and failed.failure is not None
    assert failed.failure.code == "incompatible_artifact"
    assert service.get_analysis(owner, queued.analysis_id).result is None


def test_invalid_numeric_codec_cannot_register_or_activate(
    service: AnalysisService, tmp_path: Path
) -> None:
    bundle(tmp_path)
    root = tmp_path / "bundle"
    weights = next(path for path in root.iterdir() if path.suffix == ".npz")
    weights.write_bytes(b"not-a-numeric-model")
    checksum = rehash(root)
    admin = principal(service.repository, "administrator", "admin")
    with pytest.raises(ValueError):
        service.register_bundle(
            admin,
            root,
            manifest_sha256=checksum,
            inference_runtime=InferenceRuntime(threads=1),
            activate=True,
        )
    assert service.list_models(admin) == ()
    assert service.repository.get_active_model() is None


def test_snapshot_source_and_owner_checks_precede_reading(service: AnalysisService) -> None:
    owner = principal(service.repository)
    snapshot = service.create_snapshot(owner, source(2005).export)
    other = principal(service.repository, "other-owner")
    with pytest.raises(NotFound):
        service.read_snapshot(other, snapshot.snapshot_id)
    token = service.repository.create_token(
        "integration", "integration", source_namespace="another-source"
    )
    integration = service.authenticate(token)
    with pytest.raises(Forbidden):
        service.create_snapshot(integration, source(2006).export)


def test_actual_queue_executes_registered_baseline(
    service: AnalysisService, tmp_path: Path
) -> None:
    from regimex.service_worker import create_worker_app

    published_model(service, tmp_path)
    owner = principal(service.repository)
    snapshot = service.create_snapshot(owner, source(2007).export)
    queued = service.submit_analysis(owner, snapshot.snapshot_id, idempotency_key="actual-worker")
    # The actual worker opens its own PostgreSQL connections, independent of the
    # intake transaction. No eager callback stands in for queue delivery.
    settings = ServiceSettings(
        database_dsn=SecretStr(service.repository._dsn),
        database_schema=service.repository.schema,
        object_root=tmp_path / "objects",
        work_root=service.work_root,
    )
    app = create_worker_app(settings, service)

    async def execute() -> None:
        async with app.open_async():
            await asyncio.wait_for(
                app.run_worker_async(
                    queues=["analysis"],
                    concurrency=1,
                    wait=False,
                    listen_notify=False,
                    install_signal_handlers=False,
                    shutdown_graceful_timeout=1,
                ),
                timeout=30,
            )

    asyncio.run(execute())
    result = service.get_analysis(owner, queued.analysis_id)
    assert result.operation.state == "succeeded" and result.result is not None
    with service.repository.connection() as connection:
        assert connection.execute(
            "SELECT status FROM procrastinate_jobs WHERE task_name='regimex.analysis'"
        ).fetchall() == [("succeeded",)]


def test_unavailable_worker_model_and_cache_budget_fail_explicitly(
    service: AnalysisService, tmp_path: Path
) -> None:
    published_model(service, tmp_path)
    owner = principal(service.repository)
    snapshot = service.create_snapshot(owner, source(2008).export)
    unavailable = baseline_descriptor().model_copy(
        update={
            "backend_factory": None,
            "worker_probe": None,
            "unavailable_reason": "The required isolated computation runtime is unavailable.",
        }
    )
    service.registry = ArchitectureRegistry((unavailable,))
    queued = service.submit_analysis(owner, snapshot.snapshot_id, idempotency_key="unavailable")
    failed = service.run_analysis(queued.operation.operation_id)
    assert failed is not None and failed.failure is not None
    assert failed.state == "failed" and failed.failure.code == "unsupported_configuration"
    assert service.get_analysis(owner, queued.analysis_id).result is None
    service.registry = ArchitectureRegistry((baseline_descriptor(),))
    service.cache_budget_bytes = 1
    queued = service.submit_analysis(owner, snapshot.snapshot_id, idempotency_key="cache-full")
    failed = service.run_analysis(queued.operation.operation_id)
    assert failed is not None and failed.failure is not None
    assert failed.failure.code == "capacity_limit"


def test_registration_replay_uses_verified_existing_objects(
    service: AnalysisService, tmp_path: Path
) -> None:
    published_model(service, tmp_path)
    admin = principal(service.repository, "administrator", "admin")
    existing = service.repository.get_model("active")
    repeated = service.register_bundle(
        admin,
        tmp_path / "bundle",
        manifest_sha256=existing.bundle_sha256,
        inference_runtime=existing.inference_runtime,
    )
    assert repeated == existing
    assert service.list_models(admin) == (existing,)


def test_child_process_death_is_fenced_before_real_queue_redelivery(
    service: AnalysisService, tmp_path: Path
) -> None:
    from regimex.service_worker import create_worker_app, recover_operations

    published_model(service, tmp_path)
    owner = principal(service.repository)
    snapshot = service.create_snapshot(owner, source(2009).export)
    queued = service.submit_analysis(owner, snapshot.snapshot_id, idempotency_key="worker-died")
    env = {
        **os.environ,
        "REGIMEX_CRASH_DSN": service.repository._dsn,
        "REGIMEX_CRASH_SCHEMA": service.repository.schema,
        "REGIMEX_CRASH_OPERATION": str(queued.operation.operation_id),
    }
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os; from uuid import UUID; "
            "from regimex.infrastructure.database import DatabaseRepository; "
            "r=DatabaseRepository(os.environ['REGIMEX_CRASH_DSN'],"
            "schema=os.environ['REGIMEX_CRASH_SCHEMA']); "
            "o=r.claim_operation(UUID(os.environ['REGIMEX_CRASH_OPERATION'])); "
            "assert o is not None and o.state == 'running'; os._exit(17)",
        ],
        env=env,
        capture_output=True,
        timeout=15,
        check=False,
    )
    assert child.returncode == 17
    abandoned = service.get_operation(owner, queued.operation.operation_id)
    assert abandoned.state == "running" and abandoned.attempt == 1
    settings = ServiceSettings(
        database_dsn=SecretStr(service.repository._dsn),
        database_schema=service.repository.schema,
        object_root=tmp_path / "objects",
        work_root=service.work_root,
        heartbeat_seconds=1,
        stale_seconds=3,
    )
    time.sleep(3.1)
    app = create_worker_app(settings, service)

    async def execute() -> None:
        async with app.open_async():
            assert await recover_operations(settings, service, app) == 1
            await asyncio.wait_for(
                app.run_worker_async(
                    queues=["analysis"],
                    concurrency=1,
                    wait=False,
                    listen_notify=False,
                    install_signal_handlers=False,
                    shutdown_graceful_timeout=1,
                ),
                timeout=30,
            )

    asyncio.run(execute())
    completed = service.get_analysis(owner, queued.analysis_id)
    assert completed.operation.state == "succeeded" and completed.operation.attempt == 2
    assert completed.operation.attempt_tokens[0] != completed.operation.attempt_tokens[1]
    assert completed.result is not None
    assert service.repository.recover_operations(datetime.now(UTC)) == ()
    # Both the original and recovered delivery were consumed. A delivery after
    # commit cannot overwrite the result or allocate another application attempt.
    assert service.run_analysis(queued.operation.operation_id) is None
    assert service.get_analysis(owner, queued.analysis_id) == completed


def test_cancellation_after_report_upload_cannot_publish_result_or_report(
    service: AnalysisService, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    published_model(service, tmp_path)
    owner = principal(service.repository)
    snapshot = service.create_snapshot(owner, source(2015).export)
    queued = service.submit_analysis(
        owner, snapshot.snapshot_id, idempotency_key="cancel-rendering"
    )
    original_put = service.store.put
    uploaded: list[ObjectInfo] = []

    def put(key: str, source_path: Path, *, expected_sha256: str) -> ObjectInfo:
        info = original_put(key, source_path, expected_sha256=expected_sha256)
        if key.startswith("reports/"):
            uploaded.append(info)
            service.cancel_operation(owner, queued.operation.operation_id)
        return info

    monkeypatch.setattr(service.store, "put", put)
    completed = service.run_analysis(queued.operation.operation_id)
    assert completed is not None and completed.state == "cancelled"
    summary = service.get_analysis_summary(owner, queued.analysis_id)
    assert not summary.result_available and not summary.report_available
    assert service.get_analysis(owner, queued.analysis_id).result is None
    assert service.get_report_metadata(owner, queued.analysis_id) is None
    assert len(uploaded) == 1 and service.store.stat(uploaded[0].key) == uploaded[0]
    with service.repository.connection() as connection:
        assert (
            connection.execute(
                "SELECT 1 FROM regimex_objects WHERE object_key=%s", (uploaded[0].key,)
            ).fetchone()
            is None
        )
    intruder = principal(service.repository, "different-owner")

    def forbidden_materialize(*args: object, **kwargs: object) -> ObjectInfo:
        raise AssertionError("unauthorized report request must not access object storage")

    monkeypatch.setattr(service.store, "materialize", forbidden_materialize)
    with pytest.raises(NotFound):
        service.read_report(intruder, queued.analysis_id)


def test_report_result_and_object_registration_rollback_then_retry_together(
    service: AnalysisService, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    published_model(service, tmp_path)
    owner = principal(service.repository)
    snapshot = service.create_snapshot(owner, source(2016).export)
    queued = service.submit_analysis(owner, snapshot.snapshot_id, idempotency_key="report-atomic")
    original_save = service.repository._save
    uncommitted_keys: list[str] = []

    def fail_final_save(connection: Connection[tuple[object, ...]], operation: Operation) -> None:
        if operation.state == "succeeded":
            row = connection.execute(
                "SELECT object_key FROM regimex_analysis_reports WHERE analysis_id=%s",
                (queued.analysis_id,),
            ).fetchone()
            assert row is not None and isinstance(row[0], str)
            uncommitted_keys.append(row[0])
            assert connection.execute(
                "SELECT result IS NOT NULL FROM regimex_analyses WHERE analysis_id=%s",
                (queued.analysis_id,),
            ).fetchone() == (True,)
            assert connection.execute(
                "SELECT 1 FROM regimex_objects WHERE object_key=%s", (row[0],)
            ).fetchone() == (1,)
            raise OSError("deliberate interruption before the final operation commit")
        original_save(connection, operation)

    monkeypatch.setattr(service.repository, "_save", fail_final_save)
    failed = service.run_analysis(queued.operation.operation_id)
    assert failed is not None and failed.state == "queued" and failed.attempt == 1
    summary = service.get_analysis_summary(owner, queued.analysis_id)
    assert not summary.result_available and not summary.report_available
    assert service.get_analysis(owner, queued.analysis_id).result is None
    assert len(uncommitted_keys) == 1
    with service.repository.connection() as connection:
        assert (
            connection.execute(
                "SELECT 1 FROM regimex_objects WHERE object_key=%s", (uncommitted_keys[0],)
            ).fetchone()
            is None
        )
        assert (
            connection.execute(
                "SELECT 1 FROM regimex_analysis_reports WHERE analysis_id=%s", (queued.analysis_id,)
            ).fetchone()
            is None
        )
    assert service.store.stat(uncommitted_keys[0]).size_bytes > 0
    monkeypatch.setattr(service.repository, "_save", original_save)
    completed = service.run_analysis(queued.operation.operation_id)
    assert completed is not None and completed.state == "succeeded" and completed.attempt == 2
    assert service.get_analysis_summary(owner, queued.analysis_id).report_available
    assert service.read_report(owner, queued.analysis_id).startswith(b"<!doctype html>")
