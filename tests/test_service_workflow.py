"""Real saved baseline analysis, durable results and repeated worker delivery."""

import asyncio
import errno
import hashlib
import json
import logging
import os
import signal
import stat
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from subprocess import Popen
from threading import Event
from typing import Any, Never, cast
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
from test_analysis_report import embedded_data
from test_application_training import source
from test_baseline_backend import bundle, rehash

from regimex.application import analysis as analysis_worker
from regimex.application.analysis_computation import snapshot_inputs
from regimex.application.deletion import DeletionService
from regimex.application.service import AnalysisService
from regimex.computation_worker import supervise_child
from regimex.exports import ExportPayloadV2, seal_export
from regimex.infrastructure.analysis_timings import queue_activity
from regimex.infrastructure.database import DatabaseRepository
from regimex.infrastructure.deletion_repository import DeletionRepository
from regimex.infrastructure.object_store import LocalObjectStore, ObjectInfo
from regimex.infrastructure.resource_lifecycle import owner_payload_lock
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


def test_schema_probe_remains_ready_when_active_model_metadata_is_invalid(
    service: AnalysisService, tmp_path: Path
) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx2")
    from fastapi.testclient import TestClient

    from regimex.api import create_app

    published_model(service, tmp_path)
    with TestClient(create_app(service), raise_server_exceptions=False) as client:
        assert client.get("/health/models").status_code == 200
        with service.repository.connection() as connection:
            connection.execute(
                "UPDATE regimex_models SET document = document - 'inference_runtime' "
                "WHERE model_version_id = (SELECT model_version_id FROM regimex_deployment)"
            )
        ready = client.get("/health/ready")
        assert ready.status_code == 200 and ready.json()["database"] is True
        model_health = client.get("/health/models")
        assert model_health.status_code == 503
        assert model_health.json() == {
            "deployment_configured": False,
            "worker_capability": "not_probed",
        }


def test_doctor_restores_saved_model_in_a_child_and_detects_corrupted_storage(
    service: AnalysisService, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from regimex.service_diagnostics import diagnose

    published_model(service, tmp_path)
    model = service.repository.get_model("active")
    settings = ServiceSettings(
        database_dsn=SecretStr(service.repository._dsn),
        database_schema=service.repository.schema,
        object_root=tmp_path / "objects",
        work_root=tmp_path / "doctor-work",
    )

    def forbidden_in_process_restore(*args: object, **kwargs: object) -> None:
        raise AssertionError("model restoration must happen in the bounded child process")

    monkeypatch.setattr(analysis_worker, "load_registered_predictor", forbidden_in_process_restore)
    first = diagnose(settings)
    checks = {check.name: check for check in first.checks}
    assert checks["storage"].status == checks["model_load"].status == "passed"
    assert first.model_probe_scope == "local_isolated_process"
    assert checks["analysis_queue"].status == "failed"
    assert not first.healthy
    assert not tuple(settings.work_root.glob("doctor-*"))
    assert service.repository.get_active_model() == model
    with service.repository.connection() as connection:
        assert connection.execute("SELECT count(*) FROM regimex_operations").fetchone() == (0,)
    # Corrupt the authoritative object after a successful restore. A new doctor
    # must inspect it again rather than trust a successful model cache.
    key = model.manifest.key
    committed = tmp_path / "objects" / key
    committed.chmod(0o600)
    committed.write_bytes(b"corrupted model: secret-placeholder")
    second = diagnose(settings)
    failed = {check.name: check for check in second.checks}
    assert failed["model_load"].status == "failed"
    assert "secret-placeholder" not in second.model_dump_json()
    assert service.repository.get_active_model() == model


def test_doctor_model_probe_honors_its_deadline_while_waiting_for_storage(
    service: AnalysisService, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from regimex import service_diagnostics
    from regimex.infrastructure import object_store
    from regimex.infrastructure.object_store import ObjectStore
    from regimex.service_contracts import ModelRecord

    published_model(service, tmp_path)
    settings = ServiceSettings(
        database_dsn=SecretStr(service.repository._dsn),
        database_schema=service.repository.schema,
        object_root=tmp_path / "objects",
        work_root=tmp_path / "doctor-work",
        analysis_time_budget_seconds=1,
    )
    probe = service_diagnostics._model_probe

    def probe_during_write(
        settings: ServiceSettings, store: ObjectStore, model: ModelRecord
    ) -> None:
        assert isinstance(store, LocalObjectStore)
        with store._locked_root():
            probe(settings, store, model)

    monkeypatch.setattr(object_store, "_LOCK_TIMEOUT_SECONDS", 3)
    monkeypatch.setattr(service_diagnostics, "_model_probe", probe_during_write)
    checks = {check.name: check for check in service_diagnostics.diagnose(settings).checks}
    assert checks["model_load"].status == "failed"
    assert checks["model_load"].message == "The model probe exceeded its time limit."
    assert not tuple(settings.work_root.glob("doctor-*"))


def test_report_quota_fails_explicitly_before_upload_and_success_charges_exact_bytes(
    service: AnalysisService, tmp_path: Path
) -> None:
    from regimex.infrastructure.storage_quotas import StorageQuotaRepository

    published_model(service, tmp_path)
    owner = principal(service.repository)
    snapshot = service.create_snapshot(owner, source(500).export)
    size = snapshot.object.size_bytes
    service.storage_quotas = StorageQuotaRepository(
        service.repository, max_owner_storage_bytes=size, max_integration_storage_bytes=size
    )
    queued = service.submit_analysis(owner, snapshot.snapshot_id, idempotency_key="full-storage")
    failed = service.run_analysis(queued.analysis_id)
    assert failed is not None and failed.state == "failed" and failed.failure is not None
    assert failed.failure.code == "capacity_limit"
    assert "storage limit" in failed.failure.message
    assert not failed.failure.retryable
    assert not any(item.key.startswith("reports/") for item in service.store.inventory())
    assert service.storage_usage(owner).reserved_bytes == 0

    service.storage_quotas = StorageQuotaRepository(service.repository)
    queued = service.submit_analysis(owner, snapshot.snapshot_id, idempotency_key="more-storage")
    completed = service.run_analysis(queued.analysis_id)
    assert completed is not None and completed.state == "succeeded"
    usage = service.storage_usage(owner)
    report = service.repository.get_analysis_report(owner, queued.analysis_id)
    assert report is not None
    assert usage.used_bytes == size + report.object.size_bytes
    assert usage.reserved_bytes == 0


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
            elif suffix == "/report":
                assert response.content == report
            elif suffix == "/report/preview":
                assert len(response.content) < len(report)
                assert json_bytes(embedded_data(response.content)) == json_bytes(
                    embedded_data(report)
                )


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


def test_analysis_and_job_keep_their_own_ids_across_recovery_and_deletion(
    service: AnalysisService, tmp_path: Path
) -> None:
    published_model(service, tmp_path)
    database = service.repository
    owner = principal(database)
    snapshot = service.create_snapshot(owner, source(2000).export)
    intake = service.submit_analysis(owner, snapshot.snapshot_id, idempotency_key="distinct-ids")
    operation_id = intake.operation.operation_id
    analysis_id = uuid4()
    # The analysis primary key and its job foreign key are independent.
    with database.connection() as connection:
        connection.execute(
            "UPDATE regimex_analyses SET analysis_id=%s WHERE operation_id=%s",
            (analysis_id, operation_id),
        )
        connection.execute(
            "UPDATE regimex_operations SET document=jsonb_set(document, '{resource_id}', "
            "to_jsonb(%s::text)) WHERE operation_id=%s",
            (str(analysis_id), operation_id),
        )
    scoped = Principal(
        owner_id=owner.owner_id,
        role="integration",
        token_id=uuid4(),
        source_namespace=snapshot.source_id,
        run_scope=snapshot.run_id,
    )
    assert service.get_analysis(owner, analysis_id).operation.operation_id == operation_id
    assert service.get_operation(scoped, operation_id).resource_id == str(analysis_id)
    assert database.claim_operation(operation_id) is not None
    (recovered,) = database.recover_operations(stale_after=timedelta())
    assert recovered.operation_id == operation_id and recovered.state == "queued"
    completed = service.run_analysis(operation_id)
    assert completed is not None and completed.state == "succeeded"
    assert service.get_result(scoped, analysis_id)["analysis_id"] == str(analysis_id)
    assert service.cancel_operation(scoped, operation_id) == completed
    assert queue_activity(database, [analysis_id])[analysis_id].operation_id == operation_id

    deletions = DeletionService(DeletionRepository(database), service.store)
    preview = deletions.preview(owner, "analysis", analysis_id)
    assert (
        deletions.apply(owner, preview.plan_id, expected_sha256=preview.sha256).state == "completed"
    )
    for actor in (owner, scoped):
        with pytest.raises(NotFound):
            service.get_analysis(actor, analysis_id)
        with pytest.raises(NotFound):
            service.get_operation(actor, operation_id)
        with pytest.raises(NotFound):
            service.cancel_operation(actor, operation_id)
    assert database.get_operation_internal(operation_id) == completed


def test_repeated_analysis_materializes_only_the_snapshot_in_each_attempt(
    service: AnalysisService, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    published_model(service, tmp_path)
    owner = principal(service.repository)
    snapshot = service.create_snapshot(owner, source(2001).export)
    seen: list[tuple[str, Path]] = []
    original = service.store.materialize

    def observe(
        key: str,
        destination: Path,
        *,
        expected_sha256: str,
        expected_size_bytes: int,
        checkpoint: Callable[[], None] | None = None,
    ) -> ObjectInfo:
        seen.append((key, destination))
        return original(
            key,
            destination,
            expected_sha256=expected_sha256,
            expected_size_bytes=expected_size_bytes,
            checkpoint=checkpoint,
        )

    monkeypatch.setattr(service.store, "materialize", observe)
    for key in ("one", "two"):
        queued = service.submit_analysis(owner, snapshot.snapshot_id, idempotency_key=key)
        result = service.run_analysis(queued.operation.operation_id)
        assert result is not None and result.state == "succeeded"
        ((object_key, destination),) = seen
        assert object_key == snapshot.object.key
        assert destination.name == "export.json"
        assert destination.parent.name.startswith(f"analysis-{queued.analysis_id}-")
        assert not destination.parent.exists()
        seen.clear()


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
                    assert "time limit" in result.failure.message
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


def test_cache_usage_counts_nested_files_and_propagates_directory_read_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    nested = tmp_path / "weights"
    nested.mkdir()
    (nested / "state").write_bytes(b"1234")
    (tmp_path / "config").write_bytes(b"12")
    assert analysis_worker._cached_bundle_bytes(tmp_path) == 6
    scan = os.scandir

    def fail_read(path: str) -> Any:
        if Path(path) == nested:
            raise OSError(errno.EIO, "cannot read cached weights")
        return scan(path)

    monkeypatch.setattr(os, "scandir", fail_read)
    with pytest.raises(OSError, match="cannot read cached weights"):
        analysis_worker._cached_bundle_bytes(tmp_path)


def test_cache_usage_rejects_directory_links(tmp_path: Path) -> None:
    nested = tmp_path / "weights"
    nested.mkdir()
    (nested / "link").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="regular files and directories"):
        analysis_worker._cached_bundle_bytes(tmp_path)


def test_cache_only_publishes_a_fully_read_only_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    (incoming / "weights").write_bytes(b"weights")
    destination = tmp_path / "cached"
    chmod = Path.chmod

    def fail_chmod(path: Path, mode: int) -> None:
        if path.is_dir() and mode == 0o500:
            raise OSError(errno.EIO, "cannot set directory permissions")
        chmod(path, mode)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "chmod", fail_chmod)
        with pytest.raises(OSError, match="cannot set directory permissions"):
            analysis_worker._publish_cached_bundle(incoming, destination)
    assert not destination.exists()
    analysis_worker._publish_cached_bundle(incoming, destination)
    assert stat.S_IMODE(destination.stat().st_mode) == 0o500
    assert stat.S_IMODE((destination / "weights").stat().st_mode) == 0o400
    destination.chmod(0o700)


@pytest.mark.parametrize("disk_full", [False, True])
def test_worker_persists_safe_reasons_and_only_retries_transient_storage_failures(
    service: AnalysisService,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    disk_full: bool,
) -> None:
    published_model(service, tmp_path)
    owner = principal(service.repository)
    snapshot = service.create_snapshot(owner, source(2020).export)
    queued = service.submit_analysis(owner, snapshot.snapshot_id, idempotency_key="storage-error")

    def unavailable(*args: object, **kwargs: object) -> object:
        raise OSError(errno.ENOSPC if disk_full else errno.EIO, "private storage path and token")

    monkeypatch.setattr(analysis_worker, "supervise_child", unavailable)
    attempts = 1 if disk_full else queued.operation.max_attempts
    for attempt in range(1, attempts + 1):
        operation = service.run_analysis(queued.operation.operation_id)
        assert operation is not None and operation.attempt == attempt
        assert operation.state == ("failed" if attempt == attempts else "queued")
    persisted = service.get_analysis(owner, queued.analysis_id)
    assert persisted.operation.failure is not None
    failure = persisted.operation.failure
    assert failure.code == ("capacity_limit" if disk_full else "infrastructure_failure")
    assert failure.retryable is not disk_full
    assert ("disk space" if disk_full else "service error") in failure.message
    assert "private" not in persisted.model_dump_json()
    assert persisted.result is None
    assert service.run_analysis(queued.operation.operation_id) is None


def test_report_publication_retry_reuses_uploaded_bytes_with_a_full_quota(
    service: AnalysisService, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from regimex.infrastructure.storage_quotas import StorageQuotaRepository

    published_model(service, tmp_path)
    owner = principal(service.repository)
    snapshot = service.create_snapshot(owner, source(2020).export)
    analysis = service.submit_analysis(owner, snapshot.snapshot_id, idempotency_key="report-retry")

    def interrupt_publication(*args: object, **kwargs: object) -> None:
        raise psycopg.OperationalError("connection lost at publication")

    with monkeypatch.context() as patch:
        patch.setattr(service.repository, "complete_analysis", interrupt_publication)
        first = service.run_analysis(analysis.analysis_id)
    assert first is not None and first.state == "queued"
    assert service.get_report_metadata(owner, analysis.analysis_id) is None
    usage = service.storage_usage(owner)
    assert usage.used_bytes == snapshot.object.size_bytes and usage.reserved_bytes > 0
    budget = usage.used_bytes + usage.reserved_bytes
    service.storage_quotas = StorageQuotaRepository(
        service.repository, max_owner_storage_bytes=budget, max_integration_storage_bytes=budget
    )

    second = service.run_analysis(analysis.analysis_id)
    assert second is not None and second.state == "succeeded" and second.attempt == 2
    assert service.get_report_metadata(owner, analysis.analysis_id) is not None
    after_retry = service.storage_usage(owner)
    assert (after_retry.used_bytes, after_retry.reserved_bytes) == (budget, 0)


def test_report_write_retries_after_owner_storage_lock_is_released(
    service: AnalysisService, tmp_path: Path
) -> None:
    published_model(service, tmp_path)
    owner = principal(service.repository)
    snapshot = service.create_snapshot(owner, source(2020).export)
    analysis = service.submit_analysis(owner, snapshot.snapshot_id, idempotency_key="busy-storage")

    with owner_payload_lock(service.repository, owner.owner_id):
        first = service.run_analysis(analysis.analysis_id)
    assert first is not None and first.state == "queued" and first.attempt == 1
    assert service.get_report_metadata(owner, analysis.analysis_id) is None
    second = service.run_analysis(analysis.analysis_id)
    assert second is not None and second.state == "succeeded" and second.attempt == 2


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
    service: AnalysisService, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    from regimex.application.load_timings import collect_timings
    from regimex.service_logging import RedactingJSONFormatter
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

    with caplog.at_level(logging.INFO, logger="regimex.analysis"):
        asyncio.run(execute())
    result = service.get_analysis(owner, queued.analysis_id)
    assert result.operation.state == "succeeded" and result.result is not None
    log = tmp_path / "worker.log"
    formatter = RedactingJSONFormatter(())
    log.write_text("\n".join(formatter.format(record) for record in caplog.records))
    jobs = tmp_path / "jobs.jsonl"
    jobs.write_text(json.dumps({"phase": "serial", "analysis_id": str(queued.analysis_id)}))
    output = tmp_path / "timings.json"
    assert collect_timings(service.repository, jobs, [log], output)
    report = json.loads(output.read_bytes())
    (timing,) = report["analyses"]
    assert timing["events"] == ["deferred", "started", "succeeded"]
    assert 0 < timing["backend_predict_ms"] <= timing["prediction_ms"]
    assert timing["queue_wait_ms"] >= 0
    assert timing["worker_occupancy_ms"] >= timing["child_process_ms"]
    assert all(timing[stage] > 0 for stage in ("model_load_ms", "prediction_ms", "report_ms"))
    assert report["phases"]["serial"]["median_ms"]["prediction"] == timing["prediction_ms"]


def test_analysis_reports_rendering_during_render_and_verification_afterward(
    service: AnalysisService, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from regimex.application import analysis_computation, report

    published_model(service, tmp_path)
    owner = principal(service.repository)
    snapshot = service.create_snapshot(owner, source(2007).export)
    queued = service.submit_analysis(owner, snapshot.snapshot_id, idempotency_key="progress")
    operation_id = queued.operation.operation_id
    stages = []
    render = report.render_analysis_report
    read_output = analysis_worker._read_output

    def run_child(
        payload: dict[str, Any], root: Path, *, heartbeat: Callable[[], None], **kwargs: object
    ) -> dict[str, Any]:
        heartbeat()
        stages.append(service.repository.get_operation(owner, operation_id).stage)

        def observe_render(*args: Any, **kwargs: Any) -> report.RenderedReport:
            # Poll at the render boundary even on machines where rendering is faster than a tick.
            heartbeat()
            stages.append(service.repository.get_operation(owner, operation_id).stage)
            return render(*args, **kwargs)

        monkeypatch.setattr(report, "render_analysis_report", observe_render)
        return analysis_computation.compute(payload, root)

    def verify(path: Path, limit: int, digest: str) -> bytes:
        stages.append(service.repository.get_operation(owner, operation_id).stage)
        return read_output(path, limit, digest)

    monkeypatch.setattr(analysis_worker, "supervise_child", run_child)
    monkeypatch.setattr(analysis_worker, "_read_output", verify)
    completed = service.run_analysis(operation_id)
    assert completed is not None and completed.state == "succeeded"
    assert stages == ["predicting", "rendering", "verifying", "verifying"]


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


@pytest.mark.parametrize("already_registered", (False, True))
def test_verified_external_fit_is_visible_in_training_and_replays_without_uploads(
    service: AnalysisService,
    tmp_path: Path,
    already_registered: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("httpx2")
    from fastapi.testclient import TestClient

    from regimex.api import create_app
    from regimex.application.admin_service import AdminService
    from regimex.external_evaluations import ExternalEvaluation
    from regimex.external_latency import ExternalLatencyEvidence, LatencyAtLength
    from regimex.infrastructure.admin_repository import AdminRepository
    from regimex.operations import LaunchSource
    from regimex.service_contracts import Conflict

    database = service.repository
    token = database.create_token("researcher", "admin")
    admin = database.authenticate(token)
    other = principal(database, "another-researcher", "admin")
    admin_repository = AdminRepository(database)
    runs = AdminService(
        service, admin_repository, ServiceSettings(database_dsn=SecretStr(database._dsn))
    )
    _, fit, _, checksum = bundle(tmp_path)
    runtime = InferenceRuntime(threads=1)
    root = tmp_path / "bundle"
    if already_registered:
        service.register_bundle(admin, root, manifest_sha256=checksum, inference_runtime=runtime)
    else:

        def fail_metrics(*_args: object) -> Never:
            raise RuntimeError("Interrupted while recording history")

        with monkeypatch.context() as patch:
            patch.setattr(
                "regimex.infrastructure.admin_repository.store_training_metrics", fail_metrics
            )
            with pytest.raises(RuntimeError, match="recording history"):
                service.register_bundle(
                    admin,
                    root,
                    manifest_sha256=checksum,
                    inference_runtime=runtime,
                    record_training=True,
                )
        assert runs.list_runs(admin, kind="training") == () and service.list_models(admin) == ()
    registration = LaunchSource(channel="cli", command="regimex service register-model")
    first_source = None if already_registered else registration
    model = service.register_bundle(
        admin,
        root,
        manifest_sha256=checksum,
        inference_runtime=runtime,
        record_training=True,
        display_name="Offline baseline",
        launch_source=first_source,
    )
    (run,) = runs.list_runs(admin, kind="training")
    assert run.operation.attempt == 0 and run.operation.stage == "imported"
    assert admin_repository.get_training_run_for_model(other, model.model_version_id) is None
    assert runs.list_datasets(admin) == ()
    assert runs.preview_weight_deletion(admin, run.run_id).blockers == ()
    with database.connection() as connection:
        assert connection.execute("SELECT count(*) FROM regimex_deliveries").fetchone() == (0,)
    evidence = ExternalEvaluation(
        kind="native_final",
        bundle_sha256=checksum,
        report_sha256="b" * 64,
        freeze_sha256="a" * 64,
        protocol_sha256="c" * 64,
        candidate="offline",
        sample_count=12,
        interval_macro_f1=0.5,
        passed=False,
    )
    latency = ExternalLatencyEvidence(
        manifest_sha256=checksum,
        report_sha256="d" * 64,
        timings=(LatencyAtLength(steps=64, predict_decode_ms=3.0),),
    )
    inventory = service.store.inventory()
    service.register_bundle(
        admin,
        root,
        manifest_sha256=checksum,
        inference_runtime=runtime,
        record_training=True,
        launch_source=registration,
        external_evaluations=(evidence,),
        external_latency=latency,
    )
    assert (
        service.register_bundle(
            admin,
            root,
            manifest_sha256=checksum,
            inference_runtime=runtime,
            record_training=True,
        )
        == model
    )
    assert service.store.inventory() == inventory
    with pytest.raises(Conflict, match="external report"):
        service.register_bundle(
            admin,
            root,
            manifest_sha256=checksum,
            inference_runtime=runtime,
            record_training=True,
            external_evaluations=(evidence.model_copy(update={"interval_macro_f1": 0.8}),),
        )
    with pytest.raises(NotFound):
        service.register_bundle(
            other,
            root,
            manifest_sha256=checksum,
            inference_runtime=runtime,
            record_training=True,
        )
    with TestClient(create_app(service, admin_service=runs)) as client:
        headers = {"Authorization": f"Bearer {token}"}
        path = f"/api/v1/training-runs/{run.run_id}"
        (item,) = client.get("/api/v1/training-runs", headers=headers).json()["items"]
        assert item == client.get(path, headers=headers).json()
        assert item["job"]["launch_source"] == (first_source.model_dump() if first_source else None)
        assert item["display_name"] == "Offline baseline" and "statistics" not in item
        training = item["training"]
        assert training["origin"] == "external" and training["evidence"]["run_id"] == "fixture-run"
        assert training["weights"]["sha256"] == model.members[training["weights"]["path"]].sha256
        assert training["external_evaluations"] == [evidence.model_dump(mode="json")]
        assert training["external_latency"] == latency.model_dump(mode="json")
        metrics = client.get(path + "/metrics", headers=headers).json()
        assert metrics["validation_segment_macro_f1"] == fit.validation_score
        assert metrics["history"] and metrics["selected_seed"] == fit.selected_seed
    if already_registered:
        assert (
            service.activate_model(admin, model.model_version_id, expected_previous=None) == model
        )
        assert any(
            b.code == "published_model"
            for b in runs.preview_weight_deletion(admin, run.run_id).blockers
        )
    else:
        from regimex.application.deletion import DeletionService
        from regimex.infrastructure.deletion_repository import DeletionRepository

        preview = runs.preview_weight_deletion(admin, run.run_id)
        DeletionService(DeletionRepository(database), service.store).apply(
            admin,
            preview.plan_id,
            expected_sha256=preview.sha256,
            archive_run=True,
        )
        (archived,) = runs.list_runs(admin, kind="training", archive="archived")
        assert archived.external_evaluations == (evidence,) and archived.external_latency == latency
        assert archived.weights_deletion_id == preview.plan_id
        assert runs.training_metrics(admin, run.run_id).model_dump(mode="json") == metrics


def test_existing_tcn_import_keeps_repeat_outcomes_after_weight_deletion(
    service: AnalysisService, tmp_path: Path
) -> None:
    from psycopg.types.json import Json
    from test_tcn_backend import bundle as tcn_bundle

    from regimex.application.admin_service import AdminService
    from regimex.application.deletion import DeletionService
    from regimex.infrastructure.admin_repository import AdminRepository
    from regimex.infrastructure.deletion_repository import DeletionRepository
    from regimex.model_registry import tcn_descriptor

    database = service.repository
    owner = principal(database, "researcher", "admin")
    service.registry = ArchitectureRegistry((tcn_descriptor(),))
    runs = AdminService(
        service, AdminRepository(database), ServiceSettings(database_dsn=SecretStr(database._dsn))
    )
    _, _, _, checksum = tcn_bundle(tmp_path)
    service.register_bundle(
        owner,
        tmp_path / "bundle",
        manifest_sha256=checksum,
        inference_runtime=InferenceRuntime(threads=1),
        record_training=True,
    )
    (run,) = runs.list_runs(owner, kind="training")
    saved = runs.training_metrics(owner, run.run_id)
    assert [(outcome.seed, outcome.step, outcome.reason) for outcome in saved.repeat_outcomes] == [
        (11, 3, "epoch_limit"),
        (23, 3, "epoch_limit"),
    ]
    # Recreate the checksummed history saved before imports recorded repeat outcomes.
    old = saved.model_dump(mode="json")
    old["repeat_outcomes"] = []
    with database.connection() as connection:
        connection.execute(
            "UPDATE regimex_training_metrics SET document=%s WHERE operation_id=%s",
            (
                Json({"metrics": old, "sha256": hashlib.sha256(json_bytes(old)).hexdigest()}),
                run.run_id,
            ),
        )
    preview = runs.preview_weight_deletion(owner, run.run_id)
    DeletionService(DeletionRepository(database), service.store).apply(
        owner,
        preview.plan_id,
        expected_sha256=preview.sha256,
    )
    assert runs.training_metrics(owner, run.run_id) == saved


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
            assert await asyncio.to_thread(recover_operations, settings, service) == 1
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

    def put(
        key: str,
        source_path: Path,
        *,
        expected_sha256: str,
        checkpoint: Callable[[], None] | None = None,
    ) -> ObjectInfo:
        info = original_put(
            key, source_path, expected_sha256=expected_sha256, checkpoint=checkpoint
        )
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
