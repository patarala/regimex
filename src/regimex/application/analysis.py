"""Scientific inference in a dedicated worker, without reference-label access."""

import fcntl
import hashlib
import math
import os
import re
import shutil
import stat
import tempfile
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Annotated
from uuid import UUID

import psycopg
from pydantic import Field, JsonValue

from regimex.application.service import AnalysisResult, AnalysisService, materialize_bundle
from regimex.computation_worker import ChildLimits, ComputationFailure, supervise_child
from regimex.contracts import MAX_INPUT_BYTES, StrictModel
from regimex.infrastructure.object_store import ObjectCapacityError, ObjectIntegrityError
from regimex.model_bundles import ModelBundle, verify_model_bundle
from regimex.model_registry import RegistryError
from regimex.operations import Operation, OperationFailure, Sha256
from regimex.report_contracts import MAX_REPORT_BYTES, AnalysisReportRecord
from regimex.resource_limits import ResourceBudgetExceeded
from regimex.service_contracts import ModelRecord, StaleAttempt
from regimex.storage import json_bytes

if TYPE_CHECKING:
    from regimex.model_contracts import LoadedPredictor


def _cache_bundle(
    service: AnalysisService, model: ModelRecord, *, heartbeat: Callable[[], None] | None = None
) -> Path:
    """Keep verified immutable bytes locally; repeated analysis does not redownload weights.

    A global cache lock bounds committed payload bytes across worker processes.
    Entries are retained until an operator clears this disposable cache; a full
    cache fails explicitly rather than deleting files still used by a backend.
    """
    cache = service.work_root / "model-cache"
    cache.mkdir(mode=0o700, exist_ok=True)
    if cache.is_symlink():
        raise ValueError("model cache cannot be a symbolic link")
    destination = cache / f"{model.bundle_sha256}-{model.inference_policy_sha256}"
    lock = os.open(cache / ".lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(lock).st_mode):
            raise ValueError("model cache lock must be a regular file")
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if heartbeat is not None:
                    heartbeat()
                time.sleep(0.05)
        exists = destination.exists() or destination.is_symlink()
        used = 0
        for entry in cache.iterdir():
            if entry.name == ".lock":
                continue
            if entry.name.startswith(".incoming-") and entry.is_dir() and not entry.is_symlink():
                # No other writer can own an incoming entry under this lock.
                for parent, _, _ in os.walk(entry, followlinks=False):
                    Path(parent).chmod(0o700)
                shutil.rmtree(entry)
                continue
            if re.fullmatch(r"[0-9a-f]{64}-[0-9a-f]{64}", entry.name) is None:
                raise ValueError("model cache contains an unexpected entry")
            if not entry.is_dir() or entry.is_symlink():
                raise ValueError("model cache entries must be regular directories")
            for parent, directories, names in os.walk(entry, followlinks=False):
                if any((Path(parent) / name).is_symlink() for name in directories):
                    raise ValueError("model cache cannot contain symbolic links")
                for name in names:
                    info = (Path(parent) / name).lstat()
                    if not stat.S_ISREG(info.st_mode):
                        raise ValueError("model cache must contain regular files")
                    used += info.st_size
        required = model.manifest.size_bytes + sum(
            info.size_bytes for info in model.members.values()
        )
        if used + (0 if exists else required) > service.cache_budget_bytes:
            raise ObjectCapacityError("model cache capacity is exhausted")
        if exists:
            return destination
        with tempfile.TemporaryDirectory(prefix=".incoming-", dir=cache) as temporary:
            incoming = Path(temporary) / "bundle"
            materialize_bundle(
                service.store,
                model,
                incoming,
                before_copy=None if heartbeat is None else lambda _: heartbeat(),
            )
            descriptor = service.registry.get(model.architecture_id)
            verify_model_bundle(
                incoming,
                manifest_sha256=model.bundle_sha256,
                architecture_id=descriptor.architecture_id,
                implementation_version=descriptor.implementation_version,
            )
            for parent, directories, names in os.walk(incoming):
                for name in names:
                    (Path(parent) / name).chmod(0o400)
                for name in directories:
                    (Path(parent) / name).chmod(0o500)
            incoming.rename(destination)
            destination.chmod(0o500)
        return destination
    finally:
        os.close(lock)


def load_registered_predictor(
    service: AnalysisService, model: ModelRecord, *, worker_id: str
) -> tuple[LoadedPredictor, ModelBundle]:
    """Materialize committed objects privately and let the backend validate its codec."""
    _validate_registered_model(model)
    from regimex.application.analysis_computation import restore_predictor

    return restore_predictor(
        _cache_bundle(service, model), model, service.registry, worker_id=worker_id
    )


def _validate_registered_model(model: ModelRecord) -> None:
    if (
        model.manifest.sha256 != model.bundle_sha256
        or model.manifest.size_bytes > 256 * 1024
        or sum(info.size_bytes for info in model.members.values()) + model.manifest.size_bytes
        > model.inference_runtime.model_cache_budget_bytes
        or hashlib.sha256(json_bytes(model.inference_runtime.model_dump(mode="json"))).hexdigest()
        != model.inference_policy_sha256
    ):
        raise ObjectIntegrityError("registered model digest, policy or resource budget differs")


class _Cancelled(RuntimeError):
    pass


class _Heartbeat:
    def __init__(self, service: AnalysisService, operation: Operation) -> None:
        assert operation.attempt_token is not None
        self.service = service
        self.operation_id = operation.operation_id
        self.token = operation.attempt_token
        self.stop = threading.Event()
        self.cancelled = threading.Event()
        self.error: Exception | None = None
        self.stage = "loading"
        self.deadline = time.monotonic() + service.time_budget_seconds
        self.thread = threading.Thread(target=self._run, daemon=True, name="regimex-heartbeat")

    def _run(self) -> None:
        while not self.stop.wait(min(self.service.heartbeat_seconds, 1.0)):
            try:
                self.check(self.stage)
            except Exception as exc:
                self.error = exc
                return

    def poll(self) -> None:
        if self.error is not None:
            raise self.error
        if self.cancelled.is_set():
            raise _Cancelled()
        if time.monotonic() > self.deadline:
            raise ResourceBudgetExceeded("Analysis exceeded its wall-time budget.")

    def check(self, stage: str) -> None:
        self.poll()
        self.stage = stage
        operation = self.service.repository.heartbeat_operation(
            self.operation_id, self.token, stage
        )
        if operation.state == "cancel_requested":
            self.cancelled.set()
            raise _Cancelled()

    def __enter__(self) -> _Heartbeat:
        self.check("loading")
        self.thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.stop.set()
        self.thread.join(timeout=1)


class _ComputedFiles(StrictModel):
    result_sha256: Sha256
    report_sha256: Sha256
    plotly_version: Annotated[str, Field(min_length=1, max_length=128)]


def _read_output(path: Path, limit: int, expected_sha256: str) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise ObjectIntegrityError("computed output is not a bounded regular file")
        data = stream.read(limit + 1)
    if len(data) > limit or hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ObjectIntegrityError("computed output differs from its manifest")
    return data


def run_analysis(
    service: AnalysisService, operation_id: UUID, *, worker_id: str
) -> Operation | None:
    operation = service.repository.claim_operation(operation_id)
    if operation is None:
        return None
    assert operation.attempt_token is not None
    token = operation.attempt_token
    try:
        with _Heartbeat(service, operation) as heartbeat:
            analysis = service.repository.get_analysis_internal(operation_id)
            snapshot = service.repository.get_snapshot_internal(analysis.snapshot_id)
            if operation.owner_id != snapshot.owner_id or operation.model is None:
                raise ValueError("operation snapshot ownership or model resolution differs")
            model = service.repository.get_model(operation.model.model_version_id)
            if (
                operation.model.bundle_sha256,
                operation.model.inference_policy_sha256,
            ) != (model.bundle_sha256, model.inference_policy_sha256):
                raise ObjectIntegrityError("operation immutable model differs from its registry")
            export = service._read_snapshot_record(snapshot)
            _validate_registered_model(model)
            descriptor = service.registry.get(model.architecture_id)
            if descriptor.backend_factory is None or descriptor.worker_probe is None:
                raise RegistryError("the configured backend is unavailable on this worker")
            bundle = _cache_bundle(service, model, heartbeat=heartbeat.poll)
            heartbeat.check("predicting")
            with tempfile.TemporaryDirectory(
                prefix=f"analysis-{operation_id}-", dir=service.work_root
            ) as temporary:
                root = Path(temporary)
                (root / "export.json").write_bytes(json_bytes(export.model_dump(mode="json")))
                payload: dict[str, JsonValue] = {
                    "snapshot": snapshot.model_dump(mode="json"),
                    "analysis": analysis.model_dump(mode="json"),
                    "model": model.model_dump(mode="json"),
                    "bundle_path": str(bundle.resolve()),
                    "worker_id": worker_id,
                    "descriptor": descriptor.model_dump(mode="json"),
                    "parameter_schema": descriptor.parameters_model.model_json_schema(),
                }
                remaining = max(1, math.ceil(heartbeat.deadline - time.monotonic()))
                files = _ComputedFiles.model_validate_json(
                    json_bytes(
                        supervise_child(
                            payload,
                            root,
                            limits=ChildLimits(
                                remaining,
                                service.memory_budget_bytes,
                                service.scratch_budget_bytes,
                                model.inference_runtime.threads,
                            ),
                            heartbeat=heartbeat.poll,
                            task="analysis",
                        )
                    )
                )
                heartbeat.check("rendering")
                result_bytes = _read_output(
                    root / "analysis-result.json", MAX_INPUT_BYTES, files.result_sha256
                )
                result = AnalysisResult.model_validate_json(result_bytes)
                if (
                    result.analysis_id != analysis.analysis_id
                    or result.snapshot_id != snapshot.snapshot_id
                    or result.model != operation.model
                    or result.inference_runtime != model.inference_runtime
                ):
                    raise ObjectIntegrityError("computed result differs from the claimed analysis")
                from regimex.application.report import validate_analysis_result

                validate_analysis_result(export, result, result_sha256=files.result_sha256)
                path = root / "analysis-report.html"
                _read_output(path, MAX_REPORT_BYTES, files.report_sha256)
                heartbeat.check("committing")
                report_object = service.store.put(
                    f"reports/{analysis.analysis_id}/{token}.html",
                    path,
                    expected_sha256=files.report_sha256,
                )
                report = AnalysisReportRecord(
                    analysis_id=analysis.analysis_id,
                    owner_id=operation.owner_id,
                    snapshot_id=snapshot.snapshot_id,
                    snapshot_sha256=snapshot.content_sha256,
                    source=export.source,
                    model=result.model,
                    result_sha256=files.result_sha256,
                    plotly_version=files.plotly_version,
                    object=report_object,
                    created_at=datetime.now(UTC),
                )
                heartbeat.check("committing")
                return service.repository.complete_analysis(
                    operation_id, token, result.model_dump(mode="json"), report=report
                )

    except StaleAttempt:
        return service.repository.get_analysis_internal(operation_id).operation
    except Exception as exc:
        # Error text can include paths, SQL or provider credentials. Persist only
        # stable public messages; operators can inspect the failed stage and code.
        if isinstance(exc, ComputationFailure):
            failure = exc.failure
        elif isinstance(exc, (ObjectCapacityError, ResourceBudgetExceeded)):
            failure = OperationFailure(
                code="capacity_limit",
                message="The worker storage or rendering capacity is exhausted.",
            )
        elif isinstance(exc, RegistryError):
            failure = OperationFailure(
                code="unsupported_configuration",
                message="The registered model is unavailable on this worker.",
            )
        elif isinstance(exc, (ObjectIntegrityError, ValueError)):
            failure = OperationFailure(
                code="incompatible_artifact",
                message="The committed input or model failed integrity and compatibility checks.",
            )
        else:
            failure = OperationFailure(
                code="infrastructure_failure",
                message="The analysis worker could not complete this attempt.",
                retryable=isinstance(
                    exc,
                    (
                        OSError,
                        psycopg.OperationalError,
                        psycopg.InterfaceError,
                        psycopg.errors.SerializationFailure,
                        psycopg.errors.DeadlockDetected,
                    ),
                ),
            )
        try:
            return service.repository.fail_operation(operation_id, token, failure)
        except StaleAttempt:
            return service.repository.get_analysis_internal(operation_id).operation
