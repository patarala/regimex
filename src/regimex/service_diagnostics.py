"""Explicit operator diagnostics without changing jobs, models or stored objects."""

import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import psycopg

from regimex.computation_worker import ChildLimits, ComputationFailure, supervise_child
from regimex.diagnostic_contracts import DiagnosticCheck, DiagnosticReport, QueueName
from regimex.infrastructure.database import DatabaseRepository
from regimex.infrastructure.object_store import ObjectCapacityError, ObjectStore
from regimex.model_registry import RegistryError
from regimex.resource_limits import TimeBudgetExceeded
from regimex.service_contracts import ModelRecord
from regimex.service_health import queue_status
from regimex.service_settings import ServiceSettings


def _model_probe(settings: ServiceSettings, store: ObjectStore, model: ModelRecord) -> None:
    from regimex.application.analysis import _validate_registered_model
    from regimex.application.service import materialize_bundle

    _validate_registered_model(model)
    required = model.manifest.size_bytes + sum(item.size_bytes for item in model.members.values())
    if required > settings.cache_budget_bytes or required > settings.analysis_scratch_budget_bytes:
        raise ObjectCapacityError("Model does not fit the configured probe capacity.")
    settings.work_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    started = time.monotonic()

    def check_deadline() -> None:
        if time.monotonic() - started > settings.analysis_time_budget_seconds:
            raise TimeBudgetExceeded("The model probe exceeded its time limit.")

    with tempfile.TemporaryDirectory(prefix="doctor-", dir=settings.work_root) as temporary:
        root = Path(temporary)
        materialize_bundle(store, model, root / "model", before_copy=lambda _: check_deadline())
        actual = supervise_child(
            {"model": model.model_dump(mode="json")},
            root,
            limits=ChildLimits(
                seconds=settings.analysis_time_budget_seconds,
                memory_bytes=settings.analysis_memory_budget_bytes,
                scratch_bytes=settings.analysis_scratch_budget_bytes,
                threads=model.inference_runtime.threads,
            ),
            heartbeat=check_deadline,
            task="model_probe",
        )
    expected = {
        "model_version_id": model.model_version_id,
        "bundle_sha256": model.bundle_sha256,
        "inference_policy_sha256": model.inference_policy_sha256,
        "architecture_id": model.architecture_id,
    }
    if actual != expected:
        raise ValueError("The model probe returned inconsistent identity.")


def diagnose(settings: ServiceSettings) -> DiagnosticReport:
    """Run bounded read probes and report safe causes instead of private driver errors."""
    from regimex.infrastructure.s3_object_store import S3ObjectStore
    from regimex.service_bootstrap import create_object_store

    checks: list[DiagnosticCheck] = []

    def record(name: str, status: Literal["passed", "failed", "skipped"], message: str) -> None:
        checks.append(DiagnosticCheck(name=name, status=status, message=message))

    repository = DatabaseRepository(
        settings.database_dsn.get_secret_value(), schema=settings.database_schema
    )
    connected = False
    try:
        with psycopg.connect(
            settings.database_dsn.get_secret_value(), connect_timeout=5
        ) as connection:
            connection.execute("SELECT 1")
        connected = True
        record("database", "passed", "PostgreSQL accepts connections.")
    except psycopg.Error, OSError:
        record(
            "database",
            "failed",
            "PostgreSQL is unavailable. Check the database service and credentials.",
        )
    current_schema = connected and repository.readiness()
    record(
        "schema",
        "passed" if current_schema else "failed" if connected else "skipped",
        "Migrations, queue schema and restore guard are current."
        if current_schema
        else "Apply service migrations and complete any offline restoration before serving."
        if connected
        else "Database connectivity is required to check the schema.",
    )
    store: ObjectStore | None = None
    try:
        candidate = create_object_store(settings)
        if isinstance(candidate, S3ObjectStore):
            candidate.check()
        candidate.inventory()
        store = candidate
        record("storage", "passed", "Storage configuration and bounded inventory are readable.")
    except OSError, ValueError, RuntimeError:
        record(
            "storage",
            "failed",
            "Storage could not be read. Check permissions, provider configuration and capacity.",
        )
    model = None
    if current_schema:
        try:
            model = repository.get_active_model()
            record(
                "active_model",
                "passed" if model is not None else "failed",
                "An active model version is configured."
                if model is not None
                else "No active model is configured. Register and activate a model first.",
            )
        except OSError, ValueError, psycopg.Error:
            record(
                "active_model", "failed", "The active model record could not be read or validated."
            )
    else:
        record(
            "active_model",
            "skipped",
            "A current database schema is required to read the active model.",
        )
    queues: tuple[QueueName, ...] = ("analysis", "admin", "maintenance")
    for queue in queues:
        if current_schema:
            status = queue_status(repository, queue, stale_seconds=settings.stale_seconds)
            record(
                queue + "_queue",
                "passed" if status.state in {"available", "busy"} else "failed",
                status.message,
            )
        else:
            record(
                queue + "_queue",
                "skipped",
                "A current database schema is required to check queue workers.",
            )
    if model is not None and store is not None:
        try:
            _model_probe(settings, store, model)
            record(
                "model_load",
                "passed",
                "The active model restored successfully in a local isolated process. "
                "Remote worker capability was not tested.",
            )
        except ComputationFailure as error:
            record("model_load", "failed", error.failure.message)
        except OSError, ValueError, RuntimeError, RegistryError:
            record(
                "model_load",
                "failed",
                "The active model could not be restored. Check bundle integrity, "
                "local dependencies and worker resource limits.",
            )
    else:
        record(
            "model_load",
            "skipped",
            "A readable active model and storage are required for an isolated model probe.",
        )
    return DiagnosticReport(checked_at=datetime.now(UTC), checks=tuple(checks))
