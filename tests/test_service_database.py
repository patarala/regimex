"""Real PostgreSQL checks for service authority and at-least-once effects."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta, tzinfo
from typing import TYPE_CHECKING, cast
from uuid import UUID, uuid4

import pytest
from pydantic import JsonValue

from regimex.exports import SourceIdentity
from regimex.infrastructure.object_store import ObjectInfo
from regimex.model_registry import InferenceRuntime
from regimex.operations import Operation, OperationFailure
from regimex.report_contracts import AnalysisReportRecord
from regimex.service_contracts import (
    CapacityExceeded,
    Conflict,
    Forbidden,
    ModelRecord,
    NotFound,
    Principal,
    ServiceError,
    SnapshotRecord,
    StaleAttempt,
    Unauthorized,
)
from regimex.storage import json_bytes

if TYPE_CHECKING:
    import psycopg

    from regimex.infrastructure.database import DatabaseRepository


@pytest.fixture
def repository() -> Iterator[DatabaseRepository]:
    dsn = os.environ.get("REGIMEX_TEST_DATABASE_DSN")
    if not dsn:
        pytest.skip("set REGIMEX_TEST_DATABASE_DSN to a disposable PostgreSQL database")
    psycopg = pytest.importorskip("psycopg", reason="install the service extra")
    pytest.importorskip("alembic", reason="install the service extra")
    pytest.importorskip("procrastinate", reason="install the service extra")
    from regimex.infrastructure.database import DatabaseRepository

    schema = "regimex_service_test_" + uuid4().hex
    repository = DatabaseRepository(dsn, schema=schema)
    try:
        repository.migrate()
        yield repository
    finally:
        with psycopg.connect(dsn, autocommit=True) as connection:
            connection.execute(
                psycopg.sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                    psycopg.sql.Identifier(schema)
                )
            )


def _object(key: str, digest: str = "a" * 64) -> ObjectInfo:
    return ObjectInfo(key=key, sha256=digest, size_bytes=1, version_id="fixture-1")


def _snapshot(repository: DatabaseRepository, owner: str = "alice") -> SnapshotRecord:
    return repository.add_snapshot(
        SnapshotRecord(
            snapshot_id=uuid4(),
            owner_id=owner,
            source_id="local-upload",
            run_id="run-1",
            source_snapshot_id="source-snapshot-1",
            content_sha256="b" * 64,
            object=_object(f"snapshots/{owner}"),
            created_at=datetime.now(UTC),
        )
    )


def _model(repository: DatabaseRepository, version: str = "baseline-1") -> ModelRecord:
    model = ModelRecord(
        model_version_id=version,
        architecture_id="baseline",
        bundle_sha256="a" * 64,
        manifest=_object(f"models/{version}/manifest.json"),
        members={"weights.npz": _object(f"models/{version}/weights.npz")},
        inference_runtime=InferenceRuntime(),
        inference_policy_sha256="c" * 64,
        created_at=datetime.now(UTC),
    )
    return repository.register_model(_principal("admin", "admin"), model)


def _principal(owner: str = "alice", role: str = "user") -> Principal:
    return Principal.model_validate({"owner_id": owner, "role": role, "token_id": uuid4()})


def _ready(repository: DatabaseRepository) -> tuple[Principal, SnapshotRecord]:
    model = _model(repository)
    repository.activate_model(
        _principal("admin", "admin"), model.model_version_id, expected_previous=None
    )
    return _principal(), _snapshot(repository)


def _submit(repository: DatabaseRepository, key: str = "request-1") -> UUID:
    principal, snapshot = _ready(repository)
    return repository.submit_analysis(
        principal, snapshot.snapshot_id, idempotency_key=key, request_sha256="d" * 64
    ).analysis_id


def _result(
    repository: DatabaseRepository, operation_id: UUID, values: dict[str, JsonValue]
) -> tuple[dict[str, JsonValue], AnalysisReportRecord]:
    """Align result/report metadata; actual HTML rendering has workflow tests."""
    analysis = repository.get_analysis_internal(operation_id)
    operation = analysis.operation
    assert operation.model is not None
    snapshot = repository.get_snapshot(_principal(operation.owner_id), analysis.snapshot_id)
    source = SourceIdentity(
        source_id=snapshot.source_id,
        source_version="fixture",
        configuration_id="fixture",
        run_id=snapshot.run_id,
        snapshot_id=snapshot.source_snapshot_id,
        state_revision="completed",
        run_status="completed",
        consistency="atomic_completed_snapshot",
    )
    payload: dict[str, JsonValue] = {
        "result_version": "2.0",
        "analysis_id": str(operation_id),
        "snapshot_id": str(snapshot.snapshot_id),
        "snapshot_sha256": snapshot.content_sha256,
        "source": source.model_dump(mode="json"),
        "model": operation.model.model_dump(mode="json"),
        **values,
    }
    report = AnalysisReportRecord(
        analysis_id=operation_id,
        owner_id=operation.owner_id,
        snapshot_id=snapshot.snapshot_id,
        snapshot_sha256=snapshot.content_sha256,
        source=source,
        model=operation.model,
        result_sha256=hashlib.sha256(json_bytes(payload)).hexdigest(),
        plotly_version="transaction-fixture",
        object=_object(f"reports/{operation_id}.html"),
        created_at=datetime.now(UTC),
    )
    return payload, report


def _counts(repository: DatabaseRepository) -> tuple[int, int]:
    with repository.connection() as connection:
        operations = connection.execute("SELECT count(*) FROM regimex_operations").fetchone()
        jobs = connection.execute("SELECT count(*) FROM procrastinate_jobs").fetchone()
        assert operations is not None and jobs is not None
        return int(str(operations[0])), int(str(jobs[0]))


def test_migrations_are_packaged_explicit_and_idempotent(repository: DatabaseRepository) -> None:
    assert repository.readiness()
    repository.migrate()
    assert repository.readiness()
    with repository.connection() as connection:
        connection.execute("UPDATE regimex_queue_schema SET schema_sha256=%s", ("0" * 64,))
    assert not repository.readiness()
    with pytest.raises(RuntimeError, match="Queue schema"):
        repository.migrate()


def test_exact_numeric_results_survive_storage_and_normalization_is_rejected(
    repository: DatabaseRepository,
) -> None:
    import psycopg

    analysis_id = _submit(repository)
    claimed = repository.claim_operation(analysis_id)
    assert claimed is not None and claimed.attempt_token is not None
    payload, report = _result(
        repository,
        analysis_id,
        {"values": [-0.0, 0.0, 1.0, 1e20, 100000000000000000000, 1e-7, 5e-324, 1e308]},
    )
    expected = json_bytes(payload)
    repository.complete_analysis(analysis_id, claimed.attempt_token, payload, report=report)
    saved = repository.get_analysis(_principal(), analysis_id)
    assert json_bytes(saved.result) == expected
    assert saved.operation.result_sha256 == hashlib.sha256(expected).hexdigest()
    with repository.connection() as connection:
        types = {
            str(row[0]): str(row[1])
            for row in connection.execute(
                "SELECT table_name || '.' || column_name,data_type FROM information_schema.columns "
                "WHERE table_schema=%s AND (table_name,column_name) IN "
                "(('regimex_analyses','result'),('regimex_admin_runs','request'),"
                "('regimex_admin_runs','output'),('regimex_datasets','document'),"
                "('regimex_reference_sets','document'),('regimex_evaluation_records','document'),"
                "('regimex_objects','document'),('regimex_analysis_reports','document'))",
                (repository.schema,),
            )
        }
        assert len(types) == 8
        assert types.pop("regimex_objects.document") == "jsonb"
        assert types.pop("regimex_analysis_reports.document") == "jsonb"
        assert set(types.values()) == {"json"}
        # A write through the wrong JSON codec must fail the committed digest
        # check; the reader cannot reconstruct the original number representation.
        connection.execute(
            "UPDATE regimex_analyses SET result=%s WHERE analysis_id=%s",
            (psycopg.types.json.Jsonb(payload), analysis_id),
        )
    with pytest.raises(Conflict, match="committed digest"):
        repository.get_analysis(_principal(), analysis_id)


def test_incomplete_restore_is_unready_and_cannot_admit_resources_or_be_reset_by_migrate(
    repository: DatabaseRepository,
) -> None:
    import psycopg

    with psycopg.connect(
        repository._dsn, options=f"-csearch_path={repository.schema}"
    ) as connection:
        connection.execute("DELETE FROM regimex_restore_state")
    assert not repository.readiness()
    with pytest.raises(ServiceError) as failure:
        repository.create_token("must-not-be-created", "user")
    assert failure.value.code == "restore_incomplete" and failure.value.status_code == 503
    repository.migrate()
    assert not repository.readiness()
    with psycopg.connect(
        repository._dsn, options=f"-csearch_path={repository.schema}"
    ) as connection:
        assert connection.execute("SELECT count(*) FROM regimex_tokens").fetchone() == (0,)
        assert connection.execute("SELECT count(*) FROM regimex_restore_state").fetchone() == (0,)


def test_credentials_are_hashed_revocable_and_source_bound(repository: DatabaseRepository) -> None:
    principal, token = repository.issue_token("simulator", "integration", source_namespace="sim-v1")
    assert principal.source_namespace == "sim-v1"
    assert repository.authenticate(token) == principal
    with repository.connection() as connection:
        row = connection.execute("SELECT token_sha256 FROM regimex_tokens").fetchone()
        assert row == (hashlib.sha256(token.encode()).hexdigest(),)
    repository.revoke_token(principal.token_id)
    with pytest.raises(Unauthorized):
        repository.authenticate(token)
    with pytest.raises(Unauthorized):
        repository.authenticate("bad-token")
    with pytest.raises(ValueError, match="source namespace"):
        repository.create_token("user", "user", source_namespace="untrusted")


def test_snapshot_identity_and_owner_access_survive_new_repository(
    repository: DatabaseRepository,
) -> None:
    first = _snapshot(repository)
    assert _snapshot(repository) == first
    changed = first.model_copy(update={"content_sha256": "f" * 64})
    with pytest.raises(Conflict, match="different content"):
        repository.add_snapshot(changed)
    assert repository.get_snapshot(_principal(), first.snapshot_id) == first
    for role in ("user", "admin"):
        with pytest.raises(NotFound):
            repository.get_snapshot(_principal("bob", role), first.snapshot_id)
    assert repository.list_snapshots(_principal("bob")) == ()


def test_object_identity_cannot_change_under_a_registered_key(
    repository: DatabaseRepository,
) -> None:
    snapshot = _snapshot(repository)
    changed = snapshot.model_copy(
        update={"object": snapshot.object.model_copy(update={"version_id": "different"})}
    )
    with pytest.raises(Conflict, match="object key"):
        repository.add_snapshot(changed)


def test_publication_requires_admin_cas_and_has_audit(repository: DatabaseRepository) -> None:
    first, second = _model(repository), _model(repository, "baseline-2")
    assert repository.list_models(_principal()) == ()
    with pytest.raises(NotFound):
        repository.get_visible_model(_principal(), first.model_version_id)
    assert len(repository.list_models(_principal("admin", "admin"))) == 2
    with pytest.raises(Forbidden):
        repository.activate_model(_principal(), first.model_version_id, expected_previous=None)
    repository.activate_model(
        _principal("admin", "admin"), first.model_version_id, expected_previous=None
    )
    with pytest.raises(Conflict, match="changed"):
        repository.activate_model(
            _principal("admin", "admin"), second.model_version_id, expected_previous=None
        )
    repository.activate_model(
        _principal("admin", "admin"),
        second.model_version_id,
        expected_previous=first.model_version_id,
    )
    assert repository.get_active_model() == second
    assert repository.list_models(_principal()) == (first, second)
    with repository.connection() as connection:
        assert connection.execute("SELECT count(*) FROM regimex_publication_audit").fetchone() == (
            2,
        )


def test_intake_replays_original_resolved_model_before_new_active_lookup(
    repository: DatabaseRepository,
) -> None:
    principal, snapshot = _ready(repository)
    original = repository.submit_analysis(
        principal, snapshot.snapshot_id, idempotency_key="same", request_sha256="d" * 64
    )
    second = _model(repository, "baseline-2")
    repository.activate_model(
        _principal("admin", "admin"), second.model_version_id, expected_previous="baseline-1"
    )
    replay = repository.submit_analysis(
        principal, snapshot.snapshot_id, idempotency_key="same", request_sha256="d" * 64
    )
    assert replay == original
    assert replay.operation.model is not None
    assert replay.operation.model.model_version_id == "baseline-1"
    with pytest.raises(Conflict, match="different request"):
        repository.submit_analysis(
            principal, snapshot.snapshot_id, idempotency_key="same", request_sha256="e" * 64
        )
    assert _counts(repository) == (1, 1)
    with repository.connection() as connection:
        row = connection.execute(
            "SELECT task_name,queue_name,args FROM procrastinate_jobs"
        ).fetchone()
        assert row == ("regimex.analysis", "analysis", {"operation_id": str(original.analysis_id)})


@pytest.mark.parametrize("metadata", (True, False))
def test_analysis_history_is_recent_before_pagination_with_owner_source_and_run_scope(
    repository: DatabaseRepository, monkeypatch: pytest.MonkeyPatch, metadata: bool
) -> None:
    from regimex.infrastructure import database as database_module

    owner, snapshot = _ready(repository)
    foreign_source = repository.add_snapshot(
        snapshot.model_copy(update={"snapshot_id": uuid4(), "source_id": "another-source"})
    )
    other_run = repository.add_snapshot(
        snapshot.model_copy(update={"snapshot_id": uuid4(), "run_id": "another-run"})
    )
    other_owner = repository.add_snapshot(
        snapshot.model_copy(update={"snapshot_id": uuid4(), "owner_id": "bob"})
    )
    current_time = datetime(2026, 1, 1, tzinfo=UTC)
    next_id = UUID(int=1, version=4)

    class Clock:
        @staticmethod
        def now(zone: tzinfo | None = None) -> datetime:
            assert zone is UTC
            return current_time

    # Deterministic clock/IDs make both UUID directions incorrect. All SQL,
    # transactional queue deliveries and resource authorization remain real.
    monkeypatch.setattr(database_module, "datetime", Clock)
    monkeypatch.setattr(database_module, "uuid4", lambda: next_id)
    order = []
    for index in range(24):
        current_time = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=index // 2)
        identity = 2000 - index if index < 22 else 10000 - (index - 22) * 1000
        next_id = UUID(int=identity, version=4)
        analysis = repository.submit_analysis(
            owner,
            snapshot.snapshot_id,
            idempotency_key=f"history-{index}",
            request_sha256="d" * 64,
            max_pending=64,
        )
        order.append((current_time, next_id))
        assert analysis.analysis_id == next_id
    expected = [identity for _, identity in sorted(order, reverse=True)]
    hidden = []
    for index, (who, source) in enumerate(
        ((owner, foreign_source), (owner, other_run), (_principal("bob"), other_owner))
    ):
        current_time = datetime(2026, 1, 2, tzinfo=UTC) + timedelta(minutes=index)
        next_id = UUID(int=20000 + index, version=4)
        hidden.append(
            repository.submit_analysis(
                who,
                source.snapshot_id,
                idempotency_key=f"out-of-scope-{index}",
                request_sha256="e" * 64,
                max_pending=64,
            ).analysis_id
        )
    read = repository.list_analysis_summaries if metadata else repository.list_analyses
    scoped = Principal(
        owner_id=owner.owner_id,
        role="integration",
        token_id=UUID(int=30000, version=4),
        source_namespace=snapshot.source_id,
        run_scope=snapshot.run_id,
    )
    first_page = [item.analysis_id for item in read(scoped, limit=20)]
    second_page = [item.analysis_id for item in read(scoped, limit=20, offset=20)]
    assert first_page == expected[:20]
    assert second_page == expected[20:]
    assert first_page[0] == UUID(int=10000, version=4)
    assert not read(scoped, limit=20, offset=24)
    assert [item.analysis_id for item in read(owner, limit=100)] == [
        hidden[1],
        hidden[0],
        *expected,
    ]
    source_only = scoped.model_copy(update={"run_scope": None})
    assert [item.analysis_id for item in read(source_only, limit=100)] == [hidden[1], *expected]
    assert [item.analysis_id for item in read(_principal("bob"), limit=20)] == [hidden[2]]
    assert not read(_principal("unrelated"), limit=20)


def test_submission_rollback_removes_both_resource_and_queue_task(
    repository: DatabaseRepository, monkeypatch: pytest.MonkeyPatch
) -> None:
    principal, snapshot = _ready(repository)
    enqueue = repository._enqueue

    def crash(connection: psycopg.Connection[tuple[object, ...]], operation: Operation) -> None:
        enqueue(connection, operation)
        raise RuntimeError("crash after queue insertion")

    monkeypatch.setattr(repository, "_enqueue", crash)
    with pytest.raises(RuntimeError, match="after queue"):
        repository.submit_analysis(
            principal, snapshot.snapshot_id, idempotency_key="same", request_sha256="d" * 64
        )
    assert _counts(repository) == (0, 0)
    assert repository.list_analyses(principal) == ()


def test_concurrent_idempotency_and_quota_are_serialized(repository: DatabaseRepository) -> None:
    principal, snapshot = _ready(repository)

    def submit(key: str) -> UUID:
        return repository.submit_analysis(
            principal,
            snapshot.snapshot_id,
            idempotency_key=key,
            request_sha256="d" * 64,
            max_pending=1,
        ).analysis_id

    with ThreadPoolExecutor(max_workers=4) as executor:
        ids = tuple(executor.map(submit, ["same"] * 4))
    assert len(set(ids)) == 1
    assert _counts(repository) == (1, 1)
    with pytest.raises(CapacityExceeded):
        submit("different")


def test_unpublished_models_and_other_owners_cannot_be_analyzed(
    repository: DatabaseRepository,
) -> None:
    principal, snapshot = _ready(repository)
    _model(repository, "candidate")
    for model, who, expected in (
        ("candidate", principal, NotFound),
        ("active", _principal("bob"), NotFound),
    ):
        with pytest.raises(expected):
            repository.submit_analysis(
                who,
                snapshot.snapshot_id,
                model_version_id=model,
                idempotency_key="same",
                request_sha256="d" * 64,
            )
    assert _counts(repository) == (0, 0)


def test_concurrent_delivery_claims_once_and_terminal_delivery_is_a_noop(
    repository: DatabaseRepository,
) -> None:
    operation_id = _submit(repository)
    with ThreadPoolExecutor(max_workers=4) as executor:
        attempts = tuple(executor.map(repository.claim_operation, [operation_id] * 4))
    active = [operation for operation in attempts if operation is not None]
    assert len(active) == 1
    token = active[0].attempt_token
    assert token is not None
    result, report = _result(repository, operation_id, {"rows": [0, None, 1.5]})
    completed = repository.complete_analysis(operation_id, token, result, report=report)
    assert completed.result_sha256 == hashlib.sha256(json_bytes(result)).hexdigest()
    assert repository.claim_operation(operation_id) is None
    assert repository.get_analysis(_principal(), operation_id).result == result
    with pytest.raises(StaleAttempt):
        repository.complete_analysis(operation_id, token, {"overwritten": True}, report=report)


def test_cancelled_and_recovered_attempts_cannot_commit(repository: DatabaseRepository) -> None:
    operation_id = _submit(repository)
    first = repository.claim_operation(operation_id)
    assert first is not None and first.attempt_token is not None
    recovered = repository.recover_operations(datetime.now(UTC))
    assert len(recovered) == 1 and recovered[0].state == "queued"
    second = repository.claim_operation(operation_id)
    assert second is not None and second.attempt_token is not None
    assert second.attempt_token != first.attempt_token
    _, report = _result(repository, operation_id, {})
    with pytest.raises(StaleAttempt):
        repository.complete_analysis(
            operation_id, first.attempt_token, {"stale": True}, report=report
        )
    with pytest.raises(NotFound):
        repository.cancel_operation(_principal("bob"), operation_id)
    cancelled = repository.cancel_operation(_principal(), operation_id)
    assert cancelled.state == "cancel_requested"
    with pytest.raises(StaleAttempt):
        repository.complete_analysis(
            operation_id, second.attempt_token, {"late": True}, report=report
        )
    final = repository.fail_operation(
        operation_id,
        second.attempt_token,
        OperationFailure(
            code="infrastructure_failure", message="Cancelled by owner.", retryable=True
        ),
    )
    assert final.state == "cancelled"
    assert repository.get_analysis(_principal(), operation_id).result is None
    assert _counts(repository) == (1, 2)


def test_exhausted_recovery_is_terminal_and_queued_cancel_never_starts(
    repository: DatabaseRepository,
) -> None:
    operation_id = _submit(repository)
    for _ in range(3):
        operation = repository.claim_operation(operation_id)
        assert operation is not None
        repository.recover_operations(datetime.now(UTC))
    final = repository.get_operation(_principal(), operation_id)
    assert final.state == "failed" and final.failure is not None
    assert final.failure.code == "attempts_exhausted"
    assert _counts(repository) == (1, 3)
    snapshot = repository.list_snapshots(_principal())[0]
    another = repository.submit_analysis(
        _principal(), snapshot.snapshot_id, idempotency_key="cancel-me", request_sha256="d" * 64
    )
    assert repository.cancel_operation(_principal(), another.analysis_id).state == "cancelled"
    assert repository.claim_operation(another.analysis_id) is None


def test_transient_failure_requeues_atomically_and_heartbeats_are_fenced(
    repository: DatabaseRepository,
) -> None:
    operation_id = _submit(repository)
    operation = repository.claim_operation(operation_id)
    assert operation is not None and operation.attempt_token is not None
    heartbeat = repository.heartbeat_operation(operation_id, operation.attempt_token, "predicting")
    assert heartbeat.stage == "predicting"
    assert repository.recover_operations(operation.created_at) == ()
    retried = repository.fail_operation(
        operation_id,
        operation.attempt_token,
        OperationFailure(
            code="infrastructure_failure", message="Temporary failure.", retryable=True
        ),
    )
    assert retried.state == "queued" and _counts(repository) == (1, 2)
    with repository.connection() as connection:
        row = connection.execute(
            "SELECT scheduled_at > clock_timestamp() FROM procrastinate_jobs "
            "ORDER BY id DESC LIMIT 1"
        ).fetchone()
        assert row == (True,)
    with pytest.raises(StaleAttempt):
        repository.heartbeat_operation(operation_id, operation.attempt_token, "late")


def test_result_limits_reject_nonfinite_without_partial_commit(
    repository: DatabaseRepository,
) -> None:
    operation_id = _submit(repository)
    operation = repository.claim_operation(operation_id)
    assert operation is not None and operation.attempt_token is not None
    _, report = _result(repository, operation_id, {})
    with pytest.raises(ValueError):
        repository.complete_analysis(
            operation_id, operation.attempt_token, {"score": float("nan")}, report=report
        )
    with pytest.raises(ValueError, match="16 MiB"):
        repository.complete_analysis(
            operation_id, operation.attempt_token, {"large": "x" * 2**24}, report=report
        )
    assert repository.get_analysis(_principal(), operation_id).result is None
    assert repository.get_operation(_principal(), operation_id).state == "running"


def test_integration_source_scope_applies_to_reads_lists_replays_and_cancellation(
    repository: DatabaseRepository,
) -> None:
    user, first = _ready(repository)
    second = repository.add_snapshot(
        first.model_copy(
            update={
                "snapshot_id": uuid4(),
                "source_id": "other-source",
                "object": _object("snapshots/other-source"),
            }
        )
    )
    analysis = repository.submit_analysis(
        user, second.snapshot_id, idempotency_key="foreign-source", request_sha256="d" * 64
    )
    integration = Principal(
        owner_id=user.owner_id,
        role="integration",
        token_id=uuid4(),
        source_namespace=first.source_id,
    )
    assert repository.list_snapshots(integration) == (first,)
    assert repository.list_analyses(integration) == ()
    with pytest.raises(NotFound):
        repository.get_snapshot(integration, second.snapshot_id)
    with pytest.raises(NotFound):
        repository.get_analysis(integration, analysis.analysis_id)
    with pytest.raises(NotFound):
        repository.get_operation(integration, analysis.analysis_id)
    with pytest.raises(NotFound):
        repository.cancel_operation(integration, analysis.analysis_id)
    with pytest.raises(NotFound):
        repository.submit_analysis(
            integration,
            second.snapshot_id,
            idempotency_key="foreign-source",
            request_sha256="d" * 64,
        )
    with pytest.raises(NotFound):
        repository.submit_analysis(
            integration,
            second.snapshot_id,
            idempotency_key="new-foreign-source",
            request_sha256="e" * 64,
        )
    assert _counts(repository) == (1, 1)


def test_concurrent_recovery_requeues_once_and_result_survives_reopening(
    repository: DatabaseRepository,
) -> None:
    operation_id = _submit(repository)
    assert repository.claim_operation(operation_id) is not None
    cutoff = datetime.now(UTC)
    with ThreadPoolExecutor(max_workers=4) as executor:
        recovered = tuple(executor.map(repository.recover_operations, [cutoff] * 4))
    assert sum(len(batch) for batch in recovered) == 1
    assert _counts(repository) == (1, 2)
    operation = repository.claim_operation(operation_id)
    assert operation is not None and operation.attempt_token is not None
    result, report = _result(repository, operation_id, {"complete": True})
    repository.complete_analysis(operation_id, operation.attempt_token, result, report=report)
    reopened = type(repository)(repository._dsn, schema=repository.schema)
    assert reopened.get_analysis(_principal(), operation_id).result == result
    assert reopened.claim_operation(operation_id) is None
    with reopened.connection() as connection:
        connection.execute(
            "UPDATE regimex_analyses SET result=%s WHERE operation_id=%s",
            ('{"changed":true}', operation_id),
        )
    with pytest.raises(Conflict, match="committed digest"):
        reopened.get_analysis(_principal(), operation_id)


def test_transient_retry_failure_rolls_back_state_and_new_delivery(
    repository: DatabaseRepository, monkeypatch: pytest.MonkeyPatch
) -> None:
    operation_id = _submit(repository)
    operation = repository.claim_operation(operation_id)
    assert operation is not None and operation.attempt_token is not None
    enqueue = repository._enqueue

    def crash(connection: psycopg.Connection[tuple[object, ...]], value: Operation) -> None:
        enqueue(connection, value)
        raise RuntimeError("crash after retry enqueue")

    monkeypatch.setattr(repository, "_enqueue", crash)
    with pytest.raises(RuntimeError, match="after retry"):
        repository.fail_operation(
            operation_id,
            operation.attempt_token,
            OperationFailure(
                code="infrastructure_failure", message="Temporary failure.", retryable=True
            ),
        )
    assert repository.get_operation(_principal(), operation_id).state == "running"
    assert _counts(repository) == (1, 1)


def test_failed_delivery_before_claim_is_repaired_with_a_bounded_budget(
    repository: DatabaseRepository,
) -> None:
    operation_id = _submit(repository)
    assert repository.recover_queued_deliveries() == ()
    for attempt in range(1, 4):
        with repository.connection() as connection:
            connection.execute("UPDATE procrastinate_jobs SET status='failed'")
        recovered = repository.recover_queued_deliveries()
        assert len(recovered) == 1 and recovered[0].attempt == attempt
        assert recovered[0].state == ("failed" if attempt == 3 else "queued")
        # A future scheduled task is still live, and cannot trigger another repair.
        assert repository.recover_queued_deliveries() == ()
    assert _counts(repository) == (1, 3)
    assert repository.claim_operation(operation_id) is None


def test_pruned_delivery_and_concurrent_repair_do_not_lose_or_duplicate_work(
    repository: DatabaseRepository,
) -> None:
    operation_id = _submit(repository)
    with repository.connection() as connection:
        connection.execute("DELETE FROM procrastinate_jobs")
    with ThreadPoolExecutor(max_workers=4) as executor:
        repairs = tuple(executor.map(lambda _: repository.recover_queued_deliveries(), range(4)))
    assert sum(len(batch) for batch in repairs) == 1
    assert _counts(repository) == (1, 1)
    operation = repository.claim_operation(operation_id)
    assert operation is not None and operation.attempt_token is not None
    result, report = _result(repository, operation_id, {"repaired": True})
    repository.complete_analysis(operation_id, operation.attempt_token, result, report=report)
    assert repository.get_analysis(_principal(), operation_id).result == result


def test_polling_and_lists_do_not_transfer_or_decode_result_payload(
    repository: DatabaseRepository, monkeypatch: pytest.MonkeyPatch
) -> None:
    from psycopg.types.json import set_json_loads

    operation_id = _submit(repository)
    assert not repository.get_analysis_summary(_principal(), operation_id).result_available
    operation = repository.claim_operation(operation_id)
    assert operation is not None and operation.attempt_token is not None
    payload, report = _result(repository, operation_id, {"large": "x" * 1024**2})
    repository.complete_analysis(operation_id, operation.attempt_token, payload, report=report)
    assert repository.get_analysis(_principal(), operation_id).result == payload
    snapshot = repository.list_snapshots(_principal())[0]
    queued = repository.submit_analysis(
        _principal(), snapshot.snapshot_id, idempotency_key="queued", request_sha256="d" * 64
    )
    original_connection = repository.connection

    def metadata_only_decoder(data: str | bytes) -> object:
        if len(data) > 64 * 1024:
            raise AssertionError("metadata retrieval transferred a large JSON payload")
        return cast(object, json.loads(data))

    @contextmanager
    def bounded_connection() -> Iterator[psycopg.Connection[tuple[object, ...]]]:
        with original_connection() as connection:
            set_json_loads(metadata_only_decoder, context=connection)
            yield connection

    monkeypatch.setattr(repository, "connection", bounded_connection)
    summary = repository.get_analysis_summary(_principal(), operation_id)
    assert summary.result_available and summary.operation.state == "succeeded"
    summaries = repository.list_analysis_summaries(_principal(), limit=100)
    assert {item.analysis_id for item in summaries} == {operation_id, queued.analysis_id}
    assert len(json_bytes([item.model_dump(mode="json") for item in summaries])) < 8192
    assert len(repository.list_analysis_summaries(_principal(), limit=1, offset=1)) == 1
    assert repository.list_analysis_summaries(_principal("bob")) == ()
    with pytest.raises(NotFound):
        repository.get_analysis_summary(_principal("bob"), operation_id)
    integration = Principal(
        owner_id="alice", role="integration", token_id=uuid4(), source_namespace=snapshot.source_id
    )
    assert repository.get_operation(integration, operation_id).state == "succeeded"
    assert repository.cancel_operation(integration, operation_id).state == "succeeded"
    with pytest.raises(AssertionError, match="large JSON"):
        repository.get_analysis(_principal(), operation_id)
    # An unauthorized result request also avoids loading another owner's payload.
    with pytest.raises(NotFound):
        repository.get_analysis(_principal("bob"), operation_id)


def test_success_requires_a_matching_report_in_the_same_commit(
    repository: DatabaseRepository,
) -> None:
    operation_id = _submit(repository)
    operation = repository.claim_operation(operation_id)
    assert operation is not None and operation.attempt_token is not None
    payload, report = _result(repository, operation_id, {"complete": True})
    with pytest.raises(ValueError, match="requires current report metadata"):
        repository.complete_analysis(
            operation_id, operation.attempt_token, payload, report=cast(AnalysisReportRecord, None)
        )
    with pytest.raises(Conflict, match="report identity"):
        repository.complete_analysis(
            operation_id,
            operation.attempt_token,
            payload,
            report=report.model_copy(update={"result_sha256": "0" * 64}),
        )
    summary = repository.get_analysis_summary(_principal(), operation_id)
    assert summary.operation.state == "running"
    assert not summary.result_available and not summary.report_available
    repository.complete_analysis(operation_id, operation.attempt_token, payload, report=report)
    summary = repository.get_analysis_summary(_principal(), operation_id)
    assert summary.operation.state == "succeeded"
    assert summary.result_available and summary.report_available
    assert repository.get_analysis_report(_principal(), operation_id) == report
