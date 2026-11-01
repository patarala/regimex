"""Persistent administrative workflows with real isolated scientific children."""

import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from subprocess import Popen
from typing import Any, cast
from uuid import UUID, uuid4

import pytest
from pydantic import SecretStr

pytest.importorskip("numpy")
pytest.importorskip("sklearn")
psycopg = pytest.importorskip("psycopg")
pytest.importorskip("procrastinate")
pytest.importorskip("alembic")

from test_application_training import source

from regimex.admin_contracts import (
    AdminKind,
    AdminRunRecord,
    DatasetRecord,
    DatasetSelection,
    DatasetSpec,
    EvaluationSpec,
    GenerationSpec,
    HumanAnnotation,
    PublicationSpec,
    RawGenerationPlan,
    ReferenceGenerationSpec,
    ReferenceImportSpec,
    ReferenceSetRecord,
    TrainingSpec,
)
from regimex.application.admin_service import AdminService
from regimex.application.benchmark_service import BenchmarkService
from regimex.application.service import AnalysisService
from regimex.data_catalog import Partition, ReferencePayload, seal_reference_set
from regimex.directions import (
    AnalysisConfig,
    Direction,
    DirectionAnnotations,
    DirectionInterval,
    summarize_interval,
)
from regimex.infrastructure.admin_repository import AdminRepository
from regimex.infrastructure.database import DatabaseRepository
from regimex.infrastructure.object_store import LocalObjectStore, ObjectInfo
from regimex.market import VIRTUAL_BASE_ASSETS
from regimex.model_registry import (
    ArchitectureRegistry,
    InferenceRuntime,
    TrainingPolicy,
    TrainingRuntime,
)
from regimex.operations import Operation, OperationFailure
from regimex.service_contracts import Forbidden, NotFound, Principal
from regimex.service_settings import ServiceSettings


def make_admin(repository: DatabaseRepository, tmp_path: Path) -> AdminService:
    settings = ServiceSettings(
        database_dsn=SecretStr(repository._dsn),
        database_schema=repository.schema,
        object_root=tmp_path / "objects",
        work_root=tmp_path / "work",
        admin_time_budget_seconds=120,
        admin_memory_budget_bytes=4 * 1024**3,
        admin_scratch_budget_bytes=64 * 1024**2,
    )
    base = AnalysisService(
        repository,
        LocalObjectStore(
            settings.object_root, max_object_bytes=16 * 1024**2, max_total_bytes=128 * 1024**2
        ),
        ArchitectureRegistry.discover(),
        settings.work_root,
        heartbeat_seconds=0.1,
    )
    administrative_repository = AdminRepository(repository)
    benchmark = BenchmarkService(repository, administrative_repository, base.store, base.work_root)
    return AdminService(
        base,
        administrative_repository,
        settings,
        authorize_selection=benchmark.authorize_selection,
        authorize_final_evaluation=benchmark.authorize_final_evaluation,
    )


def benchmarks(admin: AdminService) -> BenchmarkService:
    base = admin.analysis_service
    return BenchmarkService(base.repository, admin.repository, base.store, base.work_root)


@pytest.fixture
def admin(tmp_path: Path) -> Iterator[tuple[AdminService, Principal]]:
    dsn = os.environ.get("REGIMEX_TEST_DATABASE_DSN")
    if not dsn:
        pytest.skip("set REGIMEX_TEST_DATABASE_DSN to a disposable PostgreSQL database")
    repository = DatabaseRepository(dsn, schema="regimex_admin_workflow_" + uuid4().hex)
    repository.migrate()
    token = repository.create_token("fixture-administrator", "admin")
    try:
        with repository.queue_app.open():
            yield make_admin(repository, tmp_path), repository.authenticate(token)
    finally:
        with psycopg.connect(dsn, autocommit=True) as connection:
            connection.execute(
                psycopg.sql.SQL("DROP SCHEMA {} CASCADE").format(
                    psycopg.sql.Identifier(repository.schema)
                )
            )


def completed(admin: AdminService, run_id: UUID) -> None:
    result = admin.run_admin(run_id)
    assert result is not None
    assert result.operation.state == "succeeded", result.operation.model_dump_json()


def imported_dataset(
    admin: AdminService,
    principal: Principal,
    *,
    dataset_id: str = "fixture-dataset",
    indices: tuple[int, int, int] = (0, 1, 2),
) -> DatasetRecord:
    snapshots = [
        admin.analysis_service.create_snapshot(principal, source(index).export) for index in indices
    ]
    return admin.create_dataset(
        principal,
        DatasetSpec(
            dataset_id=dataset_id,
            members=tuple(
                DatasetSelection(
                    snapshot_id=snapshot.snapshot_id, partition=cast(Partition, partition)
                )
                for snapshot, partition in zip(
                    snapshots, ("train", "validation", "test"), strict=True
                )
            ),
        ),
    )


def prepared_references(
    admin: AdminService, principal: Principal
) -> tuple[DatasetRecord, ReferenceSetRecord]:
    dataset = imported_dataset(admin, principal)
    generation = admin.generate_references(
        principal,
        ReferenceGenerationSpec(
            dataset_id=dataset.dataset_id, reference_set_id="fixture-labels", partition="train"
        ),
        idempotency_key="labels",
    )
    completed(admin, generation.run_id)
    export = admin.analysis_service.read_snapshot(principal, dataset.snapshot_ids[1])
    annotations = DirectionAnnotations(
        experiment_id=export.source.snapshot_id,
        policy="human-directions/1.0",
        policy_config=AnalysisConfig().model_dump(mode="json"),
        reference_kind="human",
        reviewer="independent-fixture-reviewer",
        intervals=tuple(
            DirectionInterval(
                instrument_id="A/Q",
                start=start,
                stop=stop,
                direction=direction,
                status="observed",
                statistics=summarize_interval(export.observations, start, stop),
            )
            for start, stop, direction in (
                (0, 64, Direction.GROWTH),
                (64, 128, Direction.DECLINE),
                (128, 192, Direction.SIDEWAYS),
            )
        ),
    )
    validation = admin.import_references(
        principal,
        ReferenceImportSpec(
            dataset_id=dataset.dataset_id,
            reference_set_id="fixture-validation",
            partition="validation",
            reviewer="independent-fixture-reviewer",
            blind_review=True,
            independent=True,
            annotations=(
                HumanAnnotation(snapshot_id=dataset.snapshot_ids[1], annotation=annotations),
            ),
        ),
    )
    final_dataset = imported_dataset(
        admin, principal, dataset_id="separate-final-fixture", indices=(20, 21, 22)
    )
    final_references = admin.generate_references(
        principal,
        ReferenceGenerationSpec(
            dataset_id=final_dataset.dataset_id,
            reference_set_id="separate-final-references",
            partition="test",
        ),
        idempotency_key="separate-final-references",
    )
    completed(admin, final_references.run_id)
    benchmark = benchmarks(admin)
    benchmark.register_dataset_history(principal, dataset.dataset_id)
    namespaces = tuple(
        sorted(
            {
                member.source_namespace
                for member in (*dataset.manifest.members, *final_dataset.manifest.members)
            }
        )
    )
    benchmark.attest(
        principal,
        namespaces,
        "Complete isolated temporary database fixtures; no independent scientific quality claim.",
    )
    benchmark.reserve(
        principal, "fixture-final-benchmark", final_dataset.dataset_id, "separate-final-references"
    )
    benchmark.start_selection(principal, "fixture-final-benchmark")
    return dataset, validation


@pytest.mark.parametrize("architecture", ["baseline", "tcn"])
def test_generic_candidate_evaluate_publish_and_saved_analysis(
    admin: tuple[AdminService, Principal],
    architecture: str,
) -> None:
    service, principal = admin
    dataset, validation = prepared_references(service, principal)
    training = service.submit_training(
        principal,
        TrainingSpec(
            dataset_id=dataset.dataset_id,
            labels_reference_set_id="fixture-labels",
            validation_reference_set_id=validation.references.reference_set_id,
            model_version_id=f"fixture-trained-{architecture}",
            architecture_id=architecture,
            parameters={"regularization": [1.0], "context_sizes": [32, 128]}
            if architecture == "baseline"
            else {
                "channels": 4,
                "dilations": [1, 2],
                "dropout": 0.0,
                "smoothing": 0.0,
                "interval_weight_power": 0.0,
            },
            policy=TrainingPolicy(optimizer="lbfgs", seeds=(11,))
            if architecture == "baseline"
            else TrainingPolicy(
                optimizer="adamw",
                seeds=(11,),
                epochs=1,
                batch_size=1,
                patience=1,
                validation_interval=1,
            ),
            training_runtime=TrainingRuntime(threads=1, time_budget_seconds=120),
            inference_runtime=InferenceRuntime(threads=1),
            analysis_profile_id="direction-32/2.0",
            build_id="behavior-fixture",
            benchmark_selection_id="fixture-final-benchmark",
        ),
        idempotency_key="training",
    )
    completed(service, training.run_id)
    assert service.analysis_service.repository.get_active_model() is None
    candidate = service.analysis_service.repository.get_model(f"fixture-trained-{architecture}")
    assert candidate.quality_status == "validation_only"
    evaluation = service.submit_evaluation(
        principal,
        EvaluationSpec(
            model_version_id=candidate.model_version_id,
            dataset_id=dataset.dataset_id,
            reference_set_id=validation.references.reference_set_id,
        ),
        idempotency_key="evaluation",
    )
    completed(service, evaluation.run_id)
    metric = service.get_evaluation(principal, evaluation.run_id)
    assert metric.bundle_sha256 == candidate.bundle_sha256
    assert metric.dataset_sha256 == dataset.manifest.content_sha256
    assert metric.metrics["reference_kind"] == validation.references.kind
    assert metric.metrics["independent"] == validation.references.independent
    assert metric.metrics["blind_review"] == validation.references.blind_review
    confidence = metric.metrics["confidence_intervals"]
    assert isinstance(confidence, dict)
    assert confidence["repetitions"] == 1000
    assert confidence["experiment_count"] == 1
    assert "boundary_f1_percentile_interval" in confidence
    assert (
        metric.metrics["analysis_profile_sha256"] == validation.references.semantics.profile_sha256
    )
    assert metric.metrics["reference_policy"] == validation.references.policy
    unsupported = validation.references.model_dump(mode="json", exclude={"content_sha256"})
    unsupported["reference_set_id"] = "unsupported-decoder-fixture"
    unsupported["semantics"]["decoder_version"] = "unsupported-decoder/1.0"
    incompatible = seal_reference_set(ReferencePayload.model_validate_json(json.dumps(unsupported)))
    service.repository.add_reference_set(
        principal, validation.model_copy(update={"references": incompatible})
    )
    rejected_evaluation = service.submit_evaluation(
        principal,
        EvaluationSpec(
            model_version_id=candidate.model_version_id,
            dataset_id=dataset.dataset_id,
            reference_set_id=incompatible.reference_set_id,
        ),
        idempotency_key="unsupported-decoder-evaluation",
    )
    rejected = service.run_admin(rejected_evaluation.run_id)
    assert rejected is not None and rejected.operation.state == "failed"
    with pytest.raises(NotFound):
        service.get_evaluation(principal, rejected_evaluation.run_id)
    publication = service.submit_publication(
        principal,
        PublicationSpec(
            model_version_id=candidate.model_version_id, evaluation_run_id=evaluation.run_id
        ),
        idempotency_key="publication",
    )
    completed(service, publication.run_id)
    assert service.analysis_service.repository.get_active_model() == candidate
    assert service.get_metrics(principal)["evaluation_run_id"] == str(evaluation.run_id)
    analysis = service.analysis_service.submit_analysis(
        principal, dataset.snapshot_ids[1], idempotency_key="prediction"
    )
    result = service.analysis_service.run_analysis(analysis.analysis_id)
    assert result is not None and result.state == "succeeded"
    downloaded = service.get_reference_annotations(
        principal, validation.references.reference_set_id
    )
    assert downloaded[0].reference_kind == "human"
    assert downloaded[0].reviewer == "independent-fixture-reviewer"
    assert validation.references.independent
    benchmark = benchmarks(service)
    with pytest.raises(Forbidden):
        service.submit_final_evaluation(
            principal,
            "fixture-final-benchmark",
            candidate.model_version_id,
            idempotency_key="premature-final",
        )
    benchmark.freeze(principal, "fixture-final-benchmark", candidate.model_version_id)
    benchmark.open(principal, "fixture-final-benchmark")
    final = service.submit_final_evaluation(
        principal,
        "fixture-final-benchmark",
        candidate.model_version_id,
        idempotency_key="final-evaluation",
    )
    completed(service, final.run_id)
    final_report = service.read_evaluation_report(principal, final.run_id)
    assert final_report["scope"] == "controlled_final_benchmark"
    assert final_report["quality_status"] == "validation_only"
    assert final_report["reference_kind"] == "automatic"
    assert final_report["independent"] is False


def test_generation_has_separate_references_and_complete_immutable_dataset(
    admin: tuple[AdminService, Principal],
) -> None:
    service, principal = admin
    run = service.submit_generation(
        principal,
        GenerationSpec(
            dataset_id="generated-fixture",
            plan=RawGenerationPlan(seed=123456, experiment_count=3, step_count=64),
        ),
        idempotency_key="generation",
    )
    completed(service, run.run_id)
    dataset = service.get_dataset(principal, "generated-fixture")
    assert len(dataset.snapshot_ids) == 3
    assert {member.partition for member in dataset.manifest.members} == {
        "train",
        "validation",
        "test",
    }
    assert dataset.generation is not None and dataset.generation_provenance is not None
    assert service.list_reference_sets(principal) == ()
    assert all(member.split_group.seed is not None for member in dataset.manifest.members)
    for snapshot_id in dataset.snapshot_ids:
        export = service.analysis_service.read_snapshot(principal, snapshot_id)
        assert export.producer.version == export.source.source_version == "4.0"
        identities = {instrument.instrument_id for instrument in export.instruments}
        assert len(identities) == len(export.instruments)
        assert {row.instrument_id for row in export.observations} == identities
        for instrument in export.instruments:
            assert instrument.base_asset in VIRTUAL_BASE_ASSETS
            assert instrument.quote_asset == "USD"
            assert instrument.instrument_id == f"{instrument.base_asset}/USD"
    assert service.run_admin(run.run_id) is None


def test_sealed_test_references_are_not_available_to_training_or_downloads(
    admin: tuple[AdminService, Principal],
) -> None:
    service, principal = admin
    dataset = imported_dataset(service, principal)
    run = service.generate_references(
        principal,
        ReferenceGenerationSpec(
            dataset_id=dataset.dataset_id, reference_set_id="sealed-final-fixture", partition="test"
        ),
        idempotency_key="sealed-references",
    )
    completed(service, run.run_id)
    with pytest.raises(Forbidden):
        service.get_reference_annotations(principal, "sealed-final-fixture")
    assert all(
        item.references.partition != "test" for item in service.list_reference_sets(principal)
    )


def test_cancelled_child_is_terminated_and_receives_no_credentials(
    admin: tuple[AdminService, Principal],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, principal = admin
    run = service.submit_generation(
        principal,
        GenerationSpec(
            dataset_id="cancelled-generation",
            plan=RawGenerationPlan(seed=44, experiment_count=3, step_count=4096),
        ),
        idempotency_key="cancel-child",
    )
    monkeypatch.setenv("REGIMEX_DATABASE_DSN", "private-parent-credential")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "private-provider-credential")
    original = Popen
    children: list[Popen[bytes]] = []

    def launch(*args: Any, **kwargs: Any) -> Popen[bytes]:
        environment = cast(dict[str, str], kwargs["env"])
        assert "REGIMEX_DATABASE_DSN" not in environment
        assert "AWS_SECRET_ACCESS_KEY" not in environment
        child = cast(Popen[bytes], original(*args, **kwargs))
        children.append(child)
        service.analysis_service.cancel_operation(principal, run.run_id)
        return child

    monkeypatch.setattr(subprocess, "Popen", launch)
    result = service.run_admin(run.run_id)
    assert result is not None and result.operation.state == "cancelled"
    assert len(children) == 1 and children[0].poll() is not None
    with pytest.raises(NotFound):
        service.get_dataset(principal, "cancelled-generation")


def test_wall_budget_kills_generation_before_any_dataset_is_committed(
    admin: tuple[AdminService, Principal],
) -> None:
    service, principal = admin
    service.settings = service.settings.model_copy(update={"admin_time_budget_seconds": 1})
    run = service.submit_generation(
        principal,
        GenerationSpec(
            dataset_id="timed-generation",
            plan=RawGenerationPlan(seed=45, experiment_count=3, step_count=4096),
        ),
        idempotency_key="timeout-child",
    )
    result = service.run_admin(run.run_id)
    assert result is not None and result.operation.state == "failed"
    assert result.operation.failure is not None
    assert result.operation.failure.code == "capacity_limit"
    with pytest.raises(NotFound):
        service.get_dataset(principal, "timed-generation")


def test_computation_child_dies_when_supervising_parent_is_killed(tmp_path: Path) -> None:
    root = tmp_path / "private-scratch"
    root.mkdir(mode=0o700)
    script = """
import sys
from pathlib import Path
from regimex.admin_contracts import GenerationSpec, RawGenerationPlan
from regimex.computation_worker import ChildLimits, supervise_child
plan = RawGenerationPlan(seed=71,experiment_count=3,step_count=4096)
spec = GenerationSpec(dataset_id='parent-death-fixture',plan=plan)
payload = {'spec':spec.model_dump(mode='json'),'owner_id':'fixture','run_id':'fixture','inputs':{}}
supervise_child(payload,Path(sys.argv[1]),
    limits=ChildLimits(120,4*1024**3,64*1024**2,1),heartbeat=lambda:None)
"""
    parent = Popen(
        [sys.executable, "-c", script, str(root)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    child_pid = None
    try:
        ready = root / "child-process.json"
        deadline = time.monotonic() + 10
        while not ready.exists() and parent.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert ready.exists(), "the isolated child never installed its parent-death signal"
        identity = json.loads(ready.read_bytes())
        child_pid = int(identity["pid"])
        assert identity["parent"] == parent.pid
        parent.kill()
        parent.wait(timeout=5)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            status = Path(f"/proc/{child_pid}/stat")
            if not status.exists() or status.read_text().split()[2] == "Z":
                break
            time.sleep(0.05)
        else:
            pytest.fail("computation survived the abrupt death of its supervising parent")
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=5)
        if child_pid is not None:
            command = Path(f"/proc/{child_pid}/cmdline")
            if command.exists() and str(root).encode() in command.read_bytes():
                os.killpg(child_pid, signal.SIGKILL)


def test_child_address_space_budget_fails_as_capacity_without_committing(
    admin: tuple[AdminService, Principal],
) -> None:
    service, principal = admin
    service.settings = service.settings.model_copy(
        update={"admin_memory_budget_bytes": 64 * 1024**2}
    )
    run = service.submit_generation(
        principal,
        GenerationSpec(
            dataset_id="memory-limited-generation",
            plan=RawGenerationPlan(seed=46, experiment_count=3, step_count=64),
        ),
        idempotency_key="memory-child",
    )
    result = service.run_admin(run.run_id)
    assert result is not None and result.operation.state == "failed"
    assert result.operation.failure is not None
    assert result.operation.failure.code == "capacity_limit"
    with pytest.raises(NotFound):
        service.get_dataset(principal, "memory-limited-generation")


def test_transient_parent_database_error_fences_child_then_retries(
    admin: tuple[AdminService, Principal],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, principal = admin
    run = service.submit_generation(
        principal,
        GenerationSpec(
            dataset_id="retry-generation",
            plan=RawGenerationPlan(seed=47, experiment_count=3, step_count=64),
        ),
        idempotency_key="database-retry",
    )
    original = service.analysis_service.repository.heartbeat_operation

    def disconnected(operation_id: UUID, token: UUID, stage: str) -> Operation:
        raise psycopg.OperationalError("provider-credential-must-not-be-persisted")

    monkeypatch.setattr(service.analysis_service.repository, "heartbeat_operation", disconnected)
    first = service.run_admin(run.run_id)
    assert first is not None and first.operation.state == "queued"
    assert first.operation.attempt == 1
    monkeypatch.setattr(service.analysis_service.repository, "heartbeat_operation", original)
    completed(service, run.run_id)
    persisted = service.get_run(principal, run.run_id)
    assert persisted.operation.attempt == 2
    assert len(service.get_dataset(principal, "retry-generation").snapshot_ids) == 3
    assert "provider-credential" not in persisted.model_dump_json()


def test_cancelling_a_queued_retry_prevents_later_delivery(
    admin: tuple[AdminService, Principal],
) -> None:
    service, principal = admin
    run = service.submit_generation(
        principal,
        GenerationSpec(
            dataset_id="cancelled-retry",
            plan=RawGenerationPlan(seed=48, experiment_count=3, step_count=64),
        ),
        idempotency_key="cancel-retry",
    )
    repository = service.analysis_service.repository
    claimed = repository.claim_operation(run.run_id)
    assert claimed is not None and claimed.attempt_token is not None
    retry = repository.fail_operation(
        run.run_id,
        claimed.attempt_token,
        OperationFailure(
            code="infrastructure_failure", message="Storage unavailable.", retryable=True
        ),
    )
    assert retry.state == "queued" and retry.attempt == 1
    cancelled = service.analysis_service.cancel_operation(principal, run.run_id)
    assert cancelled.state == "cancelled"
    assert cancelled.attempt_tokens == claimed.attempt_tokens
    assert service.get_run(principal, run.run_id).operation == cancelled
    assert service.run_admin(run.run_id) is None
    with pytest.raises(NotFound):
        service.get_dataset(principal, "cancelled-retry")


def test_accepted_idempotency_replay_survives_concurrent_policy_change(
    admin: tuple[AdminService, Principal],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, principal = admin
    spec = GenerationSpec(
        dataset_id="accepted-generation",
        plan=RawGenerationPlan(seed=48, experiment_count=4, step_count=64),
    )
    accepted = service.submit_generation(principal, spec, idempotency_key="original-request")
    service.settings = service.settings.model_copy(update={"max_generation_experiments": 3})
    original = service.repository.find_replay
    first = True

    def concurrent(
        owner: Principal, kind: AdminKind, key: str, checksum: str
    ) -> AdminRunRecord | None:
        nonlocal first
        if first:
            first = False
            return None
        return original(owner, kind, key, checksum)

    monkeypatch.setattr(service.repository, "find_replay", concurrent)
    assert (
        service.submit_generation(principal, spec, idempotency_key="original-request") == accepted
    )


def test_input_scratch_budget_is_checked_before_any_object_is_copied(
    admin: tuple[AdminService, Principal],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, principal = admin
    dataset = imported_dataset(service, principal)
    run = service.generate_references(
        principal,
        ReferenceGenerationSpec(
            dataset_id=dataset.dataset_id, reference_set_id="bounded-reference", partition="train"
        ),
        idempotency_key="preflight",
    )
    service.settings = service.settings.model_copy(update={"admin_scratch_budget_bytes": 1})
    copies: list[str] = []

    def forbidden_copy(
        key: str, destination: Path, *, expected_sha256: str, expected_size_bytes: int
    ) -> ObjectInfo:
        copies.append(key)
        raise AssertionError("input bytes were copied before checking the total size")

    monkeypatch.setattr(service.analysis_service.store, "materialize", forbidden_copy)
    result = service.run_admin(run.run_id)
    assert result is not None and result.operation.failure is not None
    assert result.operation.failure.code == "capacity_limit"
    assert copies == []
    with pytest.raises(NotFound):
        service.get_reference_set(principal, "bounded-reference")


def test_input_materialization_counts_toward_attempt_wall_budget(
    admin: tuple[AdminService, Principal],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, principal = admin
    dataset = imported_dataset(service, principal)
    run = service.generate_references(
        principal,
        ReferenceGenerationSpec(
            dataset_id=dataset.dataset_id, reference_set_id="timed-reference", partition="train"
        ),
        idempotency_key="input-deadline",
    )
    service.settings = service.settings.model_copy(update={"admin_time_budget_seconds": 1})
    original = service.analysis_service.store.materialize
    copies: list[str] = []

    def delayed_copy(
        key: str, destination: Path, *, expected_sha256: str, expected_size_bytes: int
    ) -> ObjectInfo:
        copies.append(key)
        time.sleep(1.1)
        return original(
            key,
            destination,
            expected_sha256=expected_sha256,
            expected_size_bytes=expected_size_bytes,
        )

    monkeypatch.setattr(service.analysis_service.store, "materialize", delayed_copy)
    result = service.run_admin(run.run_id)
    assert result is not None and result.operation.failure is not None
    assert result.operation.failure.code == "capacity_limit"
    assert len(copies) == 1
    with pytest.raises(NotFound):
        service.get_reference_set(principal, "timed-reference")
