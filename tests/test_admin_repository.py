"""Real PostgreSQL invariants for administrative authority and atomic effects.

Small object metadata fixtures isolate transactions from scientific fitting and
storage transfers, which are exercised by the application workflow tests.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Never
from uuid import UUID, uuid4

import pytest
from pydantic import JsonValue, ValidationError

from regimex.admin_contracts import (
    AnnotationObject,
    CatalogBinding,
    DatasetRecord,
    EvaluationRecord,
    EvaluationSpec,
    GenerationSpec,
    PublicationSpec,
    RawGenerationPlan,
    ReferenceGenerationSpec,
    ReferenceSetRecord,
    TrainingSpec,
)
from regimex.benchmarks import DirectionSemantics
from regimex.data_catalog import (
    Partition,
    ReferencePayload,
    SourceSnapshot,
    SplitGroup,
    analysis_profile_hash,
    create_reference_member,
    create_snapshot_member,
    seal_dataset,
    seal_reference_set,
)
from regimex.directions import (
    AnalysisConfig,
    Direction,
    DirectionAnnotations,
    DirectionInterval,
    summarize_interval,
)
from regimex.exports import ExportPayloadV2, seal_export
from regimex.infrastructure.object_store import ObjectInfo
from regimex.model_registry import InferenceRuntime, TrainingPolicy, TrainingRuntime
from regimex.operations import Operation, OperationFailure
from regimex.service_contracts import (
    CapacityExceeded,
    Conflict,
    Forbidden,
    ModelRecord,
    NotFound,
    Principal,
    SnapshotRecord,
    StaleAttempt,
)
from regimex.storage import json_bytes

if TYPE_CHECKING:
    from regimex.infrastructure.admin_repository import AdminRepository, Connection

PROFILE = AnalysisConfig()


def principal(owner: str = "administrator", *, admin: bool = True) -> Principal:
    return Principal(owner_id=owner, role="admin" if admin else "user", token_id=uuid4())


@pytest.fixture
def repository(monkeypatch: pytest.MonkeyPatch) -> Iterator[AdminRepository]:
    dsn = os.environ.get("REGIMEX_TEST_DATABASE_DSN")
    if not dsn:
        pytest.skip("set REGIMEX_TEST_DATABASE_DSN to a disposable PostgreSQL database")
    psycopg = pytest.importorskip("psycopg", reason="install the service extra")
    pytest.importorskip("alembic", reason="install the service extra")
    pytest.importorskip("procrastinate", reason="install the service extra")
    from regimex.infrastructure.admin_repository import AdminRepository
    from regimex.infrastructure.benchmark_repository import BenchmarkRepository
    from regimex.infrastructure.database import DatabaseRepository

    schema = "regimex_admin_test_" + uuid4().hex
    database = DatabaseRepository(dsn, schema=schema)

    def software_transaction_selection(
        _connection: Connection, _owner_id: str, spec: TrainingSpec, dataset: DatasetRecord
    ) -> None:
        # These tests never fit a model: their deliberately non-executable codec
        # metadata isolates SQL effects. Real training uses the benchmark gate.
        assert spec.benchmark_selection_id is None
        assert all(
            member.source_namespace == "repository-fixtures" for member in dataset.manifest.members
        )

    monkeypatch.setattr(
        BenchmarkRepository, "validate_selection", staticmethod(software_transaction_selection)
    )
    try:
        database.migrate()
        yield AdminRepository(database)
    finally:
        with psycopg.connect(dsn, autocommit=True) as connection:
            connection.execute(
                psycopg.sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                    psycopg.sql.Identifier(schema)
                )
            )


def object_info(key: str, data: bytes = b"fixture") -> ObjectInfo:
    return ObjectInfo(
        key=key,
        size_bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        version_id="fixture-1",
    )


def sources() -> tuple[SourceSnapshot, ...]:
    results = []
    original = json.loads((Path(__file__).parent / "fixtures" / "experiment-v2.json").read_text())
    original.pop("content_sha256")
    for index in range(3):
        value = json.loads(json.dumps(original))
        value["source"].update(run_id=f"run-{index}", snapshot_id=f"snapshot-{index}")
        value["time_grid"].update(origin=0.0, bin_width=1.0, completed_horizon=64.0, step_count=64)
        initial = 100.0 + index
        value["instruments"][0]["initial_price"] = initial
        template = value["observations"][0]
        value["observations"] = [
            {
                **template,
                "step": step,
                "last_price": round(initial + 0.01 * step, 2),
                "price_state": "observed",
                "trade_count": 1,
                "trade_volume": 1.0,
            }
            for step in range(64)
        ]
        results.append(
            SourceSnapshot(
                source_namespace="repository-fixtures",
                split_group=SplitGroup(
                    namespace="fixture-seeds", group_id=f"group-{index}", seed=index
                ),
                export=seal_export(ExportPayloadV2.model_validate_json(json.dumps(value))),
            )
        )
    return tuple(results)


def dataset_records(
    dataset_id: str = "dataset-1",
    *,
    owner: str = "administrator",
    generation: GenerationSpec | None = None,
) -> tuple[DatasetRecord, tuple[SnapshotRecord, ...], tuple[CatalogBinding, ...]]:
    raw = sources()
    snapshots = tuple(
        SnapshotRecord(
            snapshot_id=uuid4(),
            owner_id=owner,
            source_id=item.export.source.source_id,
            run_id=item.export.source.run_id,
            source_snapshot_id=item.export.source.snapshot_id,
            content_sha256=item.export.content_sha256,
            object=object_info(
                f"snapshots/{owner}/{uuid4()}", json_bytes(item.export.model_dump(mode="json"))
            ),
            created_at=datetime.now(UTC),
        )
        for item in raw
    )
    bindings = tuple(
        CatalogBinding(
            snapshot_id=stored.snapshot_id,
            source_namespace=item.source_namespace,
            split_group=item.split_group,
        )
        for stored, item in zip(snapshots, raw, strict=True)
    )
    partitions: tuple[Partition, ...] = ("train", "validation", "test")
    manifest = seal_dataset(
        dataset_id,
        tuple(
            create_snapshot_member(item, partition)
            for item, partition in zip(raw, partitions, strict=True)
        ),
    )
    record = DatasetRecord(
        dataset_id=dataset_id,
        owner_id=owner,
        manifest=manifest,
        snapshot_ids=tuple(item.snapshot_id for item in snapshots),
        created_at=datetime.now(UTC),
        generation=generation,
        generation_provenance=None
        if generation is None
        else object_info(f"generation/{dataset_id}/audit"),
    )
    return record, snapshots, bindings


def admitted_dataset(repository: AdminRepository) -> DatasetRecord:
    record, snapshots, bindings = dataset_records()
    for snapshot in snapshots:
        repository.database.add_snapshot(snapshot)
    return repository.add_dataset(principal(), record, bindings=bindings)


def reference_record(dataset: DatasetRecord, partition: Partition) -> ReferenceSetRecord:
    raw = sources()
    member = next(item for item in dataset.manifest.members if item.partition == partition)
    source = next(item for item in raw if item.export.source.snapshot_id == member.snapshot_id)
    document = DirectionAnnotations(
        experiment_id=member.snapshot_id,
        policy_config=PROFILE.model_dump(mode="json"),
        intervals=(
            DirectionInterval(
                instrument_id="A/Q",
                start=0,
                stop=64,
                direction=Direction.GROWTH,
                status="observed",
                statistics=summarize_interval(source.export.observations, 0, 64),
            ),
        ),
    )
    references = seal_reference_set(
        ReferencePayload(
            reference_set_id=f"references-{partition}",
            dataset_sha256=dataset.manifest.content_sha256,
            partition=partition,
            kind="automatic",
            policy="retrospective-directions/3.0",
            semantics=DirectionSemantics(
                profile_sha256=analysis_profile_hash(PROFILE),
                decoder_version="duration-viterbi/1.0",
            ),
            analysis_profile=PROFILE,
            members=(create_reference_member(member, document),),
        )
    )
    return ReferenceSetRecord(
        dataset_id=dataset.dataset_id,
        owner_id=dataset.owner_id,
        references=references,
        annotations=(
            AnnotationObject(
                source_namespace=member.source_namespace,
                snapshot_id=member.snapshot_id,
                object=object_info(
                    f"references/{partition}/annotations.json",
                    json_bytes(document.model_dump(mode="json")),
                ),
            ),
        ),
        created_at=datetime.now(UTC),
    )


def ready_training(repository: AdminRepository) -> tuple[DatasetRecord, TrainingSpec]:
    dataset = admitted_dataset(repository)
    for partition in ("train", "validation"):
        repository.add_reference_set(principal(), reference_record(dataset, partition))
    spec = TrainingSpec(
        dataset_id=dataset.dataset_id,
        labels_reference_set_id="references-train",
        validation_reference_set_id="references-validation",
        model_version_id="candidate-1",
        architecture_id="baseline",
        parameters={"regularization": [1.0]},
        policy=TrainingPolicy(optimizer="lbfgs", seeds=(11,)),
        training_runtime=TrainingRuntime(threads=1),
        inference_runtime=InferenceRuntime(threads=1),
        analysis_profile_id="direction-32/2.0",
        build_id="repository-fixture",
    )
    return dataset, spec


def candidate(spec: TrainingSpec) -> ModelRecord:
    manifest = object_info(f"models/{spec.model_version_id}/manifest.json")
    return ModelRecord(
        model_version_id=spec.model_version_id,
        architecture_id=spec.architecture_id,
        bundle_sha256=manifest.sha256,
        manifest=manifest,
        members={},
        inference_runtime=spec.inference_runtime,
        inference_policy_sha256=hashlib.sha256(
            json_bytes(spec.inference_runtime.model_dump(mode="json"))
        ).hexdigest(),
        created_at=datetime.now(UTC),
    )


def claim(repository: AdminRepository, run_id: UUID) -> UUID:
    operation = repository.database.claim_operation(run_id)
    assert operation is not None and operation.attempt_token is not None
    return operation.attempt_token


def trained(repository: AdminRepository) -> tuple[DatasetRecord, ModelRecord]:
    dataset, spec = ready_training(repository)
    run = repository.submit(principal(), spec, idempotency_key="training")
    model = candidate(spec)
    completed = repository.complete_training(run.run_id, claim(repository, run.run_id), model)
    assert completed.operation.state == "succeeded"
    return dataset, model


def evaluated(
    repository: AdminRepository,
    dataset: DatasetRecord,
    model: ModelRecord,
    *,
    key: str = "evaluation",
    metrics: dict[str, JsonValue] | None = None,
) -> EvaluationRecord:
    spec = EvaluationSpec(
        model_version_id=model.model_version_id,
        dataset_id=dataset.dataset_id,
        reference_set_id="references-validation",
    )
    run = repository.submit(principal(), spec, idempotency_key=key)
    record = EvaluationRecord(
        run_id=run.run_id,
        owner_id=dataset.owner_id,
        model_version_id=model.model_version_id,
        bundle_sha256=model.bundle_sha256,
        inference_policy_sha256=model.inference_policy_sha256,
        dataset_id=dataset.dataset_id,
        dataset_sha256=dataset.manifest.content_sha256,
        reference_set_id="references-validation",
        reference_sha256=repository.get_reference_set(
            principal(), "references-validation"
        ).references.content_sha256,
        partition="validation",
        report=object_info(f"evaluations/{run.run_id}/report.json"),
        metrics={"segment_macro_f1": 0.25} if metrics is None else metrics,
        created_at=datetime.now(UTC),
    )
    repository.complete_evaluation(run.run_id, claim(repository, run.run_id), record)
    return record


def test_numeric_requests_documents_metrics_and_output_preserve_identity(
    repository: AdminRepository, monkeypatch: pytest.MonkeyPatch
) -> None:
    from regimex.market_config import MarketConfig, ParticipantProfile

    profile = AnalysisConfig(boundary_penalty=1e20, decoder_switch_penalty=-0.0)
    monkeypatch.setattr(sys.modules[__name__], "PROFILE", profile)
    generation = GenerationSpec(
        dataset_id="dataset-1",
        plan=RawGenerationPlan(
            seed=8,
            experiment_count=3,
            step_count=64,
            market=MarketConfig(profiles=(ParticipantProfile(cash=-0.0),)),
        ),
    )
    dataset, snapshots, bindings = dataset_records(generation=generation)
    generation_run = repository.submit(principal(), generation, idempotency_key="numeric-raw")
    repository.complete_generation(
        generation_run.run_id,
        claim(repository, generation_run.run_id),
        dataset=dataset,
        snapshots=snapshots,
        bindings=bindings,
    )
    stored = repository.get_dataset(principal(), dataset.dataset_id)
    assert json_bytes(stored.model_dump(mode="json")) == json_bytes(dataset.model_dump(mode="json"))
    for partition in ("train", "validation"):
        references = reference_record(dataset, partition)
        repository.add_reference_set(principal(), references)
        actual = repository.get_reference_set(principal(), references.references.reference_set_id)
        assert json_bytes(actual.model_dump(mode="json")) == json_bytes(
            references.model_dump(mode="json")
        )
    spec = TrainingSpec(
        dataset_id=dataset.dataset_id,
        labels_reference_set_id="references-train",
        validation_reference_set_id="references-validation",
        model_version_id="numeric-candidate",
        architecture_id="baseline",
        parameters={"regularization": [1e20], "context_sizes": [4]},
        policy=TrainingPolicy(optimizer="lbfgs", seeds=(11,), weight_decay=-0.0),
        training_runtime=TrainingRuntime(threads=1),
        inference_runtime=InferenceRuntime(threads=1),
        analysis_profile_id="numeric-fixture",
        build_id="repository-fixture",
    )
    run = repository.submit(principal(), spec, idempotency_key="numeric-training")
    saved = repository.get_run(principal(), run.run_id)
    assert json_bytes(saved.spec.model_dump(mode="json")) == json_bytes(
        spec.model_dump(mode="json")
    )
    model = candidate(spec)
    repository.complete_training(run.run_id, claim(repository, run.run_id), model)
    record = evaluated(repository, dataset, model, metrics={"zero": -0.0, "large": 1e20})
    assert json_bytes(repository.get_evaluation(principal(), record.run_id).metrics) == json_bytes(
        record.metrics
    )
    # The fenced output contract permits arbitrary bounded JSON values, even
    # though the present model lifecycle returns mostly identifiers and hashes.
    output_run = repository.submit(
        principal(),
        GenerationSpec(
            dataset_id="numeric-output",
            plan=RawGenerationPlan(seed=9, experiment_count=3, step_count=64),
        ),
        idempotency_key="numeric-output",
    )
    token = claim(repository, output_run.run_id)
    with repository.database.connection() as connection:
        active = repository._active(connection, output_run.run_id, token, "generation")
        repository._finish(connection, active, token, {"zero": -0.0, "large": 1e20})
    assert json_bytes(repository.get_run(principal(), output_run.run_id).output) == json_bytes(
        {"zero": -0.0, "large": 1e20}
    )


def test_operation_contract_accepts_only_current_version() -> None:
    now = datetime.now(UTC)
    common = {
        "operation_id": uuid4(),
        "owner_id": "admin",
        "resource_id": "fixture",
        "request_sha256": "a" * 64,
        "idempotency_key": "fixture",
        "created_at": now,
        "updated_at": now,
    }
    for kind in ("generation", "training", "evaluation", "reference", "publication"):
        operation = Operation.model_validate({**common, "kind": kind})
        assert operation.version == "1.2"
        assert Operation.model_validate_json(operation.model_dump_json()) == operation
        for version in ("1.0", "1.1"):
            with pytest.raises(ValidationError, match=r"1\.2"):
                Operation.model_validate({**common, "kind": kind, "version": version})
    with pytest.raises(ValidationError, match="kind"):
        Operation.model_validate({**common, "kind": "report"})


def test_initial_schema_rejects_unknown_kinds_and_preserves_unsupported_contracts(
    repository: AdminRepository,
) -> None:
    import psycopg

    from regimex.service_contracts import UnsupportedContract

    database = repository.database
    now = datetime.now(UTC)
    operation = Operation(
        operation_id=uuid4(),
        owner_id="operator",
        kind="generation",
        resource_id="dataset",
        request_sha256="a" * 64,
        idempotency_key="unsupported-contract",
        created_at=now,
        updated_at=now,
    )
    unsupported = {**operation.model_dump(mode="json"), "version": "unsupported"}
    statement = (
        "INSERT INTO regimex_operations (operation_id,owner_id,kind,"
        "idempotency_key,request_sha256,state,document) VALUES (%s,%s,%s,%s,%s,%s,%s)"
    )
    arguments = (
        operation.operation_id,
        operation.owner_id,
        operation.kind,
        operation.idempotency_key,
        operation.request_sha256,
        operation.state,
        psycopg.types.json.Jsonb(unsupported),
    )
    with pytest.raises(psycopg.errors.CheckViolation), database.connection() as connection:
        connection.execute(statement, (*arguments[:2], "report", *arguments[3:]))
    with database.connection() as connection:
        connection.execute(statement, arguments)
    database.migrate()
    assert database.readiness()
    with pytest.raises(UnsupportedContract):
        database.get_operation(principal(operation.owner_id), operation.operation_id)
    with database.connection() as connection:
        row = connection.execute(
            "SELECT document FROM regimex_operations WHERE operation_id=%s",
            (operation.operation_id,),
        ).fetchone()
        assert row == (unsupported,)


def test_dataset_bindings_identity_and_role_owner_checks(repository: AdminRepository) -> None:
    record, snapshots, bindings = dataset_records()
    for snapshot in snapshots:
        repository.database.add_snapshot(snapshot)
    assert repository.get_binding(principal(), snapshots[0].snapshot_id) is None
    with pytest.raises(NotFound):
        repository.add_dataset(principal(), record)
    assert repository.list_datasets(principal()) == ()
    stored = repository.add_dataset(principal(), record, bindings=bindings)
    assert repository.add_dataset(principal(), record) == stored
    assert repository.get_dataset(principal(), record.dataset_id) == stored
    assert repository.list_datasets(principal(), offset=1) == ()
    with pytest.raises(Forbidden):
        repository.get_dataset(principal(admin=False), record.dataset_id)
    with pytest.raises(NotFound):
        repository.get_dataset(principal("different-admin"), record.dataset_id)
    with pytest.raises(Conflict, match="binding"):
        repository.add_binding(
            principal(), bindings[0].model_copy(update={"source_namespace": "another"})
        )


def test_dataset_members_cannot_forge_stored_export_identity(repository: AdminRepository) -> None:
    record, snapshots, bindings = dataset_records()
    for snapshot in snapshots:
        repository.database.add_snapshot(snapshot)
    changed = record.manifest.members[0].model_copy(update={"export_sha256": "f" * 64})
    manifest = seal_dataset(record.dataset_id, (changed, *record.manifest.members[1:]))
    with pytest.raises(Conflict, match="membership"):
        repository.add_dataset(
            principal(), record.model_copy(update={"manifest": manifest}), bindings=bindings
        )
    assert repository.list_datasets(principal()) == ()


def test_reference_sets_are_independent_immutable_and_test_access_is_closed(
    repository: AdminRepository,
) -> None:
    dataset = admitted_dataset(repository)
    for partition in ("train", "validation", "test"):
        record = reference_record(dataset, partition)
        assert repository.add_reference_set(principal(), record) == record
        assert repository.add_reference_set(principal(), record) == record
    assert len(repository.list_reference_sets(principal())) == 2
    assert len(repository.list_reference_sets(principal(), allow_test=True)) == 3
    with pytest.raises(Forbidden, match="benchmark"):
        repository.get_reference_set(principal(), "references-test")
    assert (
        repository.get_reference_set(
            principal(), "references-test", allow_test=True
        ).references.partition
        == "test"
    )
    with pytest.raises(Forbidden):
        repository.get_reference_set(principal(admin=False), "references-test", allow_test=True)
    with pytest.raises(NotFound):
        repository.get_reference_set(principal("different-admin"), "references-train")


def test_previous_reference_policy_returns_private_conflict_without_rewriting(
    repository: AdminRepository, tmp_path: Path
) -> None:
    pytest.importorskip("httpx2", reason="install the service-test group")
    from fastapi.testclient import TestClient
    from psycopg.types.json import Json
    from pydantic import SecretStr

    from regimex.api import create_app
    from regimex.application.admin_service import AdminService
    from regimex.application.service import AnalysisService
    from regimex.data_catalog import content_hash
    from regimex.infrastructure.object_store import LocalObjectStore
    from regimex.model_registry import ArchitectureRegistry
    from regimex.service_contracts import UnsupportedContract
    from regimex.service_settings import ServiceSettings

    dataset = admitted_dataset(repository)
    historical = {}
    with repository.database.connection() as connection:
        for partition in ("train", "test"):
            record = reference_record(dataset, partition)
            document = record.model_dump(mode="json")
            references = document["references"]
            references["policy"] = "retrospective-directions/1.0"
            references["content_sha256"] = content_hash(
                {key: value for key, value in references.items() if key != "content_sha256"}
            )
            historical[references["reference_set_id"]] = document
            # Insert the historical fixture without weakening immutable-update guards.
            connection.execute(
                "INSERT INTO regimex_reference_sets "
                "(reference_set_id,owner_id,dataset_id,partition,content_sha256,document) "
                "VALUES (%s,%s,%s,%s,%s,%s)",
                (
                    references["reference_set_id"],
                    record.owner_id,
                    record.dataset_id,
                    partition,
                    references["content_sha256"],
                    Json(document),
                ),
            )
    with pytest.raises(UnsupportedContract):
        repository.get_reference_set(principal(), "references-test", allow_test=True)

    settings = ServiceSettings(
        database_dsn=SecretStr(repository.database._dsn),
        database_schema=repository.database.schema,
        object_root=tmp_path / "objects",
        work_root=tmp_path / "work",
    )
    analysis = AnalysisService(
        repository.database,
        LocalObjectStore(
            settings.object_root, max_object_bytes=1024**2, max_total_bytes=4 * 1024**2
        ),
        ArchitectureRegistry.discover(),
        settings.work_root,
    )
    administration = AdminService(analysis, repository, settings)
    tokens = {
        "owner": repository.database.create_token("administrator", "admin"),
        "other": repository.database.create_token("other-administrator", "admin"),
        "user": repository.database.create_token("administrator", "user"),
    }
    with TestClient(create_app(analysis, admin_service=administration)) as client:
        for partition in ("train", "test"):
            for suffix in ("", "/manifest", "/annotations"):
                route = f"/api/v1/reference-sets/references-{partition}{suffix}"
                for identity, expected in (
                    ("owner", 409 if partition == "train" else 403),
                    ("other", 404),
                    ("user", 403),
                ):
                    response = client.get(
                        route, headers={"Authorization": f"Bearer {tokens[identity]}"}
                    )
                    assert response.status_code == expected
                    assert response.headers["Cache-Control"] == "private, no-store"
                    if expected == 409:
                        assert response.json()["error"]["code"] == "unsupported_contract"
                    assert "retrospective-directions" not in response.text
        response = client.get(
            "/api/v1/reference-sets", headers={"Authorization": f"Bearer {tokens['owner']}"}
        )
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "unsupported_contract"
        response = client.get(
            "/api/v1/reference-sets", headers={"Authorization": f"Bearer {tokens['other']}"}
        )
        assert response.status_code == 200 and response.json()["items"] == []
    with repository.database.connection() as connection:
        rows = connection.execute(
            "SELECT reference_set_id,document FROM regimex_reference_sets ORDER BY reference_set_id"
        ).fetchall()
    assert {identifier: document for identifier, document in rows} == historical


def test_queue_and_admin_intake_rollback_share_one_connection(
    repository: AdminRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = repository.database._enqueue

    def fail_after_enqueue(connection: Connection, operation: Operation) -> Never:
        original(connection, operation)
        raise RuntimeError("Crash after queue insertion")

    monkeypatch.setattr(repository.database, "_enqueue", fail_after_enqueue)
    spec = GenerationSpec(
        dataset_id="generation", plan=RawGenerationPlan(seed=1, experiment_count=3, step_count=64)
    )
    with pytest.raises(RuntimeError, match="Crash"):
        repository.submit(principal(), spec, idempotency_key="rolled-back")
    with repository.database.connection() as connection:
        for table in (
            "regimex_operations",
            "regimex_admin_runs",
            "regimex_generation_targets",
            "regimex_deliveries",
            "procrastinate_jobs",
        ):
            assert connection.execute(f"SELECT count(*) FROM {table}").fetchone() == (0,)


def test_admin_intake_idempotency_preserves_allocated_ids_and_ownership(
    repository: AdminRepository,
) -> None:
    spec = GenerationSpec(
        dataset_id="first", plan=RawGenerationPlan(seed=1, experiment_count=3, step_count=64)
    )
    run = repository.submit(principal(), spec, idempotency_key="key", request_sha256="a" * 64)
    replay = repository.submit(
        principal(),
        spec.model_copy(update={"dataset_id": "allocated-again"}),
        idempotency_key="key",
        request_sha256="a" * 64,
    )
    assert replay == run
    assert repository.find_replay(principal(), "generation", "key", "a" * 64) == run
    with pytest.raises(Conflict, match="idempotency"):
        repository.find_replay(principal(), "generation", "key", "b" * 64)
    with pytest.raises(NotFound):
        repository.get_run(principal("other-admin"), run.run_id)
    with pytest.raises(Forbidden):
        repository.get_run(principal(admin=False), run.run_id)
    with pytest.raises(NotFound):
        repository.database.get_operation(principal(admin=False), run.run_id)
    with repository.database.connection() as connection:
        row = connection.execute(
            "SELECT queue_name,task_name,args FROM procrastinate_jobs"
        ).fetchone()
    assert row == ("admin", "regimex.admin", {"operation_id": str(run.run_id)})


def test_admin_pending_quota_is_serialized_across_concurrent_keys(
    repository: AdminRepository,
) -> None:
    def submit(index: int) -> str:
        try:
            repository.submit(
                principal(),
                GenerationSpec(
                    dataset_id=f"dataset-{index}",
                    plan=RawGenerationPlan(seed=index, experiment_count=3, step_count=64),
                ),
                idempotency_key=f"key-{index}",
                max_pending=1,
            )
            return "accepted"
        except CapacityExceeded:
            return "capacity"

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(submit, (0, 1))) == ["accepted", "capacity"]


def test_cancelled_generation_cannot_expose_snapshots_or_dataset(
    repository: AdminRepository,
) -> None:
    spec = GenerationSpec(
        dataset_id="generated", plan=RawGenerationPlan(seed=1, experiment_count=3, step_count=64)
    )
    run = repository.submit(principal(), spec, idempotency_key="generation")
    token = claim(repository, run.run_id)
    repository.database.cancel_operation(principal(), run.run_id)
    record, snapshots, bindings = dataset_records("generated", generation=spec)
    with pytest.raises(StaleAttempt):
        repository.complete_generation(
            run.run_id, token, dataset=record, snapshots=snapshots, bindings=bindings
        )
    assert repository.list_datasets(principal()) == ()
    assert repository.database.list_snapshots(principal()) == ()


def test_recovered_generation_fences_old_attempt_and_commits_all_effects_once(
    repository: AdminRepository,
) -> None:
    spec = GenerationSpec(
        dataset_id="generated", plan=RawGenerationPlan(seed=1, experiment_count=3, step_count=64)
    )
    run = repository.submit(principal(), spec, idempotency_key="generation")
    old_token = claim(repository, run.run_id)
    repository.database.recover_stale(datetime.now(UTC))
    new_token = claim(repository, run.run_id)
    record, snapshots, bindings = dataset_records("generated", generation=spec)
    with pytest.raises(StaleAttempt):
        repository.complete_generation(
            run.run_id, old_token, dataset=record, snapshots=snapshots, bindings=bindings
        )
    assert (
        repository.complete_generation(
            run.run_id, new_token, dataset=record, snapshots=snapshots, bindings=bindings
        ).operation.state
        == "succeeded"
    )
    assert repository.database.claim_operation(run.run_id) is None
    assert len(repository.database.list_snapshots(principal())) == 3
    assert len(repository.list_datasets(principal())) == 1


def test_generation_resources_and_operation_rollback_together(
    repository: AdminRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = GenerationSpec(
        dataset_id="generated", plan=RawGenerationPlan(seed=1, experiment_count=3, step_count=64)
    )
    run = repository.submit(principal(), spec, idempotency_key="generation")
    token = claim(repository, run.run_id)
    record, snapshots, bindings = dataset_records("generated", generation=spec)

    def crash(*_args: object, **_kwargs: object) -> Never:
        raise RuntimeError("Crash after dataset and snapshots insertion")

    with monkeypatch.context() as context:
        context.setattr(repository, "_finish", crash)
        with pytest.raises(RuntimeError, match="Crash"):
            repository.complete_generation(
                run.run_id, token, dataset=record, snapshots=snapshots, bindings=bindings
            )
    assert repository.database.list_snapshots(principal()) == ()
    assert repository.list_datasets(principal()) == ()
    assert repository.get_run(principal(), run.run_id).operation.state == "running"
    repository.complete_generation(
        run.run_id, token, dataset=record, snapshots=snapshots, bindings=bindings
    )


def test_training_rejects_partition_misuse_and_reserves_candidate_identity(
    repository: AdminRepository,
) -> None:
    _, spec = ready_training(repository)
    wrong = spec.model_copy(update={"labels_reference_set_id": "references-validation"})
    with pytest.raises(Conflict, match="separate train"):
        repository.submit(principal(), wrong, idempotency_key="wrong")
    run = repository.submit(principal(), spec, idempotency_key="training")
    with pytest.raises(Conflict, match="reserved"):
        repository.database.register_model(principal(), candidate(spec))
    with pytest.raises(Conflict, match="reserved"):
        repository.submit(principal(), spec, idempotency_key="duplicate-target")
    assert repository.get_training_run_for_model(principal(), spec.model_version_id) == run


def test_internal_training_without_benchmark_is_denied_by_transaction_guard(
    repository: AdminRepository, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, spec = ready_training(repository)
    # Restore the production gate that the SQL-only fixture replaces explicitly.
    monkeypatch.undo()
    with pytest.raises(Forbidden, match="reserved benchmark"):
        repository.submit(principal(), spec, idempotency_key="unreserved-training")
    assert repository.list_runs(principal()) == ()


def test_candidate_and_success_rollback_together_and_require_publication(
    repository: AdminRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, spec = ready_training(repository)
    run = repository.submit(principal(), spec, idempotency_key="training")
    token = claim(repository, run.run_id)
    model = candidate(spec)

    def crash(*_args: object, **_kwargs: object) -> Never:
        raise RuntimeError("Crash after candidate insertion")

    with monkeypatch.context() as context:
        context.setattr(repository, "_finish", crash)
        with pytest.raises(RuntimeError, match="Crash"):
            repository.complete_training(run.run_id, token, model)
    with pytest.raises(NotFound):
        repository.database.get_model(model.model_version_id)
    assert repository.get_run(principal(), run.run_id).operation.state == "running"
    repository.complete_training(run.run_id, token, model)
    with pytest.raises(Conflict, match="publication"):
        repository.database.activate_model(
            principal(), model.model_version_id, expected_previous=None
        )
    assert repository.database.list_models(principal(admin=False)) == ()


def test_cancelled_training_cannot_register_candidate(repository: AdminRepository) -> None:
    _, spec = ready_training(repository)
    run = repository.submit(principal(), spec, idempotency_key="training")
    token = claim(repository, run.run_id)
    repository.database.cancel_operation(principal(), run.run_id)
    with pytest.raises(StaleAttempt):
        repository.complete_training(run.run_id, token, candidate(spec))
    with pytest.raises(NotFound):
        repository.database.get_model(spec.model_version_id)


def test_fixed_evaluation_identity_and_publication_cas_are_atomic(
    repository: AdminRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset, model = trained(repository)
    evaluation = evaluated(repository, dataset, model)
    assert repository.get_evaluation(principal(), evaluation.run_id) == evaluation
    assert repository.get_publication_evaluation(model.model_version_id) is None
    spec = PublicationSpec(
        model_version_id=model.model_version_id, evaluation_run_id=evaluation.run_id
    )
    run = repository.submit(principal(), spec, idempotency_key="publish")
    token = claim(repository, run.run_id)

    def crash(*_args: object, **_kwargs: object) -> Never:
        raise RuntimeError("Crash after deployment CAS")

    with monkeypatch.context() as context:
        context.setattr(repository, "_finish", crash)
        with pytest.raises(RuntimeError, match="Crash"):
            repository.complete_publication(run.run_id, token)
    assert repository.database.get_active_model() is None
    assert repository.get_publication_evaluation(model.model_version_id) is None
    completed = repository.complete_publication(run.run_id, token)
    assert completed.operation.state == "succeeded"
    assert repository.database.get_active_model() == model
    assert repository.get_publication_evaluation(model.model_version_id) == evaluation
    newer_evaluation = evaluated(repository, dataset, model, key="later-evaluation")
    assert newer_evaluation.run_id != evaluation.run_id
    assert repository.get_publication_evaluation(model.model_version_id) == evaluation
    assert (
        repository.database.get_visible_model(principal(admin=False), model.model_version_id)
        == model
    )
    stale = repository.submit(principal(), spec, idempotency_key="stale-publish")
    with pytest.raises(Conflict, match="deployment changed"):
        repository.complete_publication(stale.run_id, claim(repository, stale.run_id))
    assert repository.database.get_active_model() == model


@pytest.mark.parametrize(
    "field", ["dataset_sha256", "reference_sha256", "bundle_sha256", "inference_policy_sha256"]
)
def test_evaluation_cannot_commit_different_immutable_inputs(
    repository: AdminRepository, field: str
) -> None:
    dataset, model = trained(repository)
    valid = evaluated(repository, dataset, model)
    spec = EvaluationSpec(
        model_version_id=model.model_version_id,
        dataset_id=dataset.dataset_id,
        reference_set_id="references-validation",
    )
    run = repository.submit(principal(), spec, idempotency_key="wrong-inputs")
    wrong = valid.model_copy(update={"run_id": run.run_id, field: "f" * 64})
    with pytest.raises(Conflict, match="fixed model"):
        repository.complete_evaluation(run.run_id, claim(repository, run.run_id), wrong)
    with pytest.raises(NotFound):
        repository.get_evaluation(principal(), run.run_id)


def test_publication_requires_matching_evaluation_and_cancelled_job_cannot_publish(
    repository: AdminRepository,
) -> None:
    dataset, model = trained(repository)
    with pytest.raises(NotFound):
        repository.submit(
            principal(),
            PublicationSpec(model_version_id=model.model_version_id, evaluation_run_id=uuid4()),
            idempotency_key="unevaluated",
        )
    evaluation = evaluated(repository, dataset, model)
    run = repository.submit(
        principal(),
        PublicationSpec(
            model_version_id=model.model_version_id, evaluation_run_id=evaluation.run_id
        ),
        idempotency_key="publication",
    )
    token = claim(repository, run.run_id)
    repository.database.cancel_operation(principal(), run.run_id)
    with pytest.raises(StaleAttempt):
        repository.complete_publication(run.run_id, token)
    assert repository.database.get_active_model() is None


def test_internal_final_evaluation_cannot_forge_a_benchmark_grant(
    repository: AdminRepository,
) -> None:
    dataset, model = trained(repository)
    test_references = reference_record(dataset, "test")
    repository.add_reference_set(principal(), test_references)
    spec = EvaluationSpec(
        model_version_id=model.model_version_id,
        dataset_id=dataset.dataset_id,
        reference_set_id=test_references.references.reference_set_id,
        partition="test",
        benchmark_id="forged-benchmark",
    )
    with pytest.raises(NotFound):
        repository.submit(principal(), spec, idempotency_key="forged-final")
    assert repository.list_runs(principal(), kind="evaluation") == ()


def test_retry_and_stale_recovery_keep_administrative_queue_routing(
    repository: AdminRepository,
) -> None:
    spec = GenerationSpec(
        dataset_id="generated", plan=RawGenerationPlan(seed=1, experiment_count=3, step_count=64)
    )
    run = repository.submit(principal(), spec, idempotency_key="generation")
    token = claim(repository, run.run_id)
    repository.database.fail_operation(
        run.run_id,
        token,
        OperationFailure(
            code="infrastructure_failure", message="Transient fixture failure", retryable=True
        ),
    )
    claim(repository, run.run_id)
    repository.database.recover_stale(datetime.now(UTC))
    with repository.database.connection() as connection:
        rows = connection.execute(
            "SELECT DISTINCT queue_name,task_name FROM procrastinate_jobs"
        ).fetchall()
    assert rows == [("admin", "regimex.admin")]
    assert repository.get_run(principal(), run.run_id).operation.attempt == 2


def test_automatic_reference_commit_is_fenced_and_bound_to_requested_profile(
    repository: AdminRepository,
) -> None:
    dataset = admitted_dataset(repository)
    record = reference_record(dataset, "train")
    spec = ReferenceGenerationSpec(
        dataset_id=dataset.dataset_id,
        reference_set_id=record.references.reference_set_id,
        partition="train",
        analysis=PROFILE,
    )
    run = repository.submit(principal(), spec, idempotency_key="references")
    token = claim(repository, run.run_id)
    assert repository.complete_reference(run.run_id, token, record).operation.state == "succeeded"
    assert repository.get_reference_set(principal(), record.references.reference_set_id) == record
    with pytest.raises(StaleAttempt):
        repository.complete_reference(run.run_id, token, record)
