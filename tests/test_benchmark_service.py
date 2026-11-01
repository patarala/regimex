"""Real database benchmark gates using deliberate metadata-only fixtures.

No test fits a model or reads a reserved scientific benchmark. Numeric codecs
are deliberately non-executable; shared scientific workflow tests cover fitting.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest

from regimex.admin_contracts import (
    AnnotationObject,
    CatalogBinding,
    DatasetRecord,
    EvaluationSpec,
    ReferenceSetRecord,
    TrainingSpec,
)
from regimex.benchmarks import (
    BenchmarkMember,
    DatasetHistory,
    DirectionSemantics,
    PartitionMembership,
    RegistryRevision,
)
from regimex.data_catalog import (
    Partition,
    ReferenceMember,
    ReferencePayload,
    SourceSnapshot,
    SplitGroup,
    analysis_profile_hash,
    content_hash,
    create_snapshot_member,
    seal_dataset,
    seal_reference_set,
)
from regimex.directions import AnalysisConfig
from regimex.exports import ExportPayloadV2, seal_export
from regimex.infrastructure.object_store import LocalObjectStore, ObjectInfo
from regimex.model_bundles import BundleMember, BundleRole, ModelBundle, TrainingEvidence
from regimex.model_registry import InferenceRuntime, TrainingPolicy, TrainingRuntime
from regimex.service_contracts import (
    Conflict,
    Forbidden,
    ModelRecord,
    NotFound,
    Principal,
    SnapshotRecord,
)
from regimex.storage import json_bytes

if TYPE_CHECKING:
    from regimex.application.benchmark_service import BenchmarkService


def principal(owner: str = "administrator", role: str = "admin") -> Principal:
    return Principal.model_validate({"owner_id": owner, "role": role, "token_id": uuid4()})


@pytest.fixture
def service(tmp_path: Path) -> Iterator[BenchmarkService]:
    dsn = os.environ.get("REGIMEX_TEST_DATABASE_DSN")
    if not dsn:
        pytest.skip("set REGIMEX_TEST_DATABASE_DSN to a disposable PostgreSQL database")
    psycopg = pytest.importorskip("psycopg")
    pytest.importorskip("alembic")
    pytest.importorskip("procrastinate")
    from regimex.application.benchmark_service import BenchmarkService
    from regimex.infrastructure.admin_repository import AdminRepository
    from regimex.infrastructure.database import DatabaseRepository

    database = DatabaseRepository(dsn, schema="regimex_benchmark_test_" + uuid4().hex)
    store = LocalObjectStore(
        tmp_path / "objects", max_object_bytes=1024**2, max_total_bytes=16 * 1024**2
    )
    try:
        database.migrate()
        yield BenchmarkService(database, AdminRepository(database), store, tmp_path / "work")
    finally:
        with psycopg.connect(dsn, autocommit=True) as connection:
            connection.execute(
                psycopg.sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                    psycopg.sql.Identifier(database.schema)
                )
            )


def put(service: BenchmarkService, tmp_path: Path, key: str, data: bytes) -> ObjectInfo:
    path = tmp_path / "upload"
    path.write_bytes(data)
    return service.store.put(key, path, expected_sha256=hashlib.sha256(data).hexdigest())


def dataset(
    service: BenchmarkService,
    tmp_path: Path,
    name: str,
    offset: int,
    *,
    run_id: str | None = None,
    source_namespace: str = "benchmark-fixtures",
) -> DatasetRecord:
    original = json.loads((Path(__file__).parent / "fixtures" / "experiment-v2.json").read_text())
    original.pop("content_sha256")
    sources = []
    snapshots = []
    bindings = []
    for index in range(3):
        value = json.loads(json.dumps(original))
        identifier = f"{name}-{index}"
        value["source"].update(
            snapshot_id=identifier,
            run_id=run_id if index == 0 and run_id is not None else identifier,
        )
        value["time_grid"].update(origin=0.0, bin_width=1.0, completed_horizon=64.0, step_count=64)
        initial = 100.0 + offset * 10 + index
        value["instruments"][0]["initial_price"] = initial
        template = value["observations"][0]
        value["observations"] = [
            {
                **template,
                "step": step,
                "last_price": initial + step / 100,
                "trade_count": 1,
                "trade_volume": 1.0,
                "price_state": "observed",
            }
            for step in range(64)
        ]
        source = SourceSnapshot(
            source_namespace=source_namespace,
            split_group=SplitGroup(
                namespace="fixture-seeds", group_id=identifier, seed=offset * 3 + index
            ),
            export=seal_export(ExportPayloadV2.model_validate_json(json.dumps(value))),
        )
        sources.append(source)
        snapshot = SnapshotRecord(
            snapshot_id=uuid4(),
            owner_id="administrator",
            source_id=source.export.source.source_id,
            run_id=source.export.source.run_id,
            source_snapshot_id=source.export.source.snapshot_id,
            content_sha256=source.export.content_sha256,
            object=put(
                service,
                tmp_path,
                f"snapshots/{identifier}",
                json_bytes(source.export.model_dump(mode="json")),
            ),
            created_at=datetime.now(UTC),
        )
        service.database.add_snapshot(snapshot)
        snapshots.append(snapshot)
        bindings.append(
            CatalogBinding(
                snapshot_id=snapshot.snapshot_id,
                source_namespace=source.source_namespace,
                split_group=source.split_group,
            )
        )
    partitions: tuple[Partition, ...] = ("train", "validation", "test")
    record = DatasetRecord(
        dataset_id=name,
        owner_id="administrator",
        manifest=seal_dataset(
            name,
            tuple(
                create_snapshot_member(source, partition)
                for source, partition in zip(sources, partitions, strict=True)
            ),
        ),
        snapshot_ids=tuple(snapshot.snapshot_id for snapshot in snapshots),
        created_at=datetime.now(UTC),
    )
    return service.admin_repository.add_dataset(principal(), record, bindings=tuple(bindings))


def references(
    service: BenchmarkService, tmp_path: Path, dataset: DatasetRecord, partition: Partition
) -> ReferenceSetRecord:
    members = tuple(member for member in dataset.manifest.members if member.partition == partition)
    objects = tuple(
        put(
            service,
            tmp_path,
            f"closed-references/{dataset.dataset_id}/{partition}/{index}",
            b"metadata-only fixture; never valid scientific annotations",
        )
        for index, _ in enumerate(members)
    )
    payload = ReferencePayload(
        reference_set_id=f"{dataset.dataset_id}-{partition}",
        dataset_sha256=dataset.manifest.content_sha256,
        partition=partition,
        kind="automatic",
        policy="retrospective-directions/3.0",
        semantics=DirectionSemantics(
            profile_sha256=analysis_profile_hash(AnalysisConfig()),
            decoder_version="duration-viterbi/1.0",
        ),
        analysis_profile=AnalysisConfig(),
        members=tuple(
            ReferenceMember(
                source_namespace=member.source_namespace,
                snapshot_id=member.snapshot_id,
                export_sha256=member.export_sha256,
                annotations_sha256=info.sha256,
            )
            for member, info in zip(members, objects, strict=True)
        ),
    )
    record = ReferenceSetRecord(
        dataset_id=dataset.dataset_id,
        owner_id=dataset.owner_id,
        references=seal_reference_set(payload),
        annotations=tuple(
            AnnotationObject(
                source_namespace=member.source_namespace,
                snapshot_id=member.snapshot_id,
                object=info,
            )
            for member, info in zip(members, objects, strict=True)
        ),
        created_at=datetime.now(UTC),
    )
    return service.admin_repository.add_reference_set(principal(), record)


def model(service: BenchmarkService, tmp_path: Path, dataset: DatasetRecord) -> ModelRecord:
    labels = references(service, tmp_path, dataset, "train")
    validation = references(service, tmp_path, dataset, "validation")
    roles: tuple[BundleRole, ...] = (
        "configuration",
        "preprocessing",
        "weights",
        "validation",
        "history",
    )
    objects = {
        role: put(service, tmp_path, f"models/fake/{role}", b"non-executable metadata fixture")
        for role in roles
    }
    manifest = ModelBundle(
        bundle_version="1.0",
        model_version_id="fixture-model",
        architecture_id="baseline",
        implementation_version="0.1.0",
        interface_version="1.0",
        observation_version="2.1",
        feature_version="3.0",
        preprocessing_version="baseline-standardization/1.0",
        decoder_version="duration-viterbi/1.0",
        analysis_profile="direction-32/2.0",
        analysis_profile_sha256=analysis_profile_hash(AnalysisConfig()),
        class_names=("sideways", "growth", "decline"),
        training=TrainingEvidence(
            run_id="fixture-training",
            dataset_sha256=dataset.manifest.content_sha256,
            split_manifest_sha256=dataset.manifest.split_manifest_sha256,
            labels=labels.references.identity(),
            validation_references=validation.references.identity(),
            seeds=(11,),
            selected_seed=11,
            selection_metric="decoded_segment_macro_f1/iou-0.50",
            validation_segment_macro_f1=0.0,
        ),
        build_id="test",
        environment_sha256="a" * 64,
        members=tuple(
            BundleMember(path=role, role=role, size_bytes=info.size_bytes, sha256=info.sha256)
            for role, info in objects.items()
        ),
    )
    manifest_info = put(
        service, tmp_path, "models/fake/manifest.json", json_bytes(manifest.model_dump(mode="json"))
    )
    runtime = InferenceRuntime()
    return service.database.register_model(
        principal(),
        ModelRecord(
            model_version_id=manifest.model_version_id,
            architecture_id="baseline",
            bundle_sha256=manifest_info.sha256,
            manifest=manifest_info,
            members={str(role): info for role, info in objects.items()},
            inference_runtime=runtime,
            inference_policy_sha256=content_hash(runtime.model_dump(mode="json")),
            created_at=datetime.now(UTC),
        ),
    )


def training(dataset: DatasetRecord, *, benchmark_id: str | None = "fresh") -> TrainingSpec:
    return TrainingSpec(
        dataset_id=dataset.dataset_id,
        labels_reference_set_id=f"{dataset.dataset_id}-train",
        validation_reference_set_id=f"{dataset.dataset_id}-validation",
        model_version_id="new-candidate",
        architecture_id="baseline",
        parameters={"regularization": [1.0]},
        policy=TrainingPolicy(optimizer="lbfgs", seeds=(11,)),
        training_runtime=TrainingRuntime(),
        inference_runtime=InferenceRuntime(),
        analysis_profile_id="direction-32/2.0",
        build_id="test",
        benchmark_selection_id=benchmark_id,
    )


def reserved(service: BenchmarkService, tmp_path: Path) -> tuple[DatasetRecord, DatasetRecord]:
    development = dataset(service, tmp_path, "development", 1)
    final = dataset(service, tmp_path, "final", 10)
    references(service, tmp_path, final, "test")
    service.register_dataset_history(principal(), development.dataset_id)
    service.attest(
        principal(),
        ("benchmark-fixtures",),
        "Only deliberate fixtures in this isolated database were inventoried.",
    )
    service.reserve(principal(), "fresh", final.dataset_id, f"{final.dataset_id}-test")
    return development, final


def test_metadata_lifecycle_never_reads_final_payload_and_blocks_selection_after_freeze(
    service: BenchmarkService, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    development, final = reserved(service, tmp_path)
    registered = model(service, tmp_path, development)
    seen: list[str] = []
    materialize = service.store.materialize

    def guarded(
        key: str, destination: Path, *, expected_sha256: str, expected_size_bytes: int
    ) -> ObjectInfo:
        seen.append(key)
        assert key == "models/fake/manifest.json", (
            "benchmark lifecycle read observations, weights or references"
        )
        return materialize(
            key,
            destination,
            expected_sha256=expected_sha256,
            expected_size_bytes=expected_size_bytes,
        )

    monkeypatch.setattr(service.store, "materialize", guarded)
    with pytest.raises(Forbidden):
        service.authorize_selection(principal(), training(development), development)
    service.start_selection(principal(), "fresh")
    service.authorize_selection(principal(), training(development), development)
    with pytest.raises(Forbidden):
        service.authorize_final_evaluation(principal(), "fresh", registered.model_version_id)
    frozen = service.freeze(principal(), "fresh", registered.model_version_id)
    assert frozen.reservation.freeze is not None and len(frozen.reservation.members) == 1
    assert frozen.reservation.members[0].experiment_id == final.manifest.members[-1].snapshot_id
    with pytest.raises(Forbidden):
        service.authorize_selection(principal(), training(development), development)
    with pytest.raises(Forbidden):
        service.authorize_final_evaluation(principal(), "fresh", registered.model_version_id)
    opened = service.open(principal(), "fresh")
    grant = service.authorize_final_evaluation(principal(), "fresh", registered.model_version_id)
    assert grant.dataset_id == final.dataset_id and grant.partition == "test"
    assert grant.bundle_sha256 == registered.bundle_sha256
    assert grant.choices_sha256 == frozen.reservation.freeze.choices_sha256
    assert opened.reservation.opened_at == grant.opened_at
    assert seen == ["models/fake/manifest.json"]
    assert registered.quality_status == "validation_only"
    with pytest.raises(Conflict):
        service.open(principal(), "fresh")


def test_current_attestation_and_exact_registered_development_identity_are_required(
    service: BenchmarkService, tmp_path: Path
) -> None:
    development, _ = reserved(service, tmp_path)
    service.start_selection(principal(), "fresh")
    with pytest.raises(Forbidden):
        service.authorize_selection(
            principal(), training(development, benchmark_id=None), development
        )
    other = dataset(service, tmp_path, "other", 20)
    with pytest.raises(Forbidden):
        service.authorize_selection(principal(), training(other), other)
    state = service.register_dataset_history(principal(), other.dataset_id)
    assert state.registry.history_attestation is None
    with pytest.raises(Forbidden):
        service.authorize_selection(principal(), training(development), development)
    service.attest(
        principal(),
        ("benchmark-fixtures",),
        "Renewed fixture inventory after another complete dataset.",
    )
    service.authorize_selection(principal(), training(development), development)
    assert (
        service.register_dataset_history(principal(), other.dataset_id).revision
        == service.get_registry(principal()).revision
    )


def test_benchmark_ownership_and_default_final_reference_denial(
    service: BenchmarkService, tmp_path: Path
) -> None:
    _, final = reserved(service, tmp_path)
    actions: tuple[Callable[[], object], ...] = (
        lambda: service.get_benchmark(principal("other"), "fresh"),
        lambda: service.start_selection(principal("other"), "fresh"),
        lambda: service.open(principal("other"), "fresh"),
    )
    for action in actions:
        with pytest.raises(NotFound):
            action()
    with pytest.raises(Forbidden):
        service.get_registry(principal(role="user"))
    with pytest.raises(Forbidden):
        service.admin_repository.get_reference_set(principal(), f"{final.dataset_id}-test")


@pytest.mark.parametrize(
    "partition", ["train", "validation", "test", "external", "development", "final-review"]
)
def test_reservation_checks_whole_B_against_every_prior_partition(
    service: BenchmarkService, tmp_path: Path, partition: str
) -> None:
    final = dataset(service, tmp_path, "final", 10)
    references(service, tmp_path, final, "test")
    first = final.manifest.members[0]
    historical = BenchmarkMember(
        source_namespace=first.source_namespace,
        experiment_id="renamed",
        group_id="old-group",
        seed=999,
        observation_hashes=first.observation_hashes,
    )
    service.import_history(
        principal(),
        DatasetHistory(
            dataset_id="imported",
            dataset_sha256="c" * 64,
            recorded_at=datetime.now(UTC),
            memberships=(
                PartitionMembership.model_validate({"partition": partition, "member": historical}),
            ),
        ),
    )
    service.attest(
        principal(), ("benchmark-fixtures",), "Fixture-only historical metadata inventory."
    )
    with pytest.raises(Conflict, match="overlaps"):
        service.reserve(principal(), "fresh", final.dataset_id, f"{final.dataset_id}-test")
    assert service.get_registry(principal()).registry.benchmarks == ()


def test_training_intake_and_freeze_share_a_transactional_gate(
    service: BenchmarkService, tmp_path: Path
) -> None:
    development, _ = reserved(service, tmp_path)
    registered = model(service, tmp_path, development)
    service.start_selection(principal(), "fresh")
    run = service.admin_repository.submit(
        principal(), training(development), idempotency_key="training", request_sha256="a" * 64
    )
    with pytest.raises(Conflict, match="pending"):
        service.freeze(principal(), "fresh", registered.model_version_id)
    service.database.cancel_operation(principal(), run.run_id)
    service.freeze(principal(), "fresh", registered.model_version_id)
    with pytest.raises(Forbidden):
        service.admin_repository.submit(
            principal(),
            training(development).model_copy(update={"model_version_id": "later"}),
            idempotency_key="later",
            request_sha256="b" * 64,
        )
    with service.database.connection() as connection:
        assert connection.execute("SELECT count(*) FROM procrastinate_jobs").fetchone() == (1,)


def test_hash_chain_is_preserved_under_concurrent_attestations_and_rewrite_is_rejected(
    service: BenchmarkService, tmp_path: Path
) -> None:
    reserved(service, tmp_path)

    def attest(index: int) -> int:
        return service.attest(
            principal(), ("benchmark-fixtures",), f"Fixture inventory assertion {index}."
        ).revision

    with ThreadPoolExecutor(max_workers=4) as executor:
        revisions = tuple(executor.map(attest, range(4)))
    assert len(set(revisions)) == 4
    previous = None
    with service.database.connection() as connection:
        rows = connection.execute(
            "SELECT document,sha256 FROM regimex_benchmark_revisions ORDER BY revision"
        ).fetchall()
        for index, row in enumerate(rows):
            revision = RegistryRevision.model_validate_json(json_bytes(row[0]))
            assert revision.revision == index and revision.previous_sha256 == previous
            assert content_hash(revision.model_dump(mode="json")) == row[1]
            previous = row[1]
    psycopg = pytest.importorskip("psycopg")
    with (
        pytest.raises(psycopg.Error, match="immutable"),
        service.database.connection() as connection,
    ):
        connection.execute(
            "UPDATE regimex_benchmark_revisions SET sha256=%s WHERE revision=0", ("0" * 64,)
        )


def test_final_reference_ingress_requires_fresh_unreserved_data(
    service: BenchmarkService, tmp_path: Path
) -> None:
    from regimex.infrastructure.benchmark_repository import BenchmarkRepository

    development, final = reserved(service, tmp_path)
    fresh = dataset(service, tmp_path, "future-final", 30)
    with service.database.connection() as connection:
        BenchmarkRepository.validate_final_reference_import(connection, principal().owner_id, fresh)
    for exposed in (development, final):
        with pytest.raises(Forbidden), service.database.connection() as connection:
            BenchmarkRepository.validate_final_reference_import(
                connection, principal().owner_id, exposed
            )


def test_final_evaluation_grants_reject_any_change_to_frozen_model_policy_or_dataset(
    service: BenchmarkService, tmp_path: Path
) -> None:
    from regimex.infrastructure.benchmark_repository import BenchmarkRepository

    development, final = reserved(service, tmp_path)
    registered = model(service, tmp_path, development)
    service.start_selection(principal(), "fresh")
    service.freeze(principal(), "fresh", registered.model_version_id)
    service.open(principal(), "fresh")
    final_refs = service.admin_repository.get_reference_set(
        principal(), "final-test", allow_test=True
    )
    spec = EvaluationSpec(
        model_version_id=registered.model_version_id,
        dataset_id=final.dataset_id,
        reference_set_id="final-test",
        partition="test",
        benchmark_id="fresh",
    )
    for changed_model, changed_dataset in (
        (registered.model_copy(update={"inference_policy_sha256": "0" * 64}), final),
        (registered.model_copy(update={"bundle_sha256": "0" * 64}), final),
        (registered, development),
    ):
        with pytest.raises(Forbidden), service.database.connection() as connection:
            BenchmarkRepository.validate_final_evaluation(
                connection,
                principal().owner_id,
                spec,
                changed_model,
                changed_dataset,
                final_refs,
            )


def test_history_chain_detects_tampering_even_if_database_owner_disables_the_trigger(
    service: BenchmarkService, tmp_path: Path
) -> None:
    from psycopg.types.json import Jsonb

    reserved(service, tmp_path)
    with service.database.connection() as connection:
        row = connection.execute(
            "SELECT revision,document FROM regimex_benchmark_revisions "
            "ORDER BY revision DESC LIMIT 1"
        ).fetchone()
        assert row is not None
        document = json.loads(json_bytes(row[1]))
        document["registry"]["history_attestation"]["statement"] = "Tampered fixture assertion."
        # A database owner can bypass triggers. Hash verification still detects
        # accidental/document-only edits; coordinated privileged rewriting is
        # outside a local hash chain's guarantees.
        connection.execute("ALTER TABLE regimex_benchmark_revisions DISABLE TRIGGER USER")
        connection.execute(
            "UPDATE regimex_benchmark_revisions SET document=%s WHERE revision=%s",
            (Jsonb(document), row[0]),
        )
        connection.execute("ALTER TABLE regimex_benchmark_revisions ENABLE TRIGGER USER")
    with pytest.raises(Conflict, match="integrity"):
        service.get_registry(principal())


def test_source_run_identity_prevents_a_renamed_resampled_benchmark(
    service: BenchmarkService, tmp_path: Path
) -> None:
    development = dataset(service, tmp_path, "development", 1)
    final = dataset(service, tmp_path, "final", 10, run_id="development-0")
    references(service, tmp_path, final, "test")
    service.register_dataset_history(principal(), development.dataset_id)
    service.attest(principal(), ("benchmark-fixtures",), "Complete fixture history.")
    with pytest.raises(Conflict, match="overlaps"):
        service.reserve(principal(), "fresh", final.dataset_id, "final-test")


def test_concurrent_training_intake_and_freeze_cannot_both_succeed(
    service: BenchmarkService, tmp_path: Path
) -> None:
    development, _ = reserved(service, tmp_path)
    registered = model(service, tmp_path, development)
    service.start_selection(principal(), "fresh")

    def freeze() -> bool:
        try:
            service.freeze(principal(), "fresh", registered.model_version_id)
            return True
        except Conflict:
            return False

    def submit() -> bool:
        try:
            service.admin_repository.submit(
                principal(),
                training(development),
                idempotency_key="racing",
                request_sha256="a" * 64,
            )
            return True
        except Forbidden:
            return False

    with ThreadPoolExecutor(max_workers=2) as executor:
        freezing, submitting = executor.submit(freeze), executor.submit(submit)
        assert freezing.result() != submitting.result()


def test_renewed_attestation_cannot_drop_immutable_reserved_or_frozen_source_scope(
    service: BenchmarkService, tmp_path: Path
) -> None:
    development = dataset(
        service, tmp_path, "development", 1, source_namespace="development-source"
    )
    final = dataset(service, tmp_path, "final", 10, source_namespace="final-source")
    references(service, tmp_path, final, "test")
    service.register_dataset_history(principal(), development.dataset_id)
    original_scope = ("development-source", "final-source")
    service.attest(principal(), original_scope, "Original isolated fixture inventory.")
    reserved = service.reserve(principal(), "fresh", final.dataset_id, "final-test")
    registered = model(service, tmp_path, development)
    extra = dataset(service, tmp_path, "extra", 30, source_namespace="new-source")
    service.register_dataset_history(principal(), extra.dataset_id)
    incomplete_scope = ("development-source", "new-source")
    complete_scope = (*original_scope, "new-source")

    def attest(scope: tuple[str, ...]) -> None:
        service.attest(principal(), scope, "Renewed isolated fixture inventory.")

    incomplete = service.attest(
        principal(), incomplete_scope, "Incomplete benchmark-source inventory."
    )
    assert incomplete.registry.history_attestation is not None
    assert "final-source" not in incomplete.registry.history_attestation.source_namespaces
    assert reserved.reservation.inventory_attestation.source_namespaces == original_scope
    with pytest.raises(Forbidden, match="scope"):
        service.start_selection(principal(), "fresh")
    attest(complete_scope)
    service.start_selection(principal(), "fresh")
    attest(incomplete_scope)
    with pytest.raises(Forbidden, match="scope"):
        service.authorize_selection(principal(), training(development), development)
    with pytest.raises(Forbidden, match="scope"):
        service.freeze(principal(), "fresh", registered.model_version_id)
    attest(complete_scope)
    frozen = service.freeze(principal(), "fresh", registered.model_version_id)
    assert frozen.reservation.freeze is not None
    assert frozen.reservation.freeze.inventory_attestation.source_namespaces == complete_scope
    attest(incomplete_scope)
    with pytest.raises(Forbidden, match="scope"):
        service.open(principal(), "fresh")
    attest(complete_scope)
    service.open(principal(), "fresh")
    attest(incomplete_scope)
    with pytest.raises(Forbidden, match="scope"):
        service.authorize_final_evaluation(principal(), "fresh", registered.model_version_id)
    attest(complete_scope)
    assert (
        service.authorize_final_evaluation(
            principal(), "fresh", registered.model_version_id
        ).partition
        == "test"
    )
