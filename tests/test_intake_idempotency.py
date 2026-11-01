"""Real transactions bind synchronous retries to the first committed resource."""

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Never

import pytest

pytest.importorskip("psycopg", reason="install the service extra")
pytest.importorskip("alembic", reason="install the service extra")
pytest.importorskip("procrastinate", reason="install the service extra")

from test_admin_repository import admitted_dataset, dataset_records, principal, reference_record
from test_admin_repository import repository as repository
from test_final_reference_import import ImportContext, final_spec, payload_paths
from test_final_reference_import import context as context

from regimex.admin_contracts import (
    DatasetRecord,
    DatasetSelection,
    DatasetSpec,
    HumanAnnotation,
    ReferenceImportSpec,
    ReferenceSetRecord,
)
from regimex.data_catalog import ReferencePayload, content_hash, seal_dataset, seal_reference_set
from regimex.directions import DirectionAnnotations, DirectionInterval, summarize_interval
from regimex.infrastructure.intake_repository import IntakeKind, IntakeReceipts, IntakeRequest
from regimex.service_contracts import Conflict, Forbidden, NotFound, ServiceError

if TYPE_CHECKING:
    from regimex.infrastructure.admin_repository import AdminRepository, Connection


def renamed_dataset(record: DatasetRecord, identity: str) -> DatasetRecord:
    return record.model_copy(
        update={"dataset_id": identity, "manifest": seal_dataset(identity, record.manifest.members)}
    )


def renamed_reference(record: ReferenceSetRecord, identity: str) -> ReferenceSetRecord:
    payload = ReferencePayload.model_validate_json(
        record.references.model_dump_json(exclude={"content_sha256"})
    )
    return record.model_copy(
        update={
            "references": seal_reference_set(
                payload.model_copy(update={"reference_set_id": identity})
            )
        }
    )


def test_dataset_receipt_replays_original_identity_and_conflicts_before_validation(
    repository: AdminRepository,
) -> None:
    record, snapshots, bindings = dataset_records("first-server-id")
    for snapshot in snapshots:
        repository.database.add_snapshot(snapshot)
    first = repository.add_dataset(
        principal(), record, bindings=bindings, idempotency_key="retry", request_sha256="a" * 64
    )
    changed_server_id = renamed_dataset(record, "second-server-id")
    assert (
        repository.add_dataset(
            principal(), changed_server_id, idempotency_key="retry", request_sha256="a" * 64
        )
        == first
    )
    with pytest.raises(Conflict):
        repository.add_dataset(
            principal(), changed_server_id, idempotency_key="retry", request_sha256="b" * 64
        )
    assert repository.list_datasets(principal()) == (first,)
    second = repository.add_dataset(
        principal(), changed_server_id, idempotency_key="new-key", request_sha256="b" * 64
    )
    assert second.dataset_id == "second-server-id"


def test_receipts_are_scoped_by_owner_and_operation(repository: AdminRepository) -> None:
    first, snapshots, bindings = dataset_records()
    for snapshot in snapshots:
        repository.database.add_snapshot(snapshot)
    repository.add_dataset(
        principal(), first, bindings=bindings, idempotency_key="shared", request_sha256="a" * 64
    )
    reference = repository.add_reference_set(
        principal(),
        reference_record(first, "validation"),
        idempotency_key="shared",
        request_sha256="b" * 64,
    )
    assert (
        repository.find_reference_replay(
            principal(), idempotency_key="shared", request_sha256="b" * 64
        )
        == reference
    )
    assert (
        repository.find_reference_replay(
            principal(), idempotency_key="shared", request_sha256="b" * 64, final=True
        )
        is None
    )
    other, snapshots, bindings = dataset_records("another-owner-resource", owner="other")
    for snapshot in snapshots:
        repository.database.add_snapshot(snapshot)
    assert (
        repository.add_dataset(
            principal("other"),
            other,
            bindings=bindings,
            idempotency_key="shared",
            request_sha256="c" * 64,
        )
        == other
    )
    with pytest.raises(NotFound):
        repository.get_dataset(principal("other"), first.dataset_id)
    with pytest.raises(Forbidden):
        repository.find_dataset_replay(
            principal(admin=False), idempotency_key="shared", request_sha256="a" * 64
        )


def test_concurrent_dataset_retries_return_one_resource(repository: AdminRepository) -> None:
    record, snapshots, bindings = dataset_records()
    for snapshot in snapshots:
        repository.database.add_snapshot(snapshot)

    def submit(index: int) -> DatasetRecord:
        return repository.add_dataset(
            principal(),
            renamed_dataset(record, f"server-{index}"),
            bindings=bindings,
            idempotency_key="simultaneous",
            request_sha256="a" * 64,
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        records = tuple(pool.map(submit, range(4)))
    assert all(item == records[0] for item in records)
    assert repository.list_datasets(principal()) == (records[0],)
    with repository.database.connection() as connection:
        assert connection.execute("SELECT count(*) FROM regimex_intake_keys").fetchone() == (1,)


def test_concurrent_reference_retries_return_original_resource(
    repository: AdminRepository,
) -> None:
    dataset = admitted_dataset(repository)
    record = reference_record(dataset, "validation")

    def submit(index: int) -> ReferenceSetRecord:
        return repository.add_reference_set(
            principal(),
            renamed_reference(record, f"reference-server-{index}"),
            idempotency_key="reference-retry",
            request_sha256="a" * 64,
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        records = tuple(pool.map(submit, range(4)))
    assert all(item == records[0] for item in records)
    assert repository.list_reference_sets(principal()) == (records[0],)
    with pytest.raises(Conflict):
        repository.add_reference_set(
            principal(),
            record,
            idempotency_key="reference-retry",
            request_sha256="b" * 64,
        )


@pytest.mark.parametrize("resource", ["dataset", "reference"])
def test_receipt_failure_rolls_back_resource_and_all_bindings(
    repository: AdminRepository, monkeypatch: pytest.MonkeyPatch, resource: str
) -> None:
    record, snapshots, bindings = dataset_records()
    for snapshot in snapshots:
        repository.database.add_snapshot(snapshot)
    if resource == "reference":
        repository.add_dataset(principal(), record, bindings=bindings)

    def fail_receipt(
        connection: Connection,
        owner_id: str,
        kind: IntakeKind,
        request: IntakeRequest,
        resource_id: str,
    ) -> Never:
        # Insert the receipt before failing to prove neither side can commit.
        original(connection, owner_id, kind, request, resource_id)
        raise RuntimeError("injected failure after receipt insertion")

    original = IntakeReceipts.add
    monkeypatch.setattr(IntakeReceipts, "add", staticmethod(fail_receipt))
    with pytest.raises(RuntimeError, match="injected failure"):
        if resource == "dataset":
            repository.add_dataset(
                principal(),
                record,
                bindings=bindings,
                idempotency_key="rollback",
                request_sha256="a" * 64,
            )
        else:
            repository.add_reference_set(
                principal(),
                reference_record(record, "validation"),
                idempotency_key="rollback",
                request_sha256="a" * 64,
            )
    with repository.database.connection() as connection:
        assert connection.execute("SELECT count(*) FROM regimex_intake_keys").fetchone() == (0,)
        if resource == "dataset":
            assert connection.execute("SELECT count(*) FROM regimex_datasets").fetchone() == (0,)
            assert connection.execute(
                "SELECT count(*) FROM regimex_catalog_bindings"
            ).fetchone() == (0,)
        else:
            assert connection.execute("SELECT count(*) FROM regimex_reference_sets").fetchone() == (
                0,
            )
            assert connection.execute(
                "SELECT count(*) FROM regimex_reference_objects"
            ).fetchone() == (0,)
    monkeypatch.setattr(IntakeReceipts, "add", staticmethod(original))
    if resource == "dataset":
        repository.add_dataset(
            principal(),
            record,
            bindings=bindings,
            idempotency_key="rollback",
            request_sha256="a" * 64,
        )
    else:
        repository.add_reference_set(
            principal(),
            reference_record(record, "validation"),
            idempotency_key="rollback",
            request_sha256="a" * 64,
        )


def test_dataset_service_replay_does_not_read_snapshots_again(
    context: ImportContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = DatasetSpec(
        dataset_id="new-server-id",
        members=tuple(
            DatasetSelection(snapshot_id=snapshot_id, partition=member.partition)
            for snapshot_id, member in zip(
                context.dataset.snapshot_ids, context.dataset.manifest.members, strict=True
            )
        ),
    )
    first = context.admin.create_dataset(context.principal, spec, idempotency_key="dataset")

    def no_read(*_args: object, **_kwargs: object) -> Never:
        raise AssertionError("replay must not materialize snapshots")

    monkeypatch.setattr(context.admin.analysis_service, "read_snapshot", no_read)
    assert context.admin.create_dataset(context.principal, spec, idempotency_key="dataset") == first
    with pytest.raises(Conflict):
        context.admin.create_dataset(
            context.principal,
            spec.model_copy(update={"dataset_id": "changed-body"}),
            idempotency_key="dataset",
        )


def test_ordinary_reference_service_replays_before_payload_reads(
    context: ImportContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot_id = next(
        snapshot_id
        for snapshot_id, member in zip(
            context.dataset.snapshot_ids, context.dataset.manifest.members, strict=True
        )
        if member.partition == "validation"
    )
    export = context.admin.analysis_service.read_snapshot(context.principal, snapshot_id)
    template = final_spec(context)
    annotation = template.annotations[0].annotation
    spec = ReferenceImportSpec(
        dataset_id=context.dataset.dataset_id,
        reference_set_id="ordinary-human",
        partition="validation",
        reviewer=template.reviewer,
        blind_review=template.blind_review,
        independent=template.independent,
        annotations=(
            HumanAnnotation(
                snapshot_id=snapshot_id,
                annotation=DirectionAnnotations(
                    experiment_id=export.source.snapshot_id,
                    reference_kind="human",
                    policy=annotation.policy,
                    reviewer=annotation.reviewer,
                    policy_config=annotation.policy_config,
                    intervals=(
                        DirectionInterval(
                            instrument_id="A/Q",
                            start=0,
                            stop=64,
                            direction=annotation.intervals[0].direction,
                            status="observed",
                            statistics=summarize_interval(export.observations, 0, 64),
                        ),
                    ),
                ),
            ),
        ),
    )
    first = context.admin.import_references(context.principal, spec, idempotency_key="ordinary")

    def no_read(*_args: object, **_kwargs: object) -> Never:
        raise AssertionError("replay must not re-read reference inputs")

    monkeypatch.setattr(context.admin.analysis_service, "read_snapshot", no_read)
    assert (
        context.admin.import_references(context.principal, spec, idempotency_key="ordinary")
        == first
    )
    with pytest.raises(Conflict):
        context.admin.import_references(
            context.principal,
            spec.model_copy(update={"reviewer": "changed-reviewer"}),
            idempotency_key="ordinary",
        )


def test_final_reference_replay_survives_reservation_and_conflicts_before_upload(
    context: ImportContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = final_spec(context)
    checksum = content_hash(spec.model_dump(mode="json", exclude={"reference_set_id"}))
    first = context.admin.import_final_references(
        context.principal, spec, idempotency_key="final", request_sha256=checksum
    )
    context.benchmark.attest(
        context.principal,
        tuple(sorted({member.source_namespace for member in context.dataset.manifest.members})),
        "Artificial fixture inventory only.",
    )
    context.benchmark.reserve(
        context.principal, "sealed", context.dataset.dataset_id, first.references.reference_set_id
    )
    before = payload_paths(context)

    def no_upload(*_args: object, **_kwargs: object) -> Never:
        raise AssertionError("replay must not read or upload sealed annotations")

    monkeypatch.setattr(context.admin, "_human_reference_record", no_upload)
    assert (
        context.admin.import_final_references(
            context.principal,
            spec.model_copy(update={"reference_set_id": "another-server-generated-id"}),
            idempotency_key="final",
            request_sha256=checksum,
        )
        == first
    )
    with pytest.raises(Conflict):
        context.admin.import_final_references(
            context.principal,
            spec.model_copy(update={"independent": False}),
            idempotency_key="final",
            request_sha256="f" * 64,
        )
    assert payload_paths(context) == before


def test_final_reference_receipt_failure_keeps_upload_unpublished(
    context: ImportContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = IntakeReceipts.add

    def fail_receipt(
        connection: Connection,
        owner_id: str,
        kind: IntakeKind,
        request: IntakeRequest,
        resource_id: str,
    ) -> Never:
        original(connection, owner_id, kind, request, resource_id)
        raise RuntimeError("injected final receipt failure")

    monkeypatch.setattr(IntakeReceipts, "add", staticmethod(fail_receipt))
    with pytest.raises(RuntimeError, match="injected final receipt failure"):
        context.admin.import_final_references(
            context.principal, final_spec(context), idempotency_key="final-rollback"
        )
    assert context.admin.repository.list_reference_sets(context.principal, allow_test=True) == ()
    with context.admin.repository.database.connection() as connection:
        assert connection.execute("SELECT count(*) FROM regimex_intake_keys").fetchone() == (0,)
        assert connection.execute("SELECT count(*) FROM regimex_reference_objects").fetchone() == (
            0,
        )
    # An immutable upload alone is not an admitted resource. A later authorized
    # retry can reuse it; reconciliation owns eventual orphan removal.
    assert payload_paths(context)


@pytest.mark.parametrize(
    ("key", "checksum"), [(None, "a" * 64), ("", "a" * 64), ("key", "not-sha256")]
)
def test_unbound_or_invalid_receipt_settings_are_rejected(
    repository: AdminRepository, key: str | None, checksum: str
) -> None:
    dataset = admitted_dataset(repository)
    with pytest.raises(ServiceError) as error:
        repository.add_dataset(principal(), dataset, idempotency_key=key, request_sha256=checksum)
    assert error.value.status_code == 422
