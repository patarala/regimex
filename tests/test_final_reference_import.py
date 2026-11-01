"""Artificial human annotations verify sealed ingress, not independent accuracy."""

import json
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import SecretStr, ValidationError

psycopg = pytest.importorskip("psycopg", reason="install the service extra")
pytest.importorskip("procrastinate", reason="install the service extra")
pytest.importorskip("alembic", reason="install the service extra")

from test_admin_repository import sources

from regimex.admin_contracts import (
    DatasetRecord,
    DatasetSelection,
    DatasetSpec,
    FinalReferenceImportSpec,
    HumanAnnotation,
    ReferenceImportSpec,
    ReferenceSetRecord,
)
from regimex.application.admin_service import AdminService
from regimex.application.benchmark_service import BenchmarkService
from regimex.application.service import AnalysisService, materialize_object
from regimex.directions import (
    AnalysisConfig,
    Direction,
    DirectionAnnotations,
    DirectionInterval,
    summarize_interval,
)
from regimex.exports import ExportPayloadV2, seal_export
from regimex.infrastructure.admin_repository import AdminRepository
from regimex.infrastructure.database import DatabaseRepository
from regimex.infrastructure.object_store import LocalObjectStore
from regimex.model_registry import ArchitectureRegistry
from regimex.service_contracts import Forbidden, NotFound, Principal, ServiceError
from regimex.service_settings import ServiceSettings

REVIEWER = "artificial-reviewer"


@dataclass
class ImportContext:
    admin: AdminService
    benchmark: BenchmarkService
    principal: Principal
    dataset: DatasetRecord


@pytest.fixture
def context(tmp_path: Path) -> Iterator[ImportContext]:
    dsn = os.environ.get("REGIMEX_TEST_DATABASE_DSN")
    if not dsn:
        pytest.skip("set REGIMEX_TEST_DATABASE_DSN to a disposable PostgreSQL database")
    repository = DatabaseRepository(dsn, schema="regimex_human_final_" + uuid4().hex)
    try:
        repository.migrate()
        identity, _token = repository.issue_token("final-import-owner", "admin")
        settings = ServiceSettings(
            database_dsn=SecretStr(dsn),
            database_schema=repository.schema,
            object_root=tmp_path / "objects",
            work_root=tmp_path / "work",
        )
        store = LocalObjectStore(
            settings.object_root, max_object_bytes=16 * 1024**2, max_total_bytes=128 * 1024**2
        )
        base = AnalysisService(
            repository, store, ArchitectureRegistry.discover(), settings.work_root
        )
        catalog = AdminRepository(repository)
        admin = AdminService(base, catalog, settings)
        benchmark = BenchmarkService(repository, catalog, store, settings.work_root)
        exports = [item.export for item in sources()]
        value = exports[0].model_dump(mode="json", exclude={"content_sha256"})
        value["source"].update(run_id="extra-run", snapshot_id="extra-snapshot")
        value["instruments"][0]["initial_price"] += 10.0
        for row in value["observations"]:
            row["last_price"] = round(row["last_price"] + 10.0, 2)
        exports.append(seal_export(ExportPayloadV2.model_validate_json(json.dumps(value))))
        snapshots = tuple(base.create_snapshot(identity, export) for export in exports)
        dataset = admin.create_dataset(
            identity,
            DatasetSpec(
                dataset_id="human-final-fixture",
                members=tuple(
                    DatasetSelection(
                        snapshot_id=snapshot.snapshot_id,
                        partition="train" if index == 0 else "validation" if index == 1 else "test",
                    )
                    for index, snapshot in enumerate(snapshots)
                ),
            ),
        )
        yield ImportContext(admin, benchmark, identity, dataset)
    finally:
        with psycopg.connect(dsn, autocommit=True) as connection:
            connection.execute(
                psycopg.sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                    psycopg.sql.Identifier(repository.schema)
                )
            )


def final_spec(
    context: ImportContext,
    *,
    reference_id: str = "sealed-human-final",
    dataset: DatasetRecord | None = None,
) -> FinalReferenceImportSpec:
    dataset = context.dataset if dataset is None else dataset
    annotations = []
    for snapshot_id, member in zip(dataset.snapshot_ids, dataset.manifest.members, strict=True):
        if member.partition != "test":
            continue
        export = context.admin.analysis_service.read_snapshot(context.principal, snapshot_id)
        annotations.append(
            HumanAnnotation(
                snapshot_id=snapshot_id,
                annotation=DirectionAnnotations(
                    experiment_id=export.source.snapshot_id,
                    policy="human-directions/1.0",
                    reference_kind="human",
                    reviewer=REVIEWER,
                    policy_config=AnalysisConfig().model_dump(mode="json"),
                    intervals=(
                        DirectionInterval(
                            instrument_id="A/Q",
                            start=0,
                            stop=64,
                            direction=Direction.GROWTH,
                            status="observed",
                            statistics=summarize_interval(export.observations, 0, 64),
                        ),
                    ),
                ),
            )
        )
    return FinalReferenceImportSpec(
        dataset_id=dataset.dataset_id,
        reference_set_id=reference_id,
        reviewer=REVIEWER,
        blind_review=True,
        independent=True,
        annotations=tuple(annotations),
    )


def payload_paths(context: ImportContext) -> tuple[Path, ...]:
    store = context.admin.analysis_service.store
    assert isinstance(store, LocalObjectStore)
    return tuple(
        sorted(
            path.relative_to(store.root)
            for path in store.root.rglob("*")
            if path.is_file() and not path.name.startswith(".regimex-")
        )
    )


def test_final_human_import_preserves_provenance_and_remains_sealed(
    context: ImportContext,
    tmp_path: Path,
) -> None:
    spec = final_spec(context)
    record = context.admin.import_final_references(context.principal, spec)
    assert record.references.partition == "test"
    assert record.references.kind == "human"
    assert record.references.reviewer == REVIEWER
    assert record.references.blind_review and record.references.independent
    assert len(record.annotations) == 2
    for index, (info, expected) in enumerate(
        zip(record.annotations, spec.annotations, strict=True)
    ):
        path = tmp_path / f"sealed-{index}.json"
        materialize_object(context.admin.analysis_service.store, info.object, path)
        assert DirectionAnnotations.model_validate_json(path.read_bytes()) == expected.annotation
    with pytest.raises(Forbidden):
        context.admin.get_reference_annotations(
            context.principal, record.references.reference_set_id
        )
    assert context.admin.list_reference_sets(context.principal) == ()
    namespaces = tuple(
        sorted(member.source_namespace for member in context.dataset.manifest.members)
    )
    context.benchmark.attest(
        context.principal, namespaces, "Artificial ingestion fixture, named sources only."
    )
    view = context.benchmark.reserve(
        context.principal,
        "artificial-final-benchmark",
        context.dataset.dataset_id,
        record.references.reference_set_id,
    )
    assert view.reservation.reference.kind == "human"
    assert view.reservation.opened_at is None
    assert view.reservation.selection_started_at is None
    assert view.frozen_choice is None
    # Ingress and reservation do not evaluate a model or establish human accuracy.
    assert context.admin.repository.list_runs(context.principal) == ()


def test_ordinary_import_cannot_accept_a_final_dto_or_constructed_partition(
    context: ImportContext,
) -> None:
    spec = final_spec(context)
    with pytest.raises(ValidationError):
        ReferenceImportSpec.model_validate_json(spec.model_dump_json())
    ordinary = ReferenceImportSpec.model_construct(**spec.model_dump())
    before = payload_paths(context)
    with pytest.raises(ServiceError) as error:
        context.admin.import_references(context.principal, ordinary)
    assert error.value.status_code == 422
    assert payload_paths(context) == before
    assert context.admin.repository.list_reference_sets(context.principal, allow_test=True) == ()


@pytest.mark.parametrize("invalid", ["coverage", "statistics", "reviewer", "independence"])
def test_invalid_human_final_input_commits_neither_references_nor_objects(
    context: ImportContext,
    invalid: str,
) -> None:
    spec = final_spec(context)
    if invalid == "coverage":
        spec = spec.model_copy(update={"annotations": spec.annotations[:1]})
    elif invalid == "statistics":
        item = spec.annotations[-1]
        interval = item.annotation.intervals[0]
        statistics = interval.statistics.model_copy(update={"trade_count": 999})
        document = item.annotation.model_copy(
            update={"intervals": (interval.model_copy(update={"statistics": statistics}),)}
        )
        changed = item.model_copy(update={"annotation": document})
        spec = spec.model_copy(update={"annotations": (*spec.annotations[:-1], changed)})
    elif invalid == "reviewer":
        spec = spec.model_copy(update={"reviewer": "a-different-reviewer"})
    else:
        spec = spec.model_copy(update={"blind_review": False})
    before = payload_paths(context)
    with pytest.raises(ServiceError) as error:
        context.admin.import_final_references(context.principal, spec)
    assert error.value.status_code == 422
    assert payload_paths(context) == before
    assert context.admin.repository.list_reference_sets(context.principal, allow_test=True) == ()


@pytest.mark.parametrize("other", ["user", "another-owner"])
def test_final_import_enforces_admin_role_and_owner_before_storing(
    context: ImportContext, other: str
) -> None:
    identity = context.principal.model_copy(
        update={"role": "user"} if other == "user" else {"owner_id": "other-owner"}
    )
    before = payload_paths(context)
    with pytest.raises(Forbidden if other == "user" else NotFound):
        context.admin.import_final_references(identity, final_spec(context))
    assert payload_paths(context) == before


def test_previously_exposed_dataset_and_alias_cannot_receive_human_final_references(
    context: ImportContext,
) -> None:
    context.benchmark.register_dataset_history(context.principal, context.dataset.dataset_id)
    alias = context.admin.create_dataset(
        context.principal,
        DatasetSpec(
            dataset_id="history-alias",
            members=tuple(
                DatasetSelection(snapshot_id=identity, partition=member.partition)
                for identity, member in zip(
                    context.dataset.snapshot_ids, context.dataset.manifest.members, strict=True
                )
            ),
        ),
    )
    before = payload_paths(context)
    for dataset in (context.dataset, alias):
        with pytest.raises(Forbidden):
            context.admin.import_final_references(
                context.principal, final_spec(context, dataset=dataset)
            )
    assert payload_paths(context) == before
    assert context.admin.repository.list_reference_sets(context.principal, allow_test=True) == ()


def test_reserved_dataset_cannot_receive_replacement_human_final_references(
    context: ImportContext,
) -> None:
    record = context.admin.import_final_references(context.principal, final_spec(context))
    namespaces = tuple(
        sorted(member.source_namespace for member in context.dataset.manifest.members)
    )
    context.benchmark.attest(context.principal, namespaces, "Artificial ingestion fixture only.")
    context.benchmark.reserve(
        context.principal,
        "artificial-final-benchmark",
        context.dataset.dataset_id,
        record.references.reference_set_id,
    )
    before = payload_paths(context)
    with pytest.raises(Forbidden):
        context.admin.import_final_references(
            context.principal, final_spec(context, reference_id="replacement")
        )
    assert payload_paths(context) == before


def test_history_race_after_upload_cannot_commit_a_final_reference_set(
    context: ImportContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = context.admin._human_reference_record

    def expose_after_upload(
        identity: Principal,
        spec: ReferenceImportSpec | FinalReferenceImportSpec,
    ) -> ReferenceSetRecord:
        record = original(identity, spec)
        context.benchmark.register_dataset_history(identity, context.dataset.dataset_id)
        return record

    monkeypatch.setattr(context.admin, "_human_reference_record", expose_after_upload)
    with pytest.raises(Forbidden):
        context.admin.import_final_references(context.principal, final_spec(context))
    # Uploaded immutable objects are not visible reference resources. Reconciliation
    # can later remove these unreferenced objects using its own authorization rules.
    assert context.admin.repository.list_reference_sets(context.principal, allow_test=True) == ()


def test_http_final_human_ingress_replays_sealed_metadata_and_rejects_bypasses(
    context: ImportContext,
) -> None:
    pytest.importorskip("fastapi", reason="install the service extra")
    pytest.importorskip("httpx2", reason="install the service-test dependency group")
    from fastapi.testclient import TestClient

    from regimex.api import create_app

    base = context.admin.analysis_service
    token = base.repository.create_token(context.principal.owner_id, "admin")
    user_token = base.repository.create_token(context.principal.owner_id, "user")
    authorization = {"Authorization": f"Bearer {token}", "Idempotency-Key": "human-final-http"}
    body = final_spec(context).model_dump(mode="json", exclude={"reference_set_id"})
    app = create_app(base, admin_service=context.admin, benchmark_service=context.benchmark)
    with TestClient(app) as client:
        created = client.post("/api/v1/benchmarks/reference-sets", json=body, headers=authorization)
        assert created.status_code == 201, created.text
        metadata = created.json()
        assert metadata["kind"] == "human"
        assert metadata["partition"] == "test"
        assert metadata["annotations_sealed"] is True
        assert "annotations" not in metadata
        assert "macro_f1" not in created.text
        payloads = payload_paths(context)
        replay = client.post("/api/v1/benchmarks/reference-sets", json=body, headers=authorization)
        assert replay.status_code == 201
        assert replay.json() == metadata
        assert payload_paths(context) == payloads
        changed = {**body, "independent": False}
        conflict = client.post(
            "/api/v1/benchmarks/reference-sets", json=changed, headers=authorization
        )
        assert conflict.status_code == 409, conflict.text
        assert payload_paths(context) == payloads
        assert (
            client.get(
                f"/api/v1/reference-sets/{metadata['reference_set_id']}/annotations",
                headers=authorization,
            ).status_code
            == 403
        )
        ordinary = client.post(
            "/api/v1/reference-sets", json={**body, "mode": "human_import"}, headers=authorization
        )
        assert ordinary.status_code == 422
        forged = client.post(
            "/api/v1/benchmarks/reference-sets",
            json={**body, "reference_set_id": "client-chosen-reference"},
            headers=authorization,
        )
        assert forged.status_code == 422
        denied = client.post(
            "/api/v1/benchmarks/reference-sets",
            json=body,
            headers={"Authorization": f"Bearer {user_token}", "Idempotency-Key": "forbidden"},
        )
        assert denied.status_code == 403
        assert payload_paths(context) == payloads
