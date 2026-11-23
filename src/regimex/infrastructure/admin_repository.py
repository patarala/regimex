"""Transactional administrative resources and fenced lifecycle publication."""

import hashlib
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import NamedTuple, cast
from uuid import UUID, uuid4

import psycopg
from psycopg.types.json import Json, Jsonb
from pydantic import JsonValue, RootModel, TypeAdapter

from regimex.admin_contracts import (
    MAX_ARCHIVE_SELECTION,
    MAX_PROGRESS_BATCH,
    MAX_PROGRESS_POINTS,
    AdminKind,
    AdminRunRecord,
    AdminSpec,
    CatalogBinding,
    DatasetRecord,
    EvaluationRecord,
    EvaluationSpec,
    GenerationSpec,
    HistorySeries,
    ImportedTrainingSpec,
    ProgressPoint,
    PublicationSpec,
    ReferenceGenerationSpec,
    ReferenceSetRecord,
    RequestDigest,
    TrainingArchiveFilter,
    TrainingMetrics,
    TrainingProgress,
    TrainingRunName,
    TrainingSpec,
    admin_request_hash,
)
from regimex.contracts import MAX_INPUT_BYTES
from regimex.data_catalog import validate_reference_membership
from regimex.directions import DIRECTION_POLICY
from regimex.external_evaluations import ExternalEvaluation
from regimex.external_latency import ExternalLatencyEvidence
from regimex.history_contracts import DisplayNameUpdate, validate_display_name
from regimex.infrastructure.database import DatabaseRepository, _read_operation
from regimex.infrastructure.intake_repository import IntakeReceipts, IntakeRequest
from regimex.infrastructure.training_metrics import read_training_metrics, store_training_metrics
from regimex.operations import TERMINAL_STATES, Operation, ResolvedModel, succeed
from regimex.report_contracts import MAX_REPORT_BYTES
from regimex.service_contracts import (
    CapacityExceeded,
    Conflict,
    Forbidden,
    ModelRecord,
    NotFound,
    Principal,
    SnapshotRecord,
    StaleAttempt,
    UnsupportedContract,
)
from regimex.storage import json_bytes
from regimex.training_statistics import TrainingFitStatistics

_SPEC = RootModel[AdminSpec]
_RECORDED_SPEC = RootModel[AdminSpec | ImportedTrainingSpec]
_EXTERNAL_EVALUATIONS = TypeAdapter(tuple[ExternalEvaluation, ...])
_POINT = RootModel[ProgressPoint]
_JSON = TypeAdapter(dict[str, JsonValue])
_CHART_FIELDS = {
    "epoch",
    "update",
    "regularization",
    "trees",
    "max_depth",
    "seed",
    "training_loss",
    "stop_reason",
    "validation_segment_macro_f1",
}
Connection = psycopg.Connection[tuple[object, ...]]
_RUN_SELECT = (
    "SELECT r.operation_id,r.request,r.output,o.document,t.display_name,"
    "COALESCE(t.name_revision,0),m.weights_deletion_id,t.archived_at,t.external_evaluations,"
    "t.external_latency FROM regimex_admin_runs r "
    "JOIN regimex_operations o ON r.operation_id=o.operation_id "
    "LEFT JOIN regimex_training_runs t ON t.operation_id=r.operation_id "
    "LEFT JOIN regimex_models m ON m.model_version_id=t.committed_model_version_id"
)


class CatalogSnapshot(NamedTuple):
    snapshot: SnapshotRecord
    binding: CatalogBinding | None


def _read_run(row: tuple[object, ...]) -> AdminRunRecord:
    (
        run_id,
        request,
        output,
        document,
        display_name,
        name_revision,
        weights_deletion_id,
        archived_at,
        external_evaluations,
        external_latency,
    ) = row
    operation = _read_operation(document)
    result = AdminRunRecord(
        run_id=cast(UUID, run_id),
        operation=operation,
        spec=_RECORDED_SPEC.model_validate_json(json_bytes(request)).root,
        output=None if output is None else _JSON.validate_python(output, strict=True),
        display_name=cast(str | None, display_name),
        name_revision=cast(int, name_revision),
        weights_deletion_id=cast(UUID | None, weights_deletion_id),
        archived_at=cast(datetime | None, archived_at),
        external_evaluations=_EXTERNAL_EVALUATIONS.validate_json(json_bytes(external_evaluations))
        if external_evaluations
        else (),
        external_latency=ExternalLatencyEvidence.model_validate_json(json_bytes(external_latency))
        if external_latency is not None
        else None,
    )
    if (
        result.output is not None
        and hashlib.sha256(json_bytes(result.output)).hexdigest() != operation.result_sha256
    ):
        raise Conflict("Administrative output differs from its committed digest.")
    return result


def _read_reference(document: object) -> ReferenceSetRecord:
    if isinstance(document, dict):
        references = document.get("references")
        if (
            isinstance(references, dict)
            and references.get("kind") == "automatic"
            and references.get("policy") != DIRECTION_POLICY
        ):
            raise UnsupportedContract()
    return ReferenceSetRecord.model_validate_json(json_bytes(document))


class AdminRepository:
    def __init__(self, database: DatabaseRepository) -> None:
        self.database = database

    @staticmethod
    def _owner(principal: Principal, owner_id: str) -> None:
        DatabaseRepository._admin(principal)
        if principal.owner_id != owner_id:
            raise NotFound()

    @staticmethod
    def _bounded(
        value: dict[str, JsonValue],
        *,
        limit: int = 64 * 1024,
        description: str = "Administrative metadata",
    ) -> dict[str, JsonValue]:
        checked = _JSON.validate_python(value, strict=True)
        if len(json_bytes(checked)) > limit:
            raise CapacityExceeded(f"{description} must not exceed {limit} bytes.")
        return checked

    @staticmethod
    def _catalog_inputs(
        connection: Connection, owner_id: str, snapshot_ids: tuple[UUID, ...]
    ) -> tuple[CatalogSnapshot, ...]:
        rows = connection.execute(
            "SELECT s.document, b.document FROM unnest(%s::uuid[]) WITH ORDINALITY AS r(id,pos) "
            "JOIN regimex_snapshots s ON s.snapshot_id=r.id AND s.owner_id=%s "
            "LEFT JOIN regimex_catalog_bindings b USING(snapshot_id) ORDER BY r.pos",
            (list(snapshot_ids), owner_id),
        ).fetchall()
        if len(rows) != len(snapshot_ids):
            raise NotFound()
        inputs = tuple(
            CatalogSnapshot(
                SnapshotRecord.model_validate_json(json_bytes(snapshot)),
                CatalogBinding.model_validate_json(json_bytes(binding))
                if binding is not None
                else None,
            )
            for snapshot, binding in rows
        )
        for requested, (snapshot, binding) in zip(snapshot_ids, inputs, strict=True):
            if snapshot.owner_id != owner_id:
                raise NotFound()
            if snapshot.snapshot_id != requested or (binding and binding.snapshot_id != requested):
                raise Conflict("Catalog metadata differs from the registered snapshot identity.")
        return inputs

    def _binding(self, connection: Connection, owner_id: str, snapshot_id: UUID) -> CatalogBinding:
        binding = self._catalog_inputs(connection, owner_id, (snapshot_id,))[0].binding
        if binding is None:
            raise NotFound("The snapshot has no admitted catalog binding.")
        return binding

    def catalog_inputs(
        self, principal: Principal, snapshot_ids: tuple[UUID, ...]
    ) -> tuple[CatalogSnapshot, ...]:
        self.database._admin(principal)
        with self.database.connection() as connection:
            return self._catalog_inputs(connection, principal.owner_id, snapshot_ids)

    def get_binding(self, principal: Principal, snapshot_id: UUID) -> CatalogBinding | None:
        return self.catalog_inputs(principal, (snapshot_id,))[0].binding

    def _add_binding(
        self, connection: Connection, owner_id: str, binding: CatalogBinding
    ) -> CatalogBinding:
        binding = CatalogBinding.model_validate_json(binding.model_dump_json())
        snapshot = self.database._snapshot(connection, binding.snapshot_id)
        if snapshot.owner_id != owner_id:
            raise NotFound()
        connection.execute(
            "INSERT INTO regimex_catalog_bindings "
            "(snapshot_id,source_namespace,group_namespace,group_id,seed,document) "
            "VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT (snapshot_id) DO NOTHING",
            (
                binding.snapshot_id,
                binding.source_namespace,
                binding.split_group.namespace,
                binding.split_group.group_id,
                binding.split_group.seed,
                Jsonb(binding.model_dump(mode="json")),
            ),
        )
        existing = self._binding(connection, owner_id, binding.snapshot_id)
        if existing != binding:
            raise Conflict("An admitted snapshot source/group binding is immutable.")
        return existing

    def add_binding(self, principal: Principal, binding: CatalogBinding) -> CatalogBinding:
        self.database._admin(principal)
        with self.database.connection() as connection:
            return self._add_binding(connection, principal.owner_id, binding)

    @staticmethod
    def _dataset(connection: Connection, owner_id: str, dataset_id: str) -> DatasetRecord:
        row = connection.execute(
            "SELECT document FROM regimex_datasets WHERE dataset_id=%s AND owner_id=%s",
            (dataset_id, owner_id),
        ).fetchone()
        if row is None:
            raise NotFound()
        return DatasetRecord.model_validate_json(json_bytes(row[0]))

    def get_dataset(self, principal: Principal, dataset_id: str) -> DatasetRecord:
        self.database._admin(principal)
        with self.database.connection() as connection:
            return self._dataset(connection, principal.owner_id, dataset_id)

    def list_datasets(
        self,
        principal: Principal,
        *,
        development_only: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[DatasetRecord, ...]:
        self.database._admin(principal)
        self.database._page(limit, offset)
        with self.database.connection() as connection:
            query = "SELECT d.document FROM regimex_datasets d WHERE d.owner_id=%s"
            if development_only:
                query += (
                    " AND EXISTS (SELECT 1 FROM regimex_benchmark_history_datasets h "
                    "WHERE h.dataset_id=d.dataset_id AND h.owner_id=d.owner_id "
                    "AND h.dataset_sha256=d.content_sha256)"
                )
            rows = connection.execute(
                query + " ORDER BY d.dataset_id LIMIT %s OFFSET %s",
                (principal.owner_id, limit, offset),
            ).fetchall()
            return tuple(DatasetRecord.model_validate_json(json_bytes(row[0])) for row in rows)

    def _add_dataset(
        self,
        connection: Connection,
        record: DatasetRecord,
        *,
        bindings: tuple[CatalogBinding, ...] = (),
        generation_run_id: UUID | None = None,
    ) -> DatasetRecord:
        record = DatasetRecord.model_validate_json(record.model_dump_json())
        self.database.lock_catalog_identity(connection, "dataset", record.dataset_id)
        reservation = connection.execute(
            "SELECT t.operation_id FROM regimex_generation_targets t WHERE t.dataset_id=%s "
            "AND NOT EXISTS (SELECT 1 FROM regimex_datasets d "
            "WHERE d.dataset_id=t.dataset_id AND d.owner_id=%s)",
            (record.dataset_id, record.owner_id),
        ).fetchone()
        if reservation is not None and reservation[0] != generation_run_id:
            raise Conflict("The dataset identity is reserved by a generation run.")
        for declared in bindings:
            self._add_binding(connection, record.owner_id, declared)
        inputs = self._catalog_inputs(connection, record.owner_id, record.snapshot_ids)
        for (snapshot, binding), member in zip(inputs, record.manifest.members, strict=True):
            if binding is None:
                raise NotFound("The snapshot has no admitted catalog binding.")
            if (
                snapshot.owner_id != record.owner_id
                or (member.source_namespace, member.split_group)
                != (binding.source_namespace, binding.split_group)
                or (member.snapshot_id, member.source_id, member.run_id, member.export_sha256)
                != (
                    snapshot.source_snapshot_id,
                    snapshot.source_id,
                    snapshot.run_id,
                    snapshot.content_sha256,
                )
            ):
                raise Conflict("Dataset membership differs from an admitted immutable snapshot.")
        if record.generation_provenance is not None:
            self.database._object(connection, record.generation_provenance)
        connection.execute(
            "INSERT INTO regimex_datasets (dataset_id,owner_id,content_sha256,split_sha256,"
            "generation_provenance_key,document) VALUES (%s,%s,%s,%s,%s,%s) "
            "ON CONFLICT (dataset_id) DO NOTHING",
            (
                record.dataset_id,
                record.owner_id,
                record.manifest.content_sha256,
                record.manifest.split_manifest_sha256,
                None if record.generation_provenance is None else record.generation_provenance.key,
                Json(record.model_dump(mode="json")),
            ),
        )
        existing = self._dataset(connection, record.owner_id, record.dataset_id)
        if existing.model_dump(exclude={"created_at"}) != record.model_dump(exclude={"created_at"}):
            raise Conflict("The dataset identity is immutable and has different membership.")
        for index, (snapshot_id, member) in enumerate(
            zip(record.snapshot_ids, record.manifest.members, strict=True)
        ):
            connection.execute(
                "INSERT INTO regimex_dataset_members (dataset_id,position,snapshot_id,partition) "
                "VALUES (%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                (record.dataset_id, index, snapshot_id, member.partition),
            )
        return existing

    def add_dataset(
        self,
        principal: Principal,
        record: DatasetRecord,
        *,
        bindings: tuple[CatalogBinding, ...] = (),
        idempotency_key: str | None = None,
        request_sha256: str | None = None,
    ) -> DatasetRecord:
        self._owner(principal, record.owner_id)
        request = IntakeRequest.optional(idempotency_key, request_sha256)
        with self.database.connection() as connection:
            if request is not None:
                previous = IntakeReceipts.find(
                    connection, principal.owner_id, "dataset_import", request, lock=True
                )
                if previous is not None:
                    return self._dataset(connection, principal.owner_id, previous)
            result = self._add_dataset(connection, record, bindings=bindings)
            if request is not None:
                IntakeReceipts.add(
                    connection, principal.owner_id, "dataset_import", request, result.dataset_id
                )
            return result

    def find_dataset_replay(
        self, principal: Principal, request: IntakeRequest | None
    ) -> DatasetRecord | None:
        self.database._admin(principal)
        if request is None:
            return None
        with self.database.connection() as connection:
            previous = IntakeReceipts.find(
                connection, principal.owner_id, "dataset_import", request
            )
            return (
                None
                if previous is None
                else self._dataset(connection, principal.owner_id, previous)
            )

    @staticmethod
    def _reference(
        connection: Connection, owner_id: str, reference_set_id: str, *, allow_test: bool = False
    ) -> ReferenceSetRecord:
        row = connection.execute(
            "SELECT partition,document FROM regimex_reference_sets "
            "WHERE reference_set_id=%s AND owner_id=%s",
            (reference_set_id, owner_id),
        ).fetchone()
        if row is None:
            raise NotFound()
        # Check the partition before parsing so callers cannot inspect sealed
        # final references through validation errors.
        if row[0] == "test" and not allow_test:
            raise Forbidden("Final-reference access requires the dedicated benchmark workflow.")
        record = _read_reference(row[1])
        if record.references.partition == "test" and not allow_test:
            raise Forbidden("Final-reference access requires the dedicated benchmark workflow.")
        return record

    def get_reference_set(
        self, principal: Principal, reference_set_id: str, *, allow_test: bool = False
    ) -> ReferenceSetRecord:
        self.database._admin(principal)
        with self.database.connection() as connection:
            return self._reference(
                connection, principal.owner_id, reference_set_id, allow_test=allow_test
            )

    def list_reference_sets(
        self,
        principal: Principal,
        *,
        dataset_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
        allow_test: bool = False,
    ) -> tuple[ReferenceSetRecord, ...]:
        self.database._admin(principal)
        self.database._page(limit, offset)
        with self.database.connection() as connection:
            query = "SELECT document FROM regimex_reference_sets WHERE owner_id=%s"
            values: list[object] = [principal.owner_id]
            if dataset_id is not None:
                self._dataset(connection, principal.owner_id, dataset_id)
                query += " AND dataset_id=%s"
                values.append(dataset_id)
            if not allow_test:
                query += " AND partition != 'test'"
            rows = connection.execute(
                query + " ORDER BY reference_set_id LIMIT %s OFFSET %s", [*values, limit, offset]
            ).fetchall()
            return tuple(_read_reference(row[0]) for row in rows)

    def _add_reference_set(
        self,
        connection: Connection,
        record: ReferenceSetRecord,
        *,
        reference_run_id: UUID | None = None,
    ) -> ReferenceSetRecord:
        record = ReferenceSetRecord.model_validate_json(record.model_dump_json())
        dataset = self._dataset(connection, record.owner_id, record.dataset_id)
        validate_reference_membership(dataset.manifest, record.references)
        self.database.lock_catalog_identity(
            connection, "reference", record.references.reference_set_id
        )
        reservation = connection.execute(
            "SELECT t.operation_id FROM regimex_reference_targets t WHERE t.reference_set_id=%s "
            "AND NOT EXISTS (SELECT 1 FROM regimex_reference_sets r "
            "WHERE r.reference_set_id=t.reference_set_id AND r.owner_id=%s)",
            (record.references.reference_set_id, record.owner_id),
        ).fetchone()
        if reservation is not None and reservation[0] != reference_run_id:
            raise Conflict("This reference identity is reserved by a generation run.")
        for annotation in record.annotations:
            if annotation.object.size_bytes > MAX_INPUT_BYTES:
                raise CapacityExceeded("Annotation content exceeds its object size bound.")
            self.database._object(connection, annotation.object)
        connection.execute(
            "INSERT INTO regimex_reference_sets (reference_set_id,owner_id,dataset_id,partition,"
            "content_sha256,document) VALUES (%s,%s,%s,%s,%s,%s) "
            "ON CONFLICT (reference_set_id) DO NOTHING",
            (
                record.references.reference_set_id,
                record.owner_id,
                record.dataset_id,
                record.references.partition,
                record.references.content_sha256,
                Json(record.model_dump(mode="json")),
            ),
        )
        existing = self._reference(
            connection, record.owner_id, record.references.reference_set_id, allow_test=True
        )
        if existing.model_dump(exclude={"created_at"}) != record.model_dump(exclude={"created_at"}):
            raise Conflict("The reference-set identity is immutable and has different content.")
        for index, member in enumerate(record.annotations):
            connection.execute(
                "INSERT INTO regimex_reference_objects "
                "(reference_set_id,position,source_namespace,source_snapshot_id,object_key) "
                "VALUES (%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                (
                    record.references.reference_set_id,
                    index,
                    member.source_namespace,
                    member.snapshot_id,
                    member.object.key,
                ),
            )
        return existing

    def add_reference_set(
        self,
        principal: Principal,
        record: ReferenceSetRecord,
        *,
        idempotency_key: str | None = None,
        request_sha256: RequestDigest | None = None,
    ) -> ReferenceSetRecord:
        self._owner(principal, record.owner_id)
        request = IntakeRequest.optional(idempotency_key, request_sha256)
        with self.database.connection() as connection:
            if request is not None:
                previous = IntakeReceipts.find(
                    connection, principal.owner_id, "reference_import", request, lock=True
                )
                if previous is not None:
                    return self._reference(connection, principal.owner_id, previous)
            result = self._add_reference_set(connection, record)
            if request is not None:
                IntakeReceipts.add(
                    connection,
                    principal.owner_id,
                    "reference_import",
                    request,
                    result.references.reference_set_id,
                )
            return result

    def find_reference_replay(
        self, principal: Principal, request: IntakeRequest | None, *, final: bool = False
    ) -> ReferenceSetRecord | None:
        self.database._admin(principal)
        if request is None:
            return None
        with self.database.connection() as connection:
            previous = IntakeReceipts.find(
                connection,
                principal.owner_id,
                "final_reference_import" if final else "reference_import",
                request,
            )
            return (
                None
                if previous is None
                else self._reference(connection, principal.owner_id, previous, allow_test=final)
            )

    def check_final_reference_import(self, principal: Principal, dataset_id: str) -> None:
        self.database._admin(principal)
        from regimex.infrastructure.benchmark_repository import BenchmarkRepository

        with self.database.connection() as connection:
            dataset = self._dataset(connection, principal.owner_id, dataset_id)
            BenchmarkRepository.validate_final_reference_import(
                connection, principal.owner_id, dataset
            )

    def add_final_reference_set(
        self,
        principal: Principal,
        record: ReferenceSetRecord,
        *,
        idempotency_key: str | None = None,
        request_sha256: RequestDigest | None = None,
    ) -> ReferenceSetRecord:
        self._owner(principal, record.owner_id)
        request = IntakeRequest.optional(idempotency_key, request_sha256)
        if record.references.partition != "test" or record.references.kind != "human":
            raise Forbidden("Sealed final import requires human test-partition references.")
        from regimex.infrastructure.benchmark_repository import BenchmarkRepository

        with self.database.connection() as connection:
            if request is not None:
                previous = IntakeReceipts.find(
                    connection, principal.owner_id, "final_reference_import", request, lock=True
                )
                if previous is not None:
                    return self._reference(
                        connection, principal.owner_id, previous, allow_test=True
                    )
            dataset = self._dataset(connection, principal.owner_id, record.dataset_id)
            # A benchmark may have been reserved during upload. Recheck while
            # holding the registry lock before publishing the reference set.
            BenchmarkRepository.validate_final_reference_import(
                connection, principal.owner_id, dataset
            )
            result = self._add_reference_set(connection, record)
            if request is not None:
                IntakeReceipts.add(
                    connection,
                    principal.owner_id,
                    "final_reference_import",
                    request,
                    result.references.reference_set_id,
                )
            return result

    def _run(
        self, connection: Connection, run_id: UUID, *, principal: Principal | None = None
    ) -> AdminRunRecord:
        # Read the state and output in the same snapshot if the worker finishes during polling.
        query = _RUN_SELECT + " WHERE r.operation_id=%s"
        values: list[object] = [run_id]
        if principal is not None:
            self.database._admin(principal)
            query += " AND o.owner_id=%s"
            values.append(principal.owner_id)
        row = connection.execute(query, values).fetchone()
        if row is None:
            raise NotFound()
        return _read_run(row)

    def get_run_internal(self, run_id: UUID) -> AdminRunRecord:
        with self.database.connection() as connection:
            return self._run(connection, run_id)

    def append_training_progress(
        self, run_id: UUID, token: UUID, first_cursor: int, rows: list[dict[str, JsonValue]]
    ) -> None:
        if not 0 < len(rows) <= MAX_PROGRESS_BATCH or not (
            1 <= first_cursor <= MAX_PROGRESS_POINTS - len(rows) + 1
        ):
            raise ValueError("Training progress batch exceeds the preview limits.")
        points = [
            _POINT.model_validate({k: v for k, v in row.items() if k in _CHART_FIELDS}).model_dump(
                mode="json"
            )
            for row in rows
        ]
        with self.database.connection(heartbeat=True) as connection:
            operation = self.database._operation(connection, run_id, lock=True)
            if (
                operation.kind != "training"
                or operation.attempt_token != token
                or operation.state != "running"
            ):
                raise StaleAttempt()
            connection.execute(
                "INSERT INTO regimex_training_progress (operation_id,attempt,position,measurement) "
                "SELECT %s,%s,%s+n-1,measurement FROM jsonb_array_elements(%s) "
                "WITH ORDINALITY AS batch(measurement,n) ON CONFLICT DO NOTHING",
                (run_id, operation.attempt, first_cursor, Jsonb(points)),
            )

    def training_progress(
        self, principal: Principal, run_id: UUID, *, attempt: int | None, after: int
    ) -> TrainingProgress:
        self.database._admin(principal)
        # Read attempt and samples in one snapshot. A concurrent retry can restart
        # the cursor without combining points from two fits.
        with self.database.connection() as connection:
            rows = connection.execute(
                "SELECT (o.document->>'attempt')::int, p.position,p.measurement "
                "FROM regimex_operations o LEFT JOIN LATERAL ("
                "SELECT position,measurement FROM regimex_training_progress "
                "WHERE operation_id=o.operation_id AND attempt=(o.document->>'attempt')::int "
                "AND position > CASE WHEN (o.document->>'attempt')::int=%s THEN %s ELSE 0 END "
                "ORDER BY position LIMIT %s) p ON TRUE "
                "WHERE o.operation_id=%s AND o.owner_id=%s AND o.kind='training' "
                "ORDER BY p.position",
                (attempt, after, MAX_PROGRESS_BATCH + 1, run_id, principal.owner_id),
            ).fetchall()
        if not rows:
            raise NotFound()
        current_attempt = cast(int, rows[0][0])
        cursor = after if attempt == current_attempt else 0
        history: dict[HistorySeries, list[ProgressPoint]] = {}
        for _, position, measurement in rows[:MAX_PROGRESS_BATCH]:
            if position is not None:
                point = _POINT.model_validate(measurement).root
                history.setdefault(point.series, []).append(point)
                cursor = cast(int, position)
        return TrainingProgress(
            attempt=current_attempt,
            cursor=cursor,
            history=history,
            has_more=len(rows) > MAX_PROGRESS_BATCH,
        )

    def get_run(self, principal: Principal, run_id: UUID) -> AdminRunRecord:
        self.database._admin(principal)
        with self.database.connection() as connection:
            record = self._run(connection, run_id, principal=principal)
        self._owner(principal, record.operation.owner_id)
        return record

    def get_training_metrics(self, principal: Principal, run_id: UUID) -> TrainingMetrics | None:
        self.database._admin(principal)
        with self.database.connection() as connection:
            return read_training_metrics(connection, principal.owner_id, run_id)

    def save_training_metrics(
        self, principal: Principal, run_id: UUID, *, bundle_sha256: str, metrics: TrainingMetrics
    ) -> TrainingMetrics:
        with self.database.connection() as connection:
            run = self._run(connection, run_id, principal=principal)
            if not isinstance(run.spec, (TrainingSpec, ImportedTrainingSpec)):
                raise NotFound()
            if (
                run.output is None
                or run.output.get("bundle_sha256") != bundle_sha256
                or run.spec.model_version_id != metrics.model_version_id
            ):
                raise Conflict("Metrics differ from the committed training result.")
            return store_training_metrics(connection, run, metrics)

    def rename_training_run(
        self, principal: Principal, run_id: UUID, update: DisplayNameUpdate
    ) -> TrainingRunName:
        self.database._admin(principal)
        update = DisplayNameUpdate.model_validate_json(update.model_dump_json())
        with self.database.connection() as connection:
            row = connection.execute(
                "UPDATE regimex_training_runs t SET display_name=%s,name_revision=name_revision+1 "
                "FROM regimex_operations o WHERE t.operation_id=%s "
                "AND o.operation_id=t.operation_id AND o.owner_id=%s AND t.name_revision=%s "
                "RETURNING t.name_revision",
                (update.display_name, run_id, principal.owner_id, update.expected_revision),
            ).fetchone()
            if row is None:
                owned = connection.execute(
                    "SELECT 1 FROM regimex_operations WHERE operation_id=%s "
                    "AND owner_id=%s AND kind='training'",
                    (run_id, principal.owner_id),
                ).fetchone()
                if owned is None:
                    raise NotFound()
                raise Conflict("The training run name changed. Refresh it before saving again.")
            return TrainingRunName(
                run_id=run_id, display_name=update.display_name, revision=cast(int, row[0])
            )

    def set_training_archive(
        self, principal: Principal, run_ids: tuple[UUID, ...], *, archived: bool
    ) -> None:
        self.database._admin(principal)
        if not 1 <= len(run_ids) <= MAX_ARCHIVE_SELECTION or len(set(run_ids)) != len(run_ids):
            raise ValueError(
                f"Select between 1 and {MAX_ARCHIVE_SELECTION} distinct training runs."
            )
        ids = list(run_ids)
        with self.database.connection() as connection:
            # Lock overlapping selections in the same order before changing any member.
            rows = connection.execute(
                "SELECT o.state FROM regimex_training_runs t JOIN regimex_operations o "
                "USING(operation_id) WHERE o.owner_id=%s AND t.operation_id=ANY(%s) "
                "ORDER BY t.operation_id FOR UPDATE OF t",
                (principal.owner_id, ids),
            ).fetchall()
            if len(rows) != len(ids):
                raise NotFound()
            if any(state not in TERMINAL_STATES for (state,) in rows):
                raise Conflict("Only completed training runs can be archived or restored.")
            connection.execute(
                "UPDATE regimex_training_runs SET archived_at="
                "CASE WHEN %s THEN statement_timestamp() ELSE NULL END "
                "WHERE operation_id=ANY(%s) AND (archived_at IS NOT NULL)<>%s",
                (archived, ids, archived),
            )

    def find_replay(
        self,
        principal: Principal,
        kind: AdminKind,
        idempotency_key: str,
        request_sha256: RequestDigest,
    ) -> AdminRunRecord | None:
        self.database._admin(principal)
        request = IntakeRequest(idempotency_key=idempotency_key, request_sha256=request_sha256)
        with self.database.connection() as connection:
            return self._find_replay(connection, principal, kind, request)

    def _find_replay(
        self, connection: Connection, principal: Principal, kind: AdminKind, request: IntakeRequest
    ) -> AdminRunRecord | None:
        query = (
            "SELECT operation_id,request_sha256 FROM regimex_operations "
            "WHERE owner_id=%s AND kind=%s AND idempotency_key=%s"
        )
        args = [principal.owner_id, kind, request.idempotency_key]
        if kind == "evaluation":
            query += (
                " UNION ALL SELECT operation_id,request_sha256 FROM regimex_evaluation_keys "
                "WHERE owner_id=%s AND idempotency_key=%s"
            )
            args.extend([principal.owner_id, request.idempotency_key])
        row = connection.execute(query, args).fetchone()
        if row is None:
            return None
        request.require_match(row[1])
        return self._run(connection, cast(UUID, row[0]), principal=principal)

    @staticmethod
    def _reuse_validation(
        connection: Connection, owner_id: str, spec: EvaluationSpec, request: IntakeRequest
    ) -> AdminRunRecord | None:
        # The owner intake lock serializes choosing a run and binding the new key.
        # Existing installations can have multiple successful runs for these inputs.
        row = connection.execute(
            _RUN_SELECT + " JOIN regimex_evaluation_requests e ON e.operation_id=r.operation_id "
            "WHERE o.owner_id=%s AND e.model_version_id=%s AND e.dataset_id=%s "
            "AND e.reference_set_id=%s "
            """AND ('{"partition":"validation","benchmark_id":null}'::jsonb """
            "|| r.request::jsonb)=%s "
            "AND o.state IN ('succeeded','queued','running','cancel_requested') "
            "ORDER BY (o.state='succeeded') DESC, (o.document->>'created_at')::timestamptz, "
            "o.operation_id LIMIT 1",
            (
                owner_id,
                spec.model_version_id,
                spec.dataset_id,
                spec.reference_set_id,
                Jsonb(spec.model_dump(mode="json")),
            ),
        ).fetchone()
        if row is None:
            return None
        run = _read_run(row)
        connection.execute(
            "INSERT INTO regimex_evaluation_keys "
            "(owner_id,idempotency_key,request_sha256,operation_id) VALUES (%s,%s,%s,%s)",
            (owner_id, request.idempotency_key, request.digest, run.run_id),
        )
        return run

    def list_runs(
        self,
        principal: Principal,
        *,
        kind: AdminKind | None = None,
        training: TrainingSpec | ImportedTrainingSpec | None = None,
        archive: TrainingArchiveFilter = "all",
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[AdminRunRecord, ...]:
        self.database._admin(principal)
        self.database._page(limit, offset)
        with self.database.connection() as connection:
            query = _RUN_SELECT + " WHERE o.owner_id=%s"
            values: list[object] = [principal.owner_id]
            if kind is not None:
                query += " AND r.kind=%s"
                values.append(kind)
            if archive == "active":
                query += " AND t.archived_at IS NULL"
            elif archive == "archived":
                query += " AND t.archived_at IS NOT NULL"
            if training is not None:
                query += (
                    " AND r.request->>'model_version_id'=%s AND (r.kind='publication' OR "
                    "(r.kind='evaluation' AND r.request->>'benchmark_id' IS NULL "
                    "AND COALESCE(r.request->>'partition','validation')='validation'"
                )
                values.append(training.model_version_id)
                if isinstance(training, TrainingSpec):
                    query += (
                        " AND r.request->>'dataset_id'=%s AND r.request->>'reference_set_id'=%s"
                    )
                    values.extend((training.dataset_id, training.validation_reference_set_id))
                query += "))"
            rows = connection.execute(
                query + " ORDER BY (o.document->>'created_at')::timestamptz DESC, "
                "r.operation_id DESC LIMIT %s OFFSET %s",
                [*values, limit, offset],
            ).fetchall()
            return tuple(_read_run(row) for row in rows)

    def _validate_spec(
        self, connection: Connection, owner_id: str, spec: AdminSpec
    ) -> ResolvedModel | None:
        if isinstance(spec, GenerationSpec):
            self.database.lock_catalog_identity(connection, "dataset", spec.dataset_id)
            exists = connection.execute(
                "SELECT 1 FROM regimex_datasets WHERE dataset_id=%s", (spec.dataset_id,)
            ).fetchone()
            if exists is not None:
                raise Conflict("Generation requires a new immutable dataset identity.")
            return None
        if isinstance(spec, ReferenceGenerationSpec):
            self._dataset(connection, owner_id, spec.dataset_id)
            self.database.lock_catalog_identity(connection, "reference", spec.reference_set_id)
            exists = connection.execute(
                "SELECT 1 FROM regimex_reference_sets WHERE reference_set_id=%s",
                (spec.reference_set_id,),
            ).fetchone()
            if exists is not None:
                raise Conflict("Generation requires a new immutable reference-set identity.")
            return None
        if isinstance(spec, TrainingSpec):
            dataset = self._dataset(connection, owner_id, spec.dataset_id)
            labels = self._reference(connection, owner_id, spec.labels_reference_set_id)
            references = self._reference(connection, owner_id, spec.validation_reference_set_id)
            if (
                labels.references.partition != "train"
                or references.references.partition != "validation"
                or labels.dataset_id != dataset.dataset_id
                or references.dataset_id != dataset.dataset_id
            ):
                raise Conflict(
                    "Training requires the dataset's separate train and validation references."
                )
            if (
                labels.references.semantics != references.references.semantics
                or labels.references.analysis_profile != references.references.analysis_profile
            ):
                raise Conflict("Training and validation reference semantics must agree.")
            from regimex.infrastructure.benchmark_repository import BenchmarkRepository

            BenchmarkRepository.validate_selection(
                connection, owner_id, spec, dataset, training_semantics=labels.references.semantics
            )
            self.database.lock_catalog_identity(connection, "model", spec.model_version_id)
            exists = connection.execute(
                "SELECT 1 FROM regimex_models WHERE model_version_id=%s", (spec.model_version_id,)
            ).fetchone()
            if exists is not None:
                raise Conflict("Training requires a new immutable model version.")
            return None
        model = self.database._model(connection, spec.model_version_id, lock=True)
        if isinstance(spec, EvaluationSpec):
            dataset = self._dataset(connection, owner_id, spec.dataset_id)
            references = self._reference(
                connection, owner_id, spec.reference_set_id, allow_test=spec.partition == "test"
            )
            if (
                references.dataset_id != dataset.dataset_id
                or references.references.partition != spec.partition
            ):
                raise Conflict("Evaluation references must match the requested dataset partition.")
            if spec.partition == "test":
                from regimex.infrastructure.benchmark_repository import BenchmarkRepository

                BenchmarkRepository.validate_final_evaluation(
                    connection, owner_id, spec, model, dataset, references
                )
            else:
                origin = connection.execute(
                    "SELECT dataset_id FROM regimex_training_runs "
                    "WHERE committed_model_version_id=%s AND NOT external",
                    (model.model_version_id,),
                ).fetchone()
                if origin is not None and origin[0] != dataset.dataset_id:
                    raise Conflict(
                        "The evaluation dataset differs from the model's training artifact."
                    )
        else:
            evaluation = self._evaluation(connection, owner_id, spec.evaluation_run_id)
            if evaluation.partition != "validation":
                raise Conflict("Publication requires validation evidence, never final-test tuning.")
            if (
                evaluation.model_version_id,
                evaluation.bundle_sha256,
                evaluation.inference_policy_sha256,
            ) != (model.model_version_id, model.bundle_sha256, model.inference_policy_sha256):
                raise Conflict("Publication evaluation must match the immutable model and policy.")
        return ResolvedModel(
            model_version_id=model.model_version_id,
            bundle_sha256=model.bundle_sha256,
            inference_policy_sha256=model.inference_policy_sha256,
        )

    def submit(
        self,
        principal: Principal,
        spec: AdminSpec,
        *,
        idempotency_key: str,
        request_sha256: RequestDigest | None = None,
        max_pending: int = 8,
        display_name: str | None = None,
    ) -> AdminRunRecord:
        self.database._admin(principal)
        display_name = validate_display_name(display_name)
        if display_name is not None and not isinstance(spec, TrainingSpec):
            raise ValueError("Only training runs can be named.")
        spec = _SPEC.model_validate_json(json_bytes(spec.model_dump(mode="json"))).root
        self._bounded(spec.model_dump(mode="json"))
        request = IntakeRequest(
            idempotency_key=idempotency_key,
            request_sha256=request_sha256 or admin_request_hash(spec, display_name),
        )
        if max_pending < 1:
            raise ValueError("pending quota must be positive")
        with self.database.connection() as connection:
            self.database._lock(connection, f"owner:{principal.owner_id}")
            previous = self._find_replay(connection, principal, spec.kind, request)
            if previous is not None:
                return previous
            if isinstance(spec, EvaluationSpec) and spec.partition == "validation":
                previous = self._reuse_validation(connection, principal.owner_id, spec, request)
                if previous is not None:
                    self._validate_spec(connection, principal.owner_id, spec)
                    return previous
            count = connection.execute(
                "SELECT count(*) FROM regimex_operations WHERE owner_id=%s "
                "AND state IN ('queued','running','cancel_requested')",
                (principal.owner_id,),
            ).fetchone()
            assert count is not None
            if cast(int, count[0]) >= max_pending:
                raise CapacityExceeded()
            model = self._validate_spec(connection, principal.owner_id, spec)
            now = datetime.now(UTC)
            run_id = uuid4()
            operation = Operation(
                operation_id=run_id,
                owner_id=principal.owner_id,
                kind=spec.kind,
                resource_id=str(run_id),
                request_sha256=request.digest,
                idempotency_key=idempotency_key,
                model=model,
                created_at=now,
                updated_at=now,
            )
            connection.execute(
                "INSERT INTO regimex_operations (operation_id,owner_id,kind,idempotency_key,"
                "request_sha256,state,document) VALUES (%s,%s,%s,%s,%s,%s,%s)",
                (
                    run_id,
                    principal.owner_id,
                    spec.kind,
                    idempotency_key,
                    request.digest,
                    operation.state,
                    Jsonb(operation.model_dump(mode="json")),
                ),
            )
            connection.execute(
                "INSERT INTO regimex_admin_runs (operation_id,kind,request) VALUES (%s,%s,%s)",
                (run_id, spec.kind, Json(spec.model_dump(mode="json"))),
            )
            self._reserve_request(connection, operation, spec, display_name)
            self.database._enqueue(connection, operation)
            return AdminRunRecord(
                run_id=run_id, operation=operation, spec=spec, display_name=display_name
            )

    @staticmethod
    def _reserve_request(
        connection: Connection, operation: Operation, spec: AdminSpec, display_name: str | None
    ) -> None:
        if isinstance(spec, GenerationSpec):
            row = connection.execute(
                "INSERT INTO regimex_generation_targets (dataset_id,operation_id) VALUES (%s,%s) "
                "ON CONFLICT DO NOTHING RETURNING operation_id",
                (spec.dataset_id, operation.operation_id),
            ).fetchone()
        elif isinstance(spec, ReferenceGenerationSpec):
            row = connection.execute(
                "INSERT INTO regimex_reference_targets (reference_set_id,operation_id) "
                "VALUES (%s,%s) ON CONFLICT DO NOTHING RETURNING operation_id",
                (spec.reference_set_id, operation.operation_id),
            ).fetchone()
        elif isinstance(spec, TrainingSpec):
            row = connection.execute(
                "INSERT INTO regimex_training_runs (operation_id,dataset_id,"
                "labels_reference_set_id,validation_reference_set_id,model_version_id,"
                "display_name) VALUES (%s,%s,%s,%s,%s,%s) "
                "ON CONFLICT DO NOTHING RETURNING operation_id",
                (
                    operation.operation_id,
                    spec.dataset_id,
                    spec.labels_reference_set_id,
                    spec.validation_reference_set_id,
                    spec.model_version_id,
                    display_name,
                ),
            ).fetchone()
        elif isinstance(spec, EvaluationSpec):
            row = connection.execute(
                "INSERT INTO regimex_evaluation_requests "
                "(operation_id,model_version_id,dataset_id,reference_set_id) "
                "VALUES (%s,%s,%s,%s) RETURNING operation_id",
                (
                    operation.operation_id,
                    spec.model_version_id,
                    spec.dataset_id,
                    spec.reference_set_id,
                ),
            ).fetchone()
        else:
            row = connection.execute(
                "INSERT INTO regimex_publication_requests "
                "(operation_id,model_version_id,evaluation_run_id) "
                "VALUES (%s,%s,%s) RETURNING operation_id",
                (operation.operation_id, spec.model_version_id, spec.evaluation_run_id),
            ).fetchone()
        if row is None:
            raise Conflict(
                "A different operation already reserved the requested immutable resource."
            )

    @contextmanager
    def _active_run(
        self, run_id: UUID, token: UUID, kind: AdminKind
    ) -> Iterator[tuple[Connection, AdminRunRecord]]:
        with self.database.completion_transaction(run_id, token, kind) as (connection, _):
            yield connection, self._run(connection, run_id)

    def _finish(
        self, connection: Connection, run: AdminRunRecord, token: UUID, output: dict[str, JsonValue]
    ) -> AdminRunRecord:
        output = self._bounded(output)
        completed = succeed(
            run.operation,
            token,
            hashlib.sha256(json_bytes(output)).hexdigest(),
            self.database._now(run.operation),
        )
        connection.execute(
            "UPDATE regimex_admin_runs SET output=%s WHERE operation_id=%s",
            (Json(output), run.run_id),
        )
        self.database._save(connection, completed)
        return run.model_copy(update={"operation": completed, "output": output})

    def complete_generation(
        self,
        run_id: UUID,
        token: UUID,
        *,
        dataset: DatasetRecord,
        snapshots: tuple[SnapshotRecord, ...],
        bindings: tuple[CatalogBinding, ...],
    ) -> AdminRunRecord:
        with self._active_run(run_id, token, "generation") as (connection, run):
            assert isinstance(run.spec, GenerationSpec)
            if (
                dataset.owner_id != run.operation.owner_id
                or dataset.dataset_id != run.spec.dataset_id
                or dataset.generation != run.spec
                or tuple(item.snapshot_id for item in snapshots) != dataset.snapshot_ids
                or len(snapshots) != run.spec.plan.experiment_count
                or any(
                    member.step_count != run.spec.plan.step_count
                    for member in dataset.manifest.members
                )
            ):
                raise Conflict("Generation output differs from its immutable request and owner.")
            for snapshot in snapshots:
                if snapshot.owner_id != run.operation.owner_id:
                    raise Conflict("Generated snapshots must retain the operation owner.")
                stored = self.database._add_snapshot(connection, snapshot)
                if stored.snapshot_id != snapshot.snapshot_id:
                    raise Conflict("Generation cannot replace another registered source snapshot.")
            dataset = self._add_dataset(
                connection, dataset, bindings=bindings, generation_run_id=run_id
            )
            return self._finish(
                connection,
                run,
                token,
                {
                    "dataset_id": dataset.dataset_id,
                    "dataset_sha256": dataset.manifest.content_sha256,
                },
            )

    def complete_reference(
        self, run_id: UUID, token: UUID, record: ReferenceSetRecord
    ) -> AdminRunRecord:
        with self._active_run(run_id, token, "reference") as (connection, run):
            assert isinstance(run.spec, ReferenceGenerationSpec)
            if (
                record.owner_id != run.operation.owner_id
                or record.dataset_id != run.spec.dataset_id
                or record.references.reference_set_id != run.spec.reference_set_id
                or record.references.partition != run.spec.partition
                or record.references.analysis_profile != run.spec.analysis
                or record.references.kind != "automatic"
            ):
                raise Conflict("Generated references differ from the immutable annotation request.")
            self._add_reference_set(connection, record, reference_run_id=run_id)
            return self._finish(
                connection,
                run,
                token,
                {
                    "reference_set_id": record.references.reference_set_id,
                    "reference_sha256": record.references.content_sha256,
                },
            )

    def complete_training(
        self,
        run_id: UUID,
        token: UUID,
        model: ModelRecord,
        *,
        metrics: TrainingMetrics,
        trainable_parameters: int | None = None,
        fit_statistics: TrainingFitStatistics | None = None,
    ) -> AdminRunRecord:
        with self._active_run(run_id, token, "training") as (connection, run):
            assert isinstance(run.spec, TrainingSpec)
            if metrics.model_version_id != run.spec.model_version_id:
                raise Conflict("Metrics refer to a different model version.")
            if (model.model_version_id, model.architecture_id, model.inference_runtime) != (
                run.spec.model_version_id,
                run.spec.architecture_id,
                run.spec.inference_runtime,
            ) or model.inference_policy_sha256 != hashlib.sha256(
                json_bytes(model.inference_runtime.model_dump(mode="json"))
            ).hexdigest():
                raise Conflict(
                    "Candidate identity, architecture or inference policy differs from training."
                )
            model = self.database._register_model(connection, model)
            connection.execute(
                "UPDATE regimex_training_runs SET committed_model_version_id=%s "
                "WHERE operation_id=%s",
                (model.model_version_id, run_id),
            )
            store_training_metrics(connection, run, metrics)
            return self._finish(
                connection,
                run,
                token,
                {
                    "model_version_id": model.model_version_id,
                    "bundle_sha256": model.bundle_sha256,
                    "quality_status": model.quality_status,
                    "validation_segment_macro_f1": metrics.validation_segment_macro_f1,
                    **(
                        {"fit_statistics": fit_statistics.model_dump(mode="json")}
                        if fit_statistics is not None
                        else {}
                    ),
                    **(
                        {"trainable_parameters": trainable_parameters}
                        if trainable_parameters is not None
                        else {}
                    ),
                },
            )

    @staticmethod
    def _evaluation(connection: Connection, owner_id: str, run_id: UUID) -> EvaluationRecord:
        row = connection.execute(
            "SELECT document FROM regimex_evaluation_records WHERE run_id=%s AND owner_id=%s",
            (run_id, owner_id),
        ).fetchone()
        if row is None:
            raise NotFound()
        return EvaluationRecord.model_validate_json(json_bytes(row[0]))

    def get_evaluation(self, principal: Principal, run_id: UUID) -> EvaluationRecord:
        self.database._admin(principal)
        with self.database.connection() as connection:
            return self._evaluation(connection, principal.owner_id, run_id)

    def get_publication_evaluation(self, model_version_id: str) -> EvaluationRecord | None:
        """Return the evaluation of the latest audited publication, never a best score."""
        with self.database.connection() as connection:
            row = connection.execute(
                "SELECT e.document FROM (SELECT evaluation_run_id,operation_id "
                "FROM regimex_publication_audit WHERE model_version_id=%s "
                "ORDER BY published_at DESC,publication_id DESC LIMIT 1) p "
                "JOIN regimex_evaluation_records e ON e.run_id=p.evaluation_run_id "
                "JOIN regimex_operations o ON o.operation_id=p.operation_id "
                "WHERE o.state='succeeded'",
                (model_version_id,),
            ).fetchone()
            return None if row is None else EvaluationRecord.model_validate_json(json_bytes(row[0]))

    def complete_evaluation(
        self, run_id: UUID, token: UUID, evaluation: EvaluationRecord
    ) -> AdminRunRecord:
        evaluation = EvaluationRecord.model_validate_json(evaluation.model_dump_json())
        self._bounded(evaluation.metrics, limit=MAX_REPORT_BYTES, description="Evaluation metrics")
        with self._active_run(run_id, token, "evaluation") as (connection, run):
            assert isinstance(run.spec, EvaluationSpec)
            model = self.database._model(connection, run.spec.model_version_id)
            dataset = self._dataset(connection, run.operation.owner_id, run.spec.dataset_id)
            references = self._reference(
                connection,
                run.operation.owner_id,
                run.spec.reference_set_id,
                allow_test=run.spec.partition == "test",
            )
            self._validate_spec(connection, run.operation.owner_id, run.spec)
            if (
                evaluation.run_id,
                evaluation.owner_id,
                evaluation.model_version_id,
                evaluation.bundle_sha256,
                evaluation.inference_policy_sha256,
                evaluation.dataset_id,
                evaluation.dataset_sha256,
                evaluation.reference_set_id,
                evaluation.reference_sha256,
                evaluation.partition,
            ) != (
                run_id,
                run.operation.owner_id,
                model.model_version_id,
                model.bundle_sha256,
                model.inference_policy_sha256,
                dataset.dataset_id,
                dataset.manifest.content_sha256,
                references.references.reference_set_id,
                references.references.content_sha256,
                run.spec.partition,
            ):
                raise Conflict(
                    "Evaluation output differs from its fixed model, dataset or reference identity."
                )
            self.database._object(connection, evaluation.report)
            connection.execute(
                "INSERT INTO regimex_evaluation_records (run_id,owner_id,model_version_id,"
                "dataset_id,reference_set_id,report_key,document) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s)",
                (
                    run_id,
                    evaluation.owner_id,
                    evaluation.model_version_id,
                    evaluation.dataset_id,
                    evaluation.reference_set_id,
                    evaluation.report.key,
                    Json(evaluation.model_dump(mode="json")),
                ),
            )
            return self._finish(
                connection,
                run,
                token,
                {
                    "evaluation_run_id": str(run_id),
                    "report_sha256": evaluation.report.sha256,
                    "quality_status": evaluation.quality_status,
                },
            )

    def complete_publication(self, run_id: UUID, token: UUID) -> AdminRunRecord:
        with self._active_run(run_id, token, "publication") as (connection, run):
            assert isinstance(run.spec, PublicationSpec)
            self._validate_spec(connection, run.operation.owner_id, run.spec)
            model = self.database._activate_model(
                connection,
                run.operation.owner_id,
                run.spec.model_version_id,
                expected_previous=run.spec.expected_previous,
                evaluation_run_id=run.spec.evaluation_run_id,
                operation_id=run_id,
            )
            return self._finish(
                connection,
                run,
                token,
                {
                    "model_version_id": model.model_version_id,
                    "bundle_sha256": model.bundle_sha256,
                    "quality_status": model.quality_status,
                },
            )

    def get_training_run_for_model(
        self, principal: Principal, model_version_id: str
    ) -> AdminRunRecord | None:
        self.database._admin(principal)
        with self.database.connection() as connection:
            row = connection.execute(
                "SELECT r.operation_id FROM regimex_training_runs r JOIN regimex_operations o "
                "ON r.operation_id=o.operation_id WHERE r.model_version_id=%s AND o.owner_id=%s",
                (model_version_id, principal.owner_id),
            ).fetchone()
            return None if row is None else self._run(connection, cast(UUID, row[0]))

    def register_external_training(
        self,
        principal: Principal,
        model: ModelRecord,
        spec: ImportedTrainingSpec,
        metrics: TrainingMetrics,
        *,
        display_name: str | None,
        evaluations: tuple[ExternalEvaluation, ...],
        latency: ExternalLatencyEvidence | None,
    ) -> ModelRecord:
        self.database._admin(principal)
        display_name = validate_display_name(display_name)
        if (
            spec.model_version_id != model.model_version_id
            or spec.architecture_id != model.architecture_id
            or spec.inference_runtime != model.inference_runtime
            or spec.model_version_id != metrics.model_version_id
            or any(item.bundle_sha256 != model.bundle_sha256 for item in evaluations)
            or (latency is not None and latency.manifest_sha256 != model.bundle_sha256)
        ):
            raise Conflict("Imported training evidence belongs to another model or policy.")
        with self.database.connection() as connection:
            self.database.lock_catalog_identity(connection, "model", model.model_version_id)
            row = connection.execute(
                "SELECT operation_id,committed_model_version_id FROM regimex_training_runs "
                "WHERE model_version_id=%s",
                (model.model_version_id,),
            ).fetchone()
            if row is not None and row[1] != model.model_version_id:
                raise Conflict("This model version is reserved by an administrative training run.")
            # The same registration may finish on another host during object upload.
            # Keep its verified object generations and immutable inference policy.
            try:
                registered = self.database._model(connection, model.model_version_id, lock=True)
            except NotFound:
                self.database._require_unreserved_version(connection, model.model_version_id)
                registered = self.database._register_model(connection, model)
            if (registered.bundle_sha256, registered.inference_policy_sha256) != (
                model.bundle_sha256,
                model.inference_policy_sha256,
            ):
                raise Conflict("The model version already has another immutable bundle or policy.")
            if row is None:
                run = self._insert_external_training(
                    connection, principal.owner_id, registered, spec, metrics, display_name
                )
            else:
                run = self._run(connection, cast(UUID, row[0]), principal=principal)
                if isinstance(run.spec, TrainingSpec):
                    if spec.evidence.run_id != str(run.run_id):
                        raise Conflict("The bundle does not identify its training run.")
                elif run.spec != spec:
                    raise Conflict("The training run already records different bundle evidence.")
            self._save_external_reports(connection, run, evaluations, latency)
            return registered

    def _insert_external_training(
        self,
        connection: Connection,
        owner_id: str,
        model: ModelRecord,
        spec: ImportedTrainingSpec,
        metrics: TrainingMetrics,
        display_name: str | None,
    ) -> AdminRunRecord:
        now = datetime.now(UTC)
        run_id = uuid4()
        request = spec.model_dump(mode="json")
        output: dict[str, JsonValue] = {
            "model_version_id": model.model_version_id,
            "bundle_sha256": model.bundle_sha256,
            "quality_status": model.quality_status,
            "validation_segment_macro_f1": metrics.validation_segment_macro_f1,
        }
        operation = Operation(
            operation_id=run_id,
            kind="training",
            owner_id=owner_id,
            resource_id=str(run_id),
            idempotency_key=str(run_id),
            request_sha256=hashlib.sha256(json_bytes(request)).hexdigest(),
            result_sha256=hashlib.sha256(json_bytes(output)).hexdigest(),
            state="succeeded",
            stage="imported",
            created_at=now,
            updated_at=now,
        )
        connection.execute(
            "INSERT INTO regimex_operations (operation_id,owner_id,kind,idempotency_key,"
            "request_sha256,state,document) VALUES (%s,%s,'training',%s,%s,'succeeded',%s)",
            (
                run_id,
                owner_id,
                operation.idempotency_key,
                operation.request_sha256,
                Jsonb(operation.model_dump(mode="json")),
            ),
        )
        connection.execute(
            "INSERT INTO regimex_admin_runs (operation_id,kind,request,output) "
            "VALUES (%s,'training',%s,%s)",
            (run_id, Json(request), Json(output)),
        )
        connection.execute(
            "INSERT INTO regimex_training_runs (operation_id,external,model_version_id,"
            "committed_model_version_id,display_name) VALUES (%s,true,%s,%s,%s)",
            (run_id, model.model_version_id, model.model_version_id, display_name),
        )
        run = AdminRunRecord(
            run_id=run_id, operation=operation, spec=spec, output=output, display_name=display_name
        )
        store_training_metrics(connection, run, metrics)
        return run

    @staticmethod
    def _save_external_reports(
        connection: Connection,
        run: AdminRunRecord,
        evaluations: tuple[ExternalEvaluation, ...],
        latency: ExternalLatencyEvidence | None,
    ) -> None:
        merged = {(item.kind, item.freeze_sha256): item for item in run.external_evaluations}
        for item in evaluations:
            if merged.setdefault((item.kind, item.freeze_sha256), item) != item:
                raise Conflict(
                    "A different external report is already recorded for this selection."
                )
        if latency is not None and run.external_latency not in (None, latency):
            raise Conflict("A different external latency report is already recorded.")
        if len(merged) > 64:
            raise CapacityExceeded("A training run supports up to 64 external evaluation reports.")
        if len(merged) != len(run.external_evaluations) or (
            latency is not None and run.external_latency is None
        ):
            connection.execute(
                "UPDATE regimex_training_runs SET external_evaluations=%s,"
                "external_latency=COALESCE(external_latency,%s) WHERE operation_id=%s",
                (
                    Json([item.model_dump(mode="json") for item in merged.values()]),
                    Json(latency.model_dump(mode="json")) if latency is not None else None,
                    run.run_id,
                ),
            )
