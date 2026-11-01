"""Transactional administrative resources and fenced lifecycle publication."""

import hashlib
from datetime import UTC, datetime
from typing import cast
from uuid import UUID, uuid4

import psycopg
from psycopg.types.json import Json, Jsonb
from pydantic import JsonValue, TypeAdapter

from regimex.admin_contracts import (
    AdminKind,
    AdminRunRecord,
    AdminSpec,
    CatalogBinding,
    DatasetRecord,
    EvaluationRecord,
    EvaluationSpec,
    GenerationSpec,
    PublicationSpec,
    ReferenceGenerationSpec,
    ReferenceSetRecord,
    TrainingSpec,
)
from regimex.contracts import MAX_INPUT_BYTES
from regimex.data_catalog import validate_reference_membership
from regimex.directions import DIRECTION_POLICY
from regimex.infrastructure.database import DatabaseRepository
from regimex.infrastructure.intake_repository import IntakeReceipts, IntakeRequest
from regimex.operations import Operation, ResolvedModel, succeed
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

_SPEC: TypeAdapter[AdminSpec] = TypeAdapter(AdminSpec)
_JSON = TypeAdapter(dict[str, JsonValue])
Connection = psycopg.Connection[tuple[object, ...]]


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
    def _bounded(value: dict[str, JsonValue], *, limit: int = 64 * 1024) -> dict[str, JsonValue]:
        checked = _JSON.validate_python(value, strict=True)
        if len(json_bytes(checked)) > limit:
            raise CapacityExceeded("Administrative metadata exceeds its configured size bound.")
        return checked

    def _binding(self, connection: Connection, owner_id: str, snapshot_id: UUID) -> CatalogBinding:
        snapshot = self.database._snapshot(connection, snapshot_id)
        if snapshot.owner_id != owner_id:
            raise NotFound()
        row = connection.execute(
            "SELECT document FROM regimex_catalog_bindings WHERE snapshot_id=%s", (snapshot_id,)
        ).fetchone()
        if row is None:
            raise NotFound("The snapshot has no admitted catalog binding.")
        return CatalogBinding.model_validate_json(json_bytes(row[0]))

    def get_binding(self, principal: Principal, snapshot_id: UUID) -> CatalogBinding | None:
        self.database._admin(principal)
        with self.database.connection() as connection:
            snapshot = self.database._snapshot(connection, snapshot_id)
            if snapshot.owner_id != principal.owner_id:
                raise NotFound()
            exists = connection.execute(
                "SELECT 1 FROM regimex_catalog_bindings WHERE snapshot_id=%s", (snapshot_id,)
            ).fetchone()
            if exists is None:
                return None
            return self._binding(connection, principal.owner_id, snapshot_id)

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
        self, principal: Principal, *, limit: int = 100, offset: int = 0
    ) -> tuple[DatasetRecord, ...]:
        self.database._admin(principal)
        self.database._page(limit, offset)
        with self.database.connection() as connection:
            rows = connection.execute(
                "SELECT document FROM regimex_datasets WHERE owner_id=%s "
                "ORDER BY dataset_id LIMIT %s OFFSET %s",
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
        reservation = connection.execute(
            "SELECT operation_id FROM regimex_generation_targets WHERE dataset_id=%s",
            (record.dataset_id,),
        ).fetchone()
        if reservation is not None and reservation[0] != generation_run_id:
            raise Conflict("The dataset identity is reserved by a generation run.")
        for binding in bindings:
            self._add_binding(connection, record.owner_id, binding)
        for snapshot_id, member in zip(record.snapshot_ids, record.manifest.members, strict=True):
            snapshot = self.database._snapshot(connection, snapshot_id)
            binding = self._binding(connection, record.owner_id, snapshot_id)
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
        self, principal: Principal, *, idempotency_key: str, request_sha256: str
    ) -> DatasetRecord | None:
        self.database._admin(principal)
        request = IntakeRequest(idempotency_key=idempotency_key, request_sha256=request_sha256)
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
        # The final-reference gate precedes parsing, including historical policies.
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
        reservation = connection.execute(
            "SELECT operation_id FROM regimex_reference_targets WHERE reference_set_id=%s",
            (record.references.reference_set_id,),
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
        request_sha256: str | None = None,
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
        self,
        principal: Principal,
        *,
        idempotency_key: str,
        request_sha256: str,
        final: bool = False,
    ) -> ReferenceSetRecord | None:
        self.database._admin(principal)
        request = IntakeRequest(idempotency_key=idempotency_key, request_sha256=request_sha256)
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
        request_sha256: str | None = None,
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
            # Uploads are not publication. Recheck under the registry lock so
            # concurrent history admission or reservation cannot race this commit.
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
        row = connection.execute(
            "SELECT request,output FROM regimex_admin_runs WHERE operation_id=%s", (run_id,)
        ).fetchone()
        if row is None:
            raise NotFound()
        operation = self.database._operation(connection, run_id, principal=principal)
        result = AdminRunRecord(
            run_id=run_id,
            operation=operation,
            spec=_SPEC.validate_json(json_bytes(row[0])),
            output=None if row[1] is None else _JSON.validate_python(row[1], strict=True),
        )
        if (
            result.output is not None
            and hashlib.sha256(json_bytes(result.output)).hexdigest() != operation.result_sha256
        ):
            raise Conflict("Administrative output differs from its committed digest.")
        return result

    def get_run_internal(self, run_id: UUID) -> AdminRunRecord:
        with self.database.connection() as connection:
            return self._run(connection, run_id)

    def get_run(self, principal: Principal, run_id: UUID) -> AdminRunRecord:
        self.database._admin(principal)
        with self.database.connection() as connection:
            record = self._run(connection, run_id, principal=principal)
        self._owner(principal, record.operation.owner_id)
        return record

    def find_replay(
        self, principal: Principal, kind: AdminKind, idempotency_key: str, request_sha256: str
    ) -> AdminRunRecord | None:
        self.database._admin(principal)
        with self.database.connection() as connection:
            row = connection.execute(
                "SELECT operation_id,request_sha256 FROM regimex_operations "
                "WHERE owner_id=%s AND kind=%s AND idempotency_key=%s",
                (principal.owner_id, kind, idempotency_key),
            ).fetchone()
            if row is None:
                return None
            if row[1] != request_sha256:
                raise Conflict("The idempotency key was already used for a different request.")
            return self._run(connection, cast(UUID, row[0]))

    def list_runs(
        self,
        principal: Principal,
        *,
        kind: AdminKind | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[AdminRunRecord, ...]:
        self.database._admin(principal)
        self.database._page(limit, offset)
        with self.database.connection() as connection:
            query = (
                "SELECT r.operation_id FROM regimex_admin_runs r JOIN regimex_operations o "
                "ON r.operation_id=o.operation_id WHERE o.owner_id=%s"
            )
            values: list[object] = [principal.owner_id]
            if kind is not None:
                query += " AND r.kind=%s"
                values.append(kind)
            rows = connection.execute(
                query + " ORDER BY r.operation_id LIMIT %s OFFSET %s", [*values, limit, offset]
            ).fetchall()
            return tuple(self._run(connection, cast(UUID, row[0])) for row in rows)

    def _validate_spec(
        self, connection: Connection, owner_id: str, spec: AdminSpec
    ) -> ResolvedModel | None:
        if isinstance(spec, GenerationSpec):
            exists = connection.execute(
                "SELECT 1 FROM regimex_datasets WHERE dataset_id=%s", (spec.dataset_id,)
            ).fetchone()
            if exists is not None:
                raise Conflict("Generation requires a new immutable dataset identity.")
            return None
        if isinstance(spec, ReferenceGenerationSpec):
            self._dataset(connection, owner_id, spec.dataset_id)
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

            BenchmarkRepository.validate_selection(connection, owner_id, spec, dataset)
            exists = connection.execute(
                "SELECT 1 FROM regimex_models WHERE model_version_id=%s", (spec.model_version_id,)
            ).fetchone()
            if exists is not None:
                raise Conflict("Training requires a new immutable model version.")
            return None
        model = self.database._model(connection, spec.model_version_id)
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
                    "WHERE committed_model_version_id=%s",
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
        request_sha256: str | None = None,
        max_pending: int = 8,
    ) -> AdminRunRecord:
        self.database._admin(principal)
        spec = _SPEC.validate_json(json_bytes(spec.model_dump(mode="json")))
        self._bounded(spec.model_dump(mode="json"))
        digest = (
            request_sha256 or hashlib.sha256(json_bytes(spec.model_dump(mode="json"))).hexdigest()
        )
        if max_pending < 1:
            raise ValueError("pending quota must be positive")
        with self.database.connection() as connection:
            self.database._lock(connection, f"owner:{principal.owner_id}")
            previous = connection.execute(
                "SELECT operation_id,request_sha256 FROM regimex_operations "
                "WHERE owner_id=%s AND kind=%s AND idempotency_key=%s",
                (principal.owner_id, spec.kind, idempotency_key),
            ).fetchone()
            if previous is not None:
                if previous[1] != digest:
                    raise Conflict("The idempotency key was already used for a different request.")
                return self._run(connection, cast(UUID, previous[0]))
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
                request_sha256=digest,
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
                    digest,
                    operation.state,
                    Jsonb(operation.model_dump(mode="json")),
                ),
            )
            connection.execute(
                "INSERT INTO regimex_admin_runs (operation_id,kind,request) VALUES (%s,%s,%s)",
                (run_id, spec.kind, Json(spec.model_dump(mode="json"))),
            )
            self._reserve_request(connection, operation, spec)
            self.database._enqueue(connection, operation)
            return AdminRunRecord(run_id=run_id, operation=operation, spec=spec)

    @staticmethod
    def _reserve_request(connection: Connection, operation: Operation, spec: AdminSpec) -> None:
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
                "labels_reference_set_id,validation_reference_set_id,model_version_id) "
                "VALUES (%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING RETURNING operation_id",
                (
                    operation.operation_id,
                    spec.dataset_id,
                    spec.labels_reference_set_id,
                    spec.validation_reference_set_id,
                    spec.model_version_id,
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

    def _active(
        self, connection: Connection, run_id: UUID, token: UUID, kind: AdminKind
    ) -> AdminRunRecord:
        operation = self.database._operation(connection, run_id, lock=True)
        if (
            operation.kind != kind
            or operation.state != "running"
            or operation.attempt_token != token
        ):
            raise StaleAttempt()
        return self._run(connection, run_id)

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
        return AdminRunRecord(run_id=run.run_id, operation=completed, spec=run.spec, output=output)

    def complete_generation(
        self,
        run_id: UUID,
        token: UUID,
        *,
        dataset: DatasetRecord,
        snapshots: tuple[SnapshotRecord, ...],
        bindings: tuple[CatalogBinding, ...],
    ) -> AdminRunRecord:
        with self.database.connection() as connection:
            run = self._active(connection, run_id, token, "generation")
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
        with self.database.connection() as connection:
            run = self._active(connection, run_id, token, "reference")
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

    def complete_training(self, run_id: UUID, token: UUID, model: ModelRecord) -> AdminRunRecord:
        with self.database.connection() as connection:
            run = self._active(connection, run_id, token, "training")
            assert isinstance(run.spec, TrainingSpec)
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
        self._bounded(evaluation.metrics)
        with self.database.connection() as connection:
            run = self._active(connection, run_id, token, "evaluation")
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
        with self.database.connection() as connection:
            run = self._active(connection, run_id, token, "publication")
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
