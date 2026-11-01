"""Attested benchmark lifecycle without reading final annotation payloads.

The service gate cannot establish that an operator has not inspected raw data or
files outside it. Inventory and review independence remain scoped attestations.
"""

import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

from psycopg.types.json import Jsonb

from regimex.admin_contracts import DatasetRecord, EvaluationSpec, ReferenceSetRecord, TrainingSpec
from regimex.application.service import materialize_object
from regimex.benchmark_contracts import (
    BenchmarkBinding,
    BenchmarkView,
    FinalEvaluationGrant,
    FrozenChoice,
    RegistryState,
    benchmark_member,
    dataset_identity_keys,
)
from regimex.benchmarks import (
    BenchmarkRegistry,
    DatasetHistory,
    DirectionSemantics,
    PartitionMembership,
    ReferenceProvenance,
    attest_history,
    freeze_choices,
    open_benchmark,
    record_selection_started,
    register_history,
    reserve_benchmark,
)
from regimex.data_catalog import content_hash, validate_reference_membership
from regimex.infrastructure.admin_repository import AdminRepository
from regimex.infrastructure.benchmark_repository import BenchmarkRepository, Connection, Transition
from regimex.infrastructure.database import DatabaseRepository
from regimex.infrastructure.object_store import ObjectIntegrityError, ObjectStore
from regimex.model_bundles import MAX_BUNDLE_MANIFEST_BYTES, ModelBundle
from regimex.service_contracts import Conflict, Forbidden, NotFound, Principal, ServiceError
from regimex.storage import json_bytes


def _keys(dataset: DatasetRecord) -> set[tuple[str, ...]]:
    return dataset_identity_keys(dataset.manifest.members)


def _reference(record: ReferenceSetRecord) -> ReferenceProvenance:
    metadata = record.references
    return ReferenceProvenance(
        reference_set_sha256=metadata.content_sha256,
        kind=metadata.kind,
        policy=metadata.policy,
        semantics=metadata.semantics,
        reviewer=metadata.reviewer,
        blind_review=metadata.blind_review,
        independent=metadata.independent,
    )


def _event_time(registry: BenchmarkRegistry) -> datetime:
    known = [item.recorded_at for item in registry.history]
    if registry.history_attestation is not None:
        known.append(registry.history_attestation.at)
    for item in registry.benchmarks:
        known.append(item.reserved_at)
        if item.selection_started_at is not None:
            known.append(item.selection_started_at)
        if item.freeze is not None:
            known.append(item.freeze.at)
        if item.opened_at is not None:
            known.append(item.opened_at)
    return max(
        datetime.now(UTC),
        max(known, default=datetime.min.replace(tzinfo=UTC)) + timedelta(microseconds=1),
    )


class BenchmarkService:
    def __init__(
        self,
        repository: DatabaseRepository,
        admin_repository: AdminRepository,
        store: ObjectStore,
        work_root: Path,
    ) -> None:
        self.database = repository
        self.admin_repository = admin_repository
        self.repository = BenchmarkRepository(repository)
        self.store = store
        self.work_root = work_root
        self.work_root.mkdir(parents=True, exist_ok=True, mode=0o700)

    def _change(self, principal: Principal, transition: Transition) -> RegistryState:
        try:
            return self.repository.change(principal, transition)
        except ServiceError:
            raise
        except ValueError as exc:
            raise Conflict(str(exc)) from exc

    def get_registry(self, principal: Principal) -> RegistryState:
        return self.repository.get_registry(principal)

    def get_benchmark(self, principal: Principal, benchmark_id: str) -> BenchmarkView:
        return self.repository.get_benchmark(principal, benchmark_id)

    @staticmethod
    def _reserved_dataset_keys(connection: Connection) -> set[tuple[str, ...]]:
        rows = connection.execute(
            "SELECT d.document FROM regimex_datasets d JOIN regimex_benchmark_bindings b "
            "ON b.dataset_id=d.dataset_id"
        )
        return {
            key
            for row in rows
            for key in _keys(DatasetRecord.model_validate_json(json_bytes(row[0])))
        }

    def register_dataset_history(self, principal: Principal, dataset_id: str) -> RegistryState:
        def transition(connection: Connection, state: RegistryState) -> BenchmarkRegistry:
            dataset = self.admin_repository._dataset(connection, principal.owner_id, dataset_id)
            existing = connection.execute(
                "SELECT dataset_sha256 FROM regimex_benchmark_history_datasets "
                "WHERE dataset_id=%s AND owner_id=%s",
                (dataset_id, principal.owner_id),
            ).fetchone()
            if existing == (dataset.manifest.content_sha256,):
                return state.registry
            if _keys(dataset) & self._reserved_dataset_keys(connection):
                raise Conflict("Development history overlaps a reserved benchmark dataset.")
            history = DatasetHistory(
                dataset_id=dataset_id,
                dataset_sha256=dataset.manifest.content_sha256,
                recorded_at=_event_time(state.registry),
                memberships=tuple(
                    PartitionMembership(partition=member.partition, member=benchmark_member(member))
                    for member in dataset.manifest.members
                ),
            )
            registry = register_history(state.registry, history)
            # Append first so the immutable source-run binding can reference its
            # exact journal revision. The enclosing transaction commits both.
            updated = self.repository.append(connection, state, registry)
            connection.execute(
                "INSERT INTO regimex_benchmark_history_datasets "
                "(dataset_id,owner_id,dataset_sha256,registered_revision) VALUES (%s,%s,%s,%s)",
                (dataset_id, principal.owner_id, dataset.manifest.content_sha256, updated.revision),
            )
            return registry

        # This transition appends alongside its binding. Avoid a second append by
        # loading the now-current state after the callback's atomic insert.
        self.database._admin(principal)
        with self.database.connection() as connection:
            self.repository.lock(connection)
            state = self.repository.load(connection)
            try:
                transition(connection, state)
            except ServiceError:
                raise
            except ValueError as exc:
                raise Conflict(str(exc)) from exc
            return self.repository.load(connection)

    def import_history(self, principal: Principal, history: DatasetHistory) -> RegistryState:
        history = DatasetHistory.model_validate_json(history.model_dump_json())
        if history.recorded_at > datetime.now(UTC):
            raise Conflict("Imported exposure history cannot be recorded in the future.")

        def transition(connection: Connection, state: RegistryState) -> BenchmarkRegistry:
            for existing in state.registry.history:
                if existing.dataset_sha256 == history.dataset_sha256:
                    if existing == history:
                        return state.registry
                    raise Conflict(
                        "An imported history identity already has other immutable metadata."
                    )
            return register_history(state.registry, history)

        return self._change(principal, transition)

    def attest(
        self, principal: Principal, source_namespaces: tuple[str, ...], statement: str
    ) -> RegistryState:
        return self._change(
            principal,
            lambda connection, state: attest_history(
                state.registry,
                source_namespaces=source_namespaces,
                attested_by=principal.owner_id,
                statement=statement,
                at=_event_time(state.registry),
            ),
        )

    def reserve(
        self,
        principal: Principal,
        benchmark_id: str,
        dataset_id: str,
        reference_set_id: str,
    ) -> BenchmarkView:
        def transition(connection: Connection, state: RegistryState) -> BenchmarkRegistry:
            dataset = self.admin_repository._dataset(connection, principal.owner_id, dataset_id)
            references = self.admin_repository._reference(
                connection, principal.owner_id, reference_set_id, allow_test=True
            )
            if references.dataset_id != dataset_id or references.references.partition != "test":
                raise Forbidden(
                    "A benchmark requires the chosen dataset's complete test reference set."
                )
            validate_reference_membership(dataset.manifest, references.references)
            if _keys(dataset) & self._reserved_dataset_keys(connection):
                raise Conflict("A fresh benchmark dataset overlaps an earlier reserved dataset.")
            # Check source-run overlap in addition to content fingerprints so
            # partial exports from the same run cannot cross partitions.
            for row in connection.execute(
                "SELECT d.document FROM regimex_datasets d "
                "JOIN regimex_benchmark_history_datasets h ON h.dataset_id=d.dataset_id"
            ):
                if _keys(dataset) & _keys(DatasetRecord.model_validate_json(json_bytes(row[0]))):
                    raise Conflict("The benchmark dataset overlaps a prior raw dataset partition.")
            at = _event_time(state.registry)
            reference = _reference(references)
            # Check all of B against all historical partitions, even though only
            # its test members become the final benchmark reservation.
            reserve_benchmark(
                state.registry,
                benchmark_id=benchmark_id,
                members=tuple(benchmark_member(member) for member in dataset.manifest.members),
                reference=reference,
                at=at,
            )
            registry = reserve_benchmark(
                state.registry,
                benchmark_id=benchmark_id,
                members=tuple(
                    benchmark_member(member)
                    for member in dataset.manifest.members
                    if member.partition == "test"
                ),
                reference=reference,
                at=at,
            )
            binding = BenchmarkBinding(
                owner_id=principal.owner_id,
                benchmark_id=benchmark_id,
                dataset_id=dataset_id,
                dataset_sha256=dataset.manifest.content_sha256,
                reference_set_id=reference_set_id,
                reference_sha256=references.references.content_sha256,
            )
            connection.execute(
                "INSERT INTO regimex_benchmark_bindings "
                "(benchmark_id,owner_id,dataset_id,reference_set_id,document) "
                "VALUES (%s,%s,%s,%s,%s)",
                (
                    benchmark_id,
                    principal.owner_id,
                    dataset_id,
                    reference_set_id,
                    Jsonb(binding.model_dump()),
                ),
            )
            return registry

        self._change(principal, transition)
        return self.get_benchmark(principal, benchmark_id)

    def start_selection(self, principal: Principal, benchmark_id: str) -> BenchmarkView:
        def transition(connection: Connection, state: RegistryState) -> BenchmarkRegistry:
            view = self.repository.view(connection, state, principal.owner_id, benchmark_id)
            self.repository.require_inventory_scope(state, view)
            return record_selection_started(
                state.registry, benchmark_id, at=_event_time(state.registry)
            )

        self._change(principal, transition)
        return self.get_benchmark(principal, benchmark_id)

    def _manifest(self, model_version_id: str) -> ModelBundle:
        if model_version_id == "active":
            raise Forbidden("Choice freezing requires an immutable model version.")
        model = self.database.get_model(model_version_id)
        if (
            model.manifest.size_bytes > MAX_BUNDLE_MANIFEST_BYTES
            or model.manifest.sha256 != model.bundle_sha256
        ):
            raise ObjectIntegrityError(
                "The frozen model manifest differs from its committed identity."
            )
        with tempfile.TemporaryDirectory(prefix="freeze-", dir=self.work_root) as temporary:
            path = Path(temporary) / "manifest.json"
            materialize_object(self.store, model.manifest, path)
            manifest = ModelBundle.model_validate_json(path.read_bytes())
        if (
            manifest.model_version_id != model_version_id
            or manifest.architecture_id != model.architecture_id
        ):
            raise ObjectIntegrityError("The model manifest differs from its registered identity.")
        return manifest

    def freeze(
        self, principal: Principal, benchmark_id: str, model_version_id: str
    ) -> BenchmarkView:
        self.database._admin(principal)
        # Materialize only the bounded manifest. Weights and annotation objects
        # are not needed to bind choices already verified during registration.
        manifest = self._manifest(model_version_id)

        def transition(connection: Connection, state: RegistryState) -> BenchmarkRegistry:
            view = self.repository.view(connection, state, principal.owner_id, benchmark_id)
            self.repository.require_inventory_scope(state, view)
            active = connection.execute(
                "SELECT 1 FROM regimex_admin_runs a JOIN regimex_operations o "
                "ON a.operation_id=o.operation_id WHERE a.kind='training' "
                "AND a.request->>'benchmark_selection_id'=%s "
                "AND o.state IN ('queued','running','cancel_requested') LIMIT 1",
                (benchmark_id,),
            ).fetchone()
            if active is not None:
                raise Conflict(
                    "Choices cannot freeze while selection training is pending or running."
                )
            model = self.database._model(connection, model_version_id)
            semantics = DirectionSemantics(
                profile_sha256=manifest.analysis_profile_sha256,
                decoder_version=manifest.decoder_version,
            )
            labels = manifest.training.labels
            if labels.direction_semantics_sha256 != content_hash(semantics.model_dump(mode="json")):
                raise Conflict(
                    "Frozen training-label semantics differ from the model's decoder profile."
                )
            try:
                record = self.admin_repository._reference(
                    connection, principal.owner_id, labels.reference_set_id
                )
            except NotFound:
                if labels.kind != "automatic":
                    raise Forbidden(
                        "Human training labels require their persisted review provenance."
                    ) from None
                training_reference = ReferenceProvenance(
                    reference_set_sha256=labels.content_sha256,
                    kind=labels.kind,
                    policy=labels.policy,
                    semantics=semantics,
                )
            else:
                if record.references.identity() != labels:
                    raise Conflict(
                        "The frozen training-label identity differs from its registered provenance."
                    )
                training_reference = _reference(record)
            choices_sha256 = content_hash(
                {
                    "version": "frozen-model-choice/1.0",
                    "model_version_id": model_version_id,
                    "bundle_sha256": model.bundle_sha256,
                    "inference_policy_sha256": model.inference_policy_sha256,
                    "training": manifest.training.model_dump(mode="json"),
                    "analysis_profile_sha256": manifest.analysis_profile_sha256,
                }
            )
            registry = freeze_choices(
                state.registry,
                benchmark_id,
                at=_event_time(state.registry),
                choices_sha256=choices_sha256,
                model_artifact_sha256=model.bundle_sha256,
                training_dataset_hashes=(manifest.training.dataset_sha256,),
                training_reference=training_reference,
            )
            frozen = FrozenChoice(
                benchmark_id=benchmark_id,
                owner_id=principal.owner_id,
                model_version_id=model_version_id,
                bundle_sha256=model.bundle_sha256,
                inference_policy_sha256=model.inference_policy_sha256,
                choices_sha256=choices_sha256,
            )
            connection.execute(
                "INSERT INTO regimex_benchmark_choices (benchmark_id,model_version_id,document) "
                "VALUES (%s,%s,%s)",
                (benchmark_id, model_version_id, Jsonb(frozen.model_dump())),
            )
            return registry

        self._change(principal, transition)
        return self.get_benchmark(principal, benchmark_id)

    def open(self, principal: Principal, benchmark_id: str) -> BenchmarkView:
        def transition(connection: Connection, state: RegistryState) -> BenchmarkRegistry:
            view = self.repository.view(connection, state, principal.owner_id, benchmark_id)
            self.repository.require_inventory_scope(state, view)
            return open_benchmark(state.registry, benchmark_id, at=_event_time(state.registry))

        self._change(principal, transition)
        return self.get_benchmark(principal, benchmark_id)

    def authorize_selection(
        self, principal: Principal, spec: TrainingSpec, dataset: DatasetRecord
    ) -> None:
        self.database._admin(principal)
        with self.database.connection() as connection:
            self.repository.validate_selection(connection, principal.owner_id, spec, dataset)

    def authorize_final_evaluation(
        self, principal: Principal, benchmark_id: str, model_version_id: str
    ) -> FinalEvaluationGrant:
        self.database._admin(principal)
        with self.database.connection() as connection:
            self.repository.lock(connection)
            binding = self.repository.binding(connection, principal.owner_id, benchmark_id)
            dataset = self.admin_repository._dataset(
                connection, principal.owner_id, binding.dataset_id
            )
            references = self.admin_repository._reference(
                connection, principal.owner_id, binding.reference_set_id, allow_test=True
            )
            model = self.database._model(connection, model_version_id)
            spec = EvaluationSpec(
                model_version_id=model_version_id,
                dataset_id=binding.dataset_id,
                reference_set_id=binding.reference_set_id,
                partition="test",
                benchmark_id=benchmark_id,
            )
            return self.repository.validate_final_evaluation(
                connection, principal.owner_id, spec, model, dataset, references
            )
