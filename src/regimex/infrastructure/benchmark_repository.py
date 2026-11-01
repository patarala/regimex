"""Serialized benchmark revisions and transaction-bound scientific access checks."""

import hashlib
from collections.abc import Callable
from typing import cast

import psycopg
from psycopg.types.json import Jsonb

from regimex.admin_contracts import DatasetRecord, EvaluationSpec, ReferenceSetRecord, TrainingSpec
from regimex.benchmark_contracts import (
    BenchmarkBinding,
    BenchmarkView,
    FinalEvaluationGrant,
    FrozenChoice,
    RegistryState,
    dataset_identity_keys,
)
from regimex.benchmarks import BenchmarkRegistry, RegistryRevision, _member_keys, _preserves_history
from regimex.infrastructure.database import DatabaseRepository
from regimex.service_contracts import Conflict, Forbidden, ModelRecord, NotFound, Principal
from regimex.storage import json_bytes

Connection = psycopg.Connection[tuple[object, ...]]
Transition = Callable[[Connection, RegistryState], BenchmarkRegistry]


class BenchmarkRepository:
    def __init__(self, database: DatabaseRepository) -> None:
        self.database = database

    @staticmethod
    def lock(connection: Connection) -> None:
        connection.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended("
            "current_schema() || ':regimex-benchmark-registry',0))"
        )

    @staticmethod
    def load(connection: Connection) -> RegistryState:
        previous = BenchmarkRegistry()
        previous_sha256 = None
        revision_number = -1
        for row in connection.execute(
            "SELECT revision,sha256,previous_sha256,document "
            "FROM regimex_benchmark_revisions ORDER BY revision"
        ):
            revision = RegistryRevision.model_validate_json(json_bytes(row[3]))
            checksum = hashlib.sha256(json_bytes(revision.model_dump(mode="json"))).hexdigest()
            if (
                revision.revision != revision_number + 1
                or revision.revision != row[0]
                or revision.previous_sha256 != previous_sha256
                or revision.previous_sha256 != row[2]
                or checksum != row[1]
            ):
                raise Conflict("The immutable benchmark revision chain failed integrity checks.")
            try:
                _preserves_history(previous, revision.registry)
            except ValueError as exc:
                raise Conflict("The benchmark revision rewrites historical evidence.") from exc
            previous = revision.registry
            previous_sha256 = checksum
            revision_number = revision.revision
        return RegistryState(revision=revision_number, sha256=previous_sha256, registry=previous)

    @staticmethod
    def append(
        connection: Connection, state: RegistryState, registry: BenchmarkRegistry
    ) -> RegistryState:
        registry = BenchmarkRegistry.model_validate_json(registry.model_dump_json())
        _preserves_history(state.registry, registry)
        if registry == state.registry:
            return state
        revision = RegistryRevision(
            revision=state.revision + 1, previous_sha256=state.sha256, registry=registry
        )
        checksum = hashlib.sha256(json_bytes(revision.model_dump(mode="json"))).hexdigest()
        connection.execute(
            "INSERT INTO regimex_benchmark_revisions (revision,sha256,previous_sha256,document) "
            "VALUES (%s,%s,%s,%s)",
            (
                revision.revision,
                checksum,
                state.sha256,
                Jsonb(revision.model_dump(mode="json")),
            ),
        )
        return RegistryState(revision=revision.revision, sha256=checksum, registry=registry)

    def get_registry(self, principal: Principal) -> RegistryState:
        self.database._admin(principal)
        with self.database.connection() as connection:
            return self.load(connection)

    def change(self, principal: Principal, transition: Transition) -> RegistryState:
        self.database._admin(principal)
        with self.database.connection() as connection:
            self.lock(connection)
            state = self.load(connection)
            return self.append(connection, state, transition(connection, state))

    @staticmethod
    def binding(connection: Connection, owner_id: str, benchmark_id: str) -> BenchmarkBinding:
        row = connection.execute(
            "SELECT document FROM regimex_benchmark_bindings WHERE benchmark_id=%s AND owner_id=%s",
            (benchmark_id, owner_id),
        ).fetchone()
        if row is None:
            raise NotFound()
        return BenchmarkBinding.model_validate_json(json_bytes(row[0]))

    @staticmethod
    def choice(connection: Connection, benchmark_id: str) -> FrozenChoice | None:
        row = connection.execute(
            "SELECT document FROM regimex_benchmark_choices WHERE benchmark_id=%s", (benchmark_id,)
        ).fetchone()
        return None if row is None else FrozenChoice.model_validate_json(json_bytes(row[0]))

    @classmethod
    def view(
        cls, connection: Connection, state: RegistryState, owner_id: str, benchmark_id: str
    ) -> BenchmarkView:
        binding = cls.binding(connection, owner_id, benchmark_id)
        for reservation in state.registry.benchmarks:
            if reservation.benchmark_id == benchmark_id:
                if state.sha256 is None:
                    break
                frozen = cls.choice(connection, benchmark_id)
                if (reservation.freeze is None) != (frozen is None):
                    raise Conflict("The benchmark freeze and immutable model binding disagree.")
                return BenchmarkView(
                    binding=binding,
                    reservation=reservation,
                    frozen_choice=frozen,
                    registry_revision=state.revision,
                    registry_sha256=state.sha256,
                )
        raise Conflict("The benchmark binding is absent from its immutable journal.")

    def get_benchmark(self, principal: Principal, benchmark_id: str) -> BenchmarkView:
        self.database._admin(principal)
        with self.database.connection() as connection:
            self.lock(connection)
            return self.view(connection, self.load(connection), principal.owner_id, benchmark_id)

    @staticmethod
    def require_inventory_scope(state: RegistryState, view: BenchmarkView) -> None:
        """A later assertion cannot narrow the scope that justified this benchmark."""
        attestation = state.registry.history_attestation
        required = set(view.reservation.inventory_attestation.source_namespaces)
        if view.reservation.freeze is not None:
            required.update(view.reservation.freeze.inventory_attestation.source_namespaces)
        if attestation is None or not required <= set(attestation.source_namespaces):
            raise Forbidden(
                "The current inventory attestation must retain "
                "the reserved and frozen source scope."
            )

    @classmethod
    def validate_selection(
        cls, connection: Connection, owner_id: str, spec: TrainingSpec, dataset: DatasetRecord
    ) -> None:
        cls.lock(connection)
        if spec.benchmark_selection_id is None:
            raise Forbidden("Managed model selection requires a reserved benchmark.")
        state = cls.load(connection)
        view = cls.view(connection, state, owner_id, spec.benchmark_selection_id)
        cls.require_inventory_scope(state, view)
        reservation = view.reservation
        if (
            reservation.selection_started_at is None
            or reservation.freeze is not None
            or reservation.opened_at is not None
            or state.registry.history_attestation is None
        ):
            raise Forbidden("Selection requires a current attestation and an open selection phase.")
        row = connection.execute(
            "SELECT dataset_sha256 FROM regimex_benchmark_history_datasets "
            "WHERE dataset_id=%s AND owner_id=%s",
            (dataset.dataset_id, owner_id),
        ).fetchone()
        if (
            spec.dataset_id != dataset.dataset_id
            or dataset.owner_id != owner_id
            or row != (dataset.manifest.content_sha256,)
            or dataset.manifest.content_sha256
            not in {item.dataset_sha256 for item in state.registry.history}
        ):
            raise Forbidden(
                "The complete immutable development dataset must be registered as history."
            )

    @classmethod
    def validate_final_reference_import(
        cls, connection: Connection, owner_id: str, dataset: DatasetRecord
    ) -> None:
        """Seal final references before any history or reservation can expose them."""
        cls.lock(connection)
        if dataset.owner_id != owner_id:
            raise NotFound()
        state = cls.load(connection)
        keys = dataset_identity_keys(dataset.manifest.members)
        historical = {
            key
            for history in state.registry.history
            for membership in history.memberships
            for key in _member_keys(membership.member)
        }
        if (
            dataset.manifest.content_sha256
            in {history.dataset_sha256 for history in state.registry.history}
            or keys & historical
        ):
            raise Forbidden(
                "Final references require a dataset independent of all exposure history."
            )
        rows = connection.execute(
            "SELECT d.document FROM regimex_datasets d WHERE EXISTS ("
            "SELECT 1 FROM regimex_benchmark_history_datasets h WHERE h.dataset_id=d.dataset_id) "
            "OR EXISTS (SELECT 1 FROM regimex_benchmark_bindings b WHERE b.dataset_id=d.dataset_id)"
        )
        for row in rows:
            previous = DatasetRecord.model_validate_json(json_bytes(row[0]))
            if keys & dataset_identity_keys(previous.manifest.members):
                raise Forbidden(
                    "Final references cannot rewrite a reserved or previously exposed dataset."
                )

    @classmethod
    def validate_final_evaluation(
        cls,
        connection: Connection,
        owner_id: str,
        spec: EvaluationSpec,
        model: ModelRecord,
        dataset: DatasetRecord,
        references: ReferenceSetRecord,
    ) -> FinalEvaluationGrant:
        cls.lock(connection)
        benchmark_id = getattr(spec, "benchmark_id", None)
        if spec.partition != "test" or not isinstance(benchmark_id, str):
            raise Forbidden("Final evaluation requires an explicit dedicated benchmark request.")
        state = cls.load(connection)
        view = cls.view(connection, state, owner_id, benchmark_id)
        cls.require_inventory_scope(state, view)
        frozen = view.frozen_choice
        if (
            view.reservation.opened_at is None
            or view.reservation.freeze is None
            or frozen is None
            or state.registry.history_attestation is None
        ):
            raise Forbidden(
                "Final references remain closed until choices freeze and exposure is recorded."
            )
        binding = view.binding
        if (
            model.model_version_id != frozen.model_version_id
            or model.bundle_sha256 != frozen.bundle_sha256
            or model.inference_policy_sha256 != frozen.inference_policy_sha256
            or spec.model_version_id != frozen.model_version_id
            or dataset.owner_id != owner_id
            or dataset.dataset_id != binding.dataset_id
            or dataset.manifest.content_sha256 != binding.dataset_sha256
            or spec.dataset_id != binding.dataset_id
            or references.owner_id != owner_id
            or references.dataset_id != binding.dataset_id
            or references.references.reference_set_id != binding.reference_set_id
            or references.references.content_sha256 != binding.reference_sha256
            or references.references.partition != "test"
            or spec.reference_set_id != binding.reference_set_id
            or frozen.choices_sha256 != view.reservation.freeze.choices_sha256
        ):
            raise Forbidden(
                "Final evaluation differs from its frozen model, dataset or references."
            )
        return FinalEvaluationGrant(
            **binding.model_dump(exclude={"partition"}),
            model_version_id=frozen.model_version_id,
            bundle_sha256=frozen.bundle_sha256,
            inference_policy_sha256=frozen.inference_policy_sha256,
            choices_sha256=frozen.choices_sha256,
            opened_at=view.reservation.opened_at,
            registry_revision=state.revision,
            registry_sha256=cast(str, state.sha256),
        )
