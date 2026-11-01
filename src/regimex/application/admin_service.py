"""Administrative lifecycle use cases; scientific imports stay in child processes."""

import hashlib
import math
import tempfile
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast
from uuid import UUID

import psycopg
from pydantic import JsonValue, ValidationError

from regimex.admin_contracts import (
    AdminKind,
    AdminRunRecord,
    AdminSpec,
    AnnotationObject,
    CatalogBinding,
    DatasetRecord,
    DatasetSpec,
    EvaluationRecord,
    EvaluationSpec,
    FinalReferenceImportSpec,
    GenerationSpec,
    PublicationSpec,
    ReferenceGenerationSpec,
    ReferenceImportSpec,
    ReferenceSetRecord,
    TrainingSpec,
)
from regimex.application.admin_results import AdminResultCommitter
from regimex.application.service import AnalysisService, materialize_bundle, materialize_object
from regimex.benchmark_contracts import FinalEvaluationGrant
from regimex.benchmarks import DirectionSemantics
from regimex.computation_worker import (
    ChildLimits,
    ComputationCancelled,
    ComputationFailure,
    supervise_child,
)
from regimex.contracts import MAX_INPUT_BYTES, StrictModel
from regimex.data_catalog import (
    ReferencePayload,
    SourceSnapshot,
    SplitGroup,
    analysis_profile_hash,
    content_hash,
    create_reference_member,
    create_snapshot_member,
    seal_dataset,
    seal_reference_set,
    validate_reference_membership,
)
from regimex.directions import AnalysisConfig, DirectionAnnotations
from regimex.exports import ExportV2
from regimex.infrastructure.intake_repository import IntakeRequest
from regimex.infrastructure.object_store import ObjectCapacityError, ObjectInfo
from regimex.model_registry import RegistryError
from regimex.operations import OperationFailure
from regimex.service_contracts import (
    CapacityExceeded,
    Forbidden,
    ModelRecord,
    Principal,
    ServiceError,
    SnapshotRecord,
    StaleAttempt,
)
from regimex.service_settings import ServiceSettings
from regimex.storage import json_bytes

if TYPE_CHECKING:
    from regimex.infrastructure.admin_repository import AdminRepository

SelectionAuthorization = Callable[[Principal, TrainingSpec, DatasetRecord], None]
FinalAuthorization = Callable[[Principal, str, str], FinalEvaluationGrant]


@dataclass(frozen=True, slots=True)
class _InputPlan:
    context: dict[str, JsonValue]
    copies: tuple[tuple[ObjectInfo, Path], ...]
    model: ModelRecord | None


def _validate_request[T, **P](function: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    try:
        return function(*args, **kwargs)
    except ServiceError:
        raise
    except RegistryError as exc:
        raise ServiceError("unsupported_configuration", str(exc), 422) from exc
    except ValidationError as exc:
        raise ServiceError(
            "invalid_data", "The request failed strict configuration validation.", 422
        ) from exc
    except ValueError as exc:
        raise ServiceError("invalid_data", str(exc), 422) from exc


class AdminService:
    def __init__(
        self,
        analysis_service: AnalysisService,
        repository: AdminRepository,
        settings: ServiceSettings,
        *,
        authorize_selection: SelectionAuthorization | None = None,
        authorize_final_evaluation: FinalAuthorization | None = None,
    ) -> None:
        self.analysis_service = analysis_service
        self.repository = repository
        self.settings = settings
        self.authorize_selection = authorize_selection
        self.authorize_final_evaluation = authorize_final_evaluation

    @staticmethod
    def _admin(principal: Principal) -> None:
        if principal.role != "admin":
            raise Forbidden()

    @staticmethod
    def _profile(analysis: AnalysisConfig, profile_id: str) -> None:
        if profile_id != "direction-32/2.0" or analysis != AnalysisConfig():
            raise ServiceError(
                "unsupported_configuration",
                "The service supports the direction-32/2.0 analysis profile.",
                422,
            )

    def _submit(
        self,
        principal: Principal,
        spec: AdminSpec,
        *,
        idempotency_key: str,
        request_sha256: str | None = None,
        _final_grant: FinalEvaluationGrant | None = None,
    ) -> AdminRunRecord:
        self._admin(principal)
        checksum = request_sha256 or content_hash(spec.model_dump(mode="json"))
        previous = self.repository.find_replay(principal, spec.kind, idempotency_key, checksum)
        if previous is not None:
            return previous
        try:
            if isinstance(spec, GenerationSpec):
                if spec.plan.experiment_count > self.settings.max_generation_experiments:
                    raise CapacityExceeded("Generation exceeds the server experiment limit.")
            elif isinstance(spec, ReferenceGenerationSpec):
                self._profile(spec.analysis, spec.analysis_profile_id)
                self.repository.get_dataset(principal, spec.dataset_id)
            elif isinstance(spec, TrainingSpec):
                self._check_training(principal, spec)
            elif isinstance(spec, EvaluationSpec):
                self._check_evaluation(principal, spec, _final_grant)
            else:
                self._check_publication(principal, spec)
        except ServiceError:
            accepted = self.repository.find_replay(principal, spec.kind, idempotency_key, checksum)
            if accepted is not None:
                return accepted
            raise
        return self.repository.submit(
            principal,
            spec,
            idempotency_key=idempotency_key,
            request_sha256=checksum,
            max_pending=self.settings.max_admin_pending_per_owner,
        )

    def _check_training(self, principal: Principal, spec: TrainingSpec) -> None:
        dataset = self.repository.get_dataset(principal, spec.dataset_id)
        labels = self.repository.get_reference_set(principal, spec.labels_reference_set_id)
        validation = self.repository.get_reference_set(principal, spec.validation_reference_set_id)
        if (
            labels.references.partition != "train"
            or validation.references.partition != "validation"
        ):
            raise ServiceError(
                "invalid_data",
                "Training requires train labels and validation references.",
                422,
            )
        for references in (labels.references, validation.references):
            _validate_request(validate_reference_membership, dataset.manifest, references)
        if labels.references.semantics != validation.references.semantics:
            raise ServiceError(
                "invalid_data", "Training and validation direction semantics differ.", 422
            )
        self._profile(labels.references.analysis_profile, spec.analysis_profile_id)
        descriptor = _validate_request(
            lambda: self.analysis_service.registry.get(spec.architecture_id)
        )
        _validate_request(lambda: descriptor.validate_parameters(spec.parameters))
        _validate_request(lambda: descriptor.validate_training_policy(spec.policy))
        if (
            spec.training_runtime.time_budget_seconds > self.settings.admin_time_budget_seconds
            or spec.training_runtime.memory_budget_bytes > self.settings.admin_memory_budget_bytes
            or len(spec.policy.seeds) > self.settings.max_training_seeds
            or spec.policy.epochs > self.settings.max_training_epochs
        ):
            raise CapacityExceeded("Training exceeds the allowed computation budget.")
        if self.authorize_selection is None:
            raise Forbidden("Training requires an authorized reserved-benchmark selection scope.")
        self.authorize_selection(principal, spec, dataset)

    def _check_evaluation(
        self,
        principal: Principal,
        spec: EvaluationSpec,
        final_grant: FinalEvaluationGrant | None,
    ) -> None:
        self.analysis_service.get_model(principal, spec.model_version_id)
        dataset = self.repository.get_dataset(principal, spec.dataset_id)
        if spec.partition == "test" and (
            final_grant is None
            or (
                final_grant.benchmark_id,
                final_grant.model_version_id,
                final_grant.dataset_id,
                final_grant.reference_set_id,
            )
            != (
                spec.benchmark_id,
                spec.model_version_id,
                spec.dataset_id,
                spec.reference_set_id,
            )
        ):
            raise Forbidden("Final evaluation requires a frozen benchmark authorization.")
        evaluation_refs = self.repository.get_reference_set(
            principal, spec.reference_set_id, allow_test=final_grant is not None
        )
        if evaluation_refs.references.partition != spec.partition:
            raise Forbidden("Ordinary evaluation cannot access final references.")
        _validate_request(
            lambda: validate_reference_membership(dataset.manifest, evaluation_refs.references)
        )

    def _check_publication(self, principal: Principal, spec: PublicationSpec) -> None:
        model = self.analysis_service.get_model(principal, spec.model_version_id)
        evaluation = self.repository.get_evaluation(principal, spec.evaluation_run_id)
        if (model.model_version_id, model.bundle_sha256, model.inference_policy_sha256) != (
            evaluation.model_version_id,
            evaluation.bundle_sha256,
            evaluation.inference_policy_sha256,
        ):
            raise ServiceError(
                "invalid_data",
                "Publication evaluation differs from the immutable model or policy.",
                422,
            )

    def submit_generation(
        self,
        principal: Principal,
        spec: GenerationSpec,
        *,
        idempotency_key: str,
        request_sha256: str | None = None,
    ) -> AdminRunRecord:
        return self._submit(
            principal, spec, idempotency_key=idempotency_key, request_sha256=request_sha256
        )

    def generate_references(
        self,
        principal: Principal,
        spec: ReferenceGenerationSpec,
        *,
        idempotency_key: str,
        request_sha256: str | None = None,
    ) -> AdminRunRecord:
        return self._submit(
            principal, spec, idempotency_key=idempotency_key, request_sha256=request_sha256
        )

    def submit_training(
        self,
        principal: Principal,
        spec: TrainingSpec,
        *,
        idempotency_key: str,
        request_sha256: str | None = None,
    ) -> AdminRunRecord:
        return self._submit(
            principal, spec, idempotency_key=idempotency_key, request_sha256=request_sha256
        )

    def submit_evaluation(
        self,
        principal: Principal,
        spec: EvaluationSpec,
        *,
        idempotency_key: str,
        request_sha256: str | None = None,
    ) -> AdminRunRecord:
        if spec.partition != "validation":
            raise Forbidden("Use the dedicated benchmark endpoint for final evaluation.")
        return self._submit(
            principal, spec, idempotency_key=idempotency_key, request_sha256=request_sha256
        )

    def submit_final_evaluation(
        self,
        principal: Principal,
        benchmark_id: str,
        model_version_id: str,
        *,
        idempotency_key: str,
        request_sha256: str | None = None,
    ) -> AdminRunRecord:
        self._admin(principal)
        checksum = request_sha256 or content_hash(
            {"benchmark_id": benchmark_id, "model_version_id": model_version_id}
        )
        previous = self.repository.find_replay(principal, "evaluation", idempotency_key, checksum)
        if previous is not None:
            return previous
        if self.authorize_final_evaluation is None:
            raise Forbidden("Final benchmark authorization is unavailable.")
        grant = self.authorize_final_evaluation(principal, benchmark_id, model_version_id)
        spec = EvaluationSpec(
            model_version_id=model_version_id,
            dataset_id=grant.dataset_id,
            reference_set_id=grant.reference_set_id,
            partition="test",
            benchmark_id=benchmark_id,
        )
        return self._submit(
            principal,
            spec,
            idempotency_key=idempotency_key,
            request_sha256=checksum,
            _final_grant=grant,
        )

    def submit_publication(
        self,
        principal: Principal,
        spec: PublicationSpec,
        *,
        idempotency_key: str,
        request_sha256: str | None = None,
    ) -> AdminRunRecord:
        return self._submit(
            principal, spec, idempotency_key=idempotency_key, request_sha256=request_sha256
        )

    def get_run(self, principal: Principal, run_id: UUID) -> AdminRunRecord:
        return self.repository.get_run(principal, run_id)

    def list_runs(
        self,
        principal: Principal,
        *,
        kind: AdminKind | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[AdminRunRecord, ...]:
        return self.repository.list_runs(principal, kind=kind, limit=limit, offset=offset)

    def get_dataset(self, principal: Principal, dataset_id: str) -> DatasetRecord:
        return self.repository.get_dataset(principal, dataset_id)

    def list_datasets(
        self, principal: Principal, *, limit: int = 100, offset: int = 0
    ) -> tuple[DatasetRecord, ...]:
        return self.repository.list_datasets(principal, limit=limit, offset=offset)

    def get_reference_set(self, principal: Principal, reference_set_id: str) -> ReferenceSetRecord:
        return self.repository.get_reference_set(principal, reference_set_id)

    def list_reference_sets(
        self, principal: Principal, *, limit: int = 100, offset: int = 0
    ) -> tuple[ReferenceSetRecord, ...]:
        return self.repository.list_reference_sets(principal, limit=limit, offset=offset)

    def get_evaluation(self, principal: Principal, run_id: UUID) -> EvaluationRecord:
        return self.repository.get_evaluation(principal, run_id)

    def get_metrics(
        self, principal: Principal, model_version_id: str = "active"
    ) -> dict[str, JsonValue]:
        model = self.analysis_service.get_model(principal, model_version_id)
        evaluation = self.repository.get_publication_evaluation(model.model_version_id)
        return {
            "model_version_id": model.model_version_id,
            "bundle_sha256": model.bundle_sha256,
            "inference_policy_sha256": model.inference_policy_sha256,
            "quality_status": model.quality_status,
            "evaluation_available": evaluation is not None,
            "evaluation_run_id": None if evaluation is None else str(evaluation.run_id),
            "dataset_sha256": None if evaluation is None else evaluation.dataset_sha256,
            "reference_sha256": None if evaluation is None else evaluation.reference_sha256,
            "metrics": None if evaluation is None else evaluation.metrics,
        }

    def _binding(self, principal: Principal, snapshot: SnapshotRecord) -> CatalogBinding:
        existing = self.repository.get_binding(principal, snapshot.snapshot_id)
        if existing is not None:
            return existing
        return CatalogBinding(
            snapshot_id=snapshot.snapshot_id,
            source_namespace=content_hash([snapshot.owner_id, snapshot.source_id, snapshot.run_id]),
            split_group=SplitGroup(
                namespace=content_hash([snapshot.owner_id, snapshot.source_id]),
                group_id=snapshot.run_id,
                seed=None,
            ),
        )

    @staticmethod
    def _sync_request(
        spec: StrictModel, idempotency_key: str | None, request_sha256: str | None
    ) -> IntakeRequest | None:
        checksum = request_sha256
        if idempotency_key is not None and checksum is None:
            checksum = content_hash(spec.model_dump(mode="json"))
        return IntakeRequest.optional(idempotency_key, checksum)

    def create_dataset(
        self,
        principal: Principal,
        spec: DatasetSpec,
        *,
        idempotency_key: str | None = None,
        request_sha256: str | None = None,
    ) -> DatasetRecord:
        self._admin(principal)
        request = self._sync_request(spec, idempotency_key, request_sha256)
        if request is not None:
            previous = self.repository.find_dataset_replay(principal, **request.model_dump())
            if previous is not None:
                return previous
        if len(spec.members) > self.settings.max_generation_experiments:
            raise CapacityExceeded("Dataset membership exceeds the server experiment limit.")
        members = []
        bindings = []
        for selection in spec.members:
            snapshot = self.analysis_service.get_snapshot(principal, selection.snapshot_id)
            binding = self._binding(principal, snapshot)
            export = self.analysis_service.read_snapshot(principal, selection.snapshot_id)
            members.append(
                create_snapshot_member(
                    SourceSnapshot(
                        source_namespace=binding.source_namespace,
                        split_group=binding.split_group,
                        export=export,
                    ),
                    selection.partition,
                )
            )
            bindings.append(binding)
        record = DatasetRecord(
            dataset_id=spec.dataset_id,
            owner_id=principal.owner_id,
            manifest=_validate_request(lambda: seal_dataset(spec.dataset_id, tuple(members))),
            snapshot_ids=tuple(member.snapshot_id for member in spec.members),
            created_at=datetime.now(UTC),
        )
        return self.repository.add_dataset(
            principal,
            record,
            bindings=tuple(bindings),
            idempotency_key=None if request is None else request.idempotency_key,
            request_sha256=None if request is None else request.request_sha256,
        )

    def _store_json(self, path: Path, value: object, key: str) -> ObjectInfo:
        data = json_bytes(value)
        if len(data) > MAX_INPUT_BYTES:
            raise CapacityExceeded("Administrative JSON payload exceeds 16 MiB.")
        path.write_bytes(data)
        try:
            return self.analysis_service.store.put(
                key, path, expected_sha256=hashlib.sha256(data).hexdigest()
            )
        except ObjectCapacityError as exc:
            raise CapacityExceeded("Administrative storage capacity is exhausted.") from exc

    def import_references(
        self,
        principal: Principal,
        spec: ReferenceImportSpec,
        *,
        idempotency_key: str | None = None,
        request_sha256: str | None = None,
    ) -> ReferenceSetRecord:
        self._admin(principal)
        request = self._sync_request(spec, idempotency_key, request_sha256)
        if request is not None:
            previous = self.repository.find_reference_replay(principal, **request.model_dump())
            if previous is not None:
                return previous
        if spec.partition not in {"train", "validation"}:
            raise ServiceError(
                "invalid_data", "Ordinary reference imports require train or validation.", 422
            )
        spec = _validate_request(ReferenceImportSpec.model_validate_json, spec.model_dump_json())
        return self.repository.add_reference_set(
            principal,
            self._human_reference_record(principal, spec),
            idempotency_key=None if request is None else request.idempotency_key,
            request_sha256=None if request is None else request.request_sha256,
        )

    def import_final_references(
        self,
        principal: Principal,
        spec: FinalReferenceImportSpec,
        *,
        idempotency_key: str | None = None,
        request_sha256: str | None = None,
    ) -> ReferenceSetRecord:
        """Seal independently supplied final annotations before reservation or exposure."""
        self._admin(principal)
        request = self._sync_request(spec, idempotency_key, request_sha256)
        if request is not None:
            previous = self.repository.find_reference_replay(
                principal, final=True, **request.model_dump()
            )
            if previous is not None:
                return previous
        if spec.partition != "test":
            raise ServiceError(
                "invalid_data", "Final reference imports require the test partition.", 422
            )
        spec = _validate_request(
            FinalReferenceImportSpec.model_validate_json, spec.model_dump_json()
        )
        try:
            self.repository.check_final_reference_import(principal, spec.dataset_id)
            record = self._human_reference_record(principal, spec)
            return self.repository.add_final_reference_set(
                principal,
                record,
                idempotency_key=None if request is None else request.idempotency_key,
                request_sha256=None if request is None else request.request_sha256,
            )
        except ServiceError:
            # A concurrent request may have committed before reservation made
            # further imports unavailable. Its receipt still replays the result.
            if request is not None:
                accepted = self.repository.find_reference_replay(
                    principal, final=True, **request.model_dump()
                )
                if accepted is not None:
                    return accepted
            raise

    def _human_reference_record(
        self, principal: Principal, spec: ReferenceImportSpec | FinalReferenceImportSpec
    ) -> ReferenceSetRecord:
        self._profile(spec.analysis, spec.analysis_profile_id)
        dataset = self.get_dataset(principal, spec.dataset_id)
        expected = [
            (snapshot_id, member)
            for snapshot_id, member in zip(
                dataset.snapshot_ids, dataset.manifest.members, strict=True
            )
            if member.partition == spec.partition
        ]
        if tuple(item.snapshot_id for item in spec.annotations) != tuple(
            identity for identity, _ in expected
        ):
            raise ServiceError(
                "invalid_data", "Human annotations must cover the complete ordered partition.", 422
            )
        members = []
        for (snapshot_id, member), item in zip(expected, spec.annotations, strict=True):
            annotations = item.annotation
            export = self.analysis_service.read_snapshot(principal, snapshot_id)
            if (
                annotations.reference_kind != "human"
                or annotations.reviewer != spec.reviewer
                or annotations.policy_config != spec.analysis.model_dump(mode="json")
            ):
                raise ServiceError(
                    "invalid_data",
                    "Human annotations must match the supplied reviewer and analysis profile.",
                    422,
                )
            _validate_request(self._validate_annotations, export, annotations)
            members.append(create_reference_member(member, annotations))
        references = _validate_request(
            lambda: seal_reference_set(
                ReferencePayload(
                    reference_set_id=spec.reference_set_id,
                    dataset_sha256=dataset.manifest.content_sha256,
                    partition=spec.partition,
                    kind="human",
                    policy="human-directions/1.0",
                    semantics=DirectionSemantics(
                        profile_sha256=analysis_profile_hash(spec.analysis),
                        decoder_version="duration-viterbi/1.0",
                    ),
                    analysis_profile=spec.analysis,
                    reviewer=spec.reviewer,
                    blind_review=spec.blind_review,
                    independent=spec.independent,
                    members=tuple(members),
                )
            )
        )
        objects = []
        with tempfile.TemporaryDirectory(
            prefix="human-reference-", dir=self.analysis_service.work_root
        ) as temporary:
            for index, (reference_member, item) in enumerate(
                zip(references.members, spec.annotations, strict=True)
            ):
                annotations = item.annotation
                checksum = content_hash(annotations.model_dump(mode="json"))
                reference_key = content_hash([principal.owner_id, spec.reference_set_id])
                info = self._store_json(
                    Path(temporary) / f"annotation-{index}.json",
                    annotations.model_dump(mode="json"),
                    f"references/{reference_key}/{checksum}.json",
                )
                objects.append(
                    AnnotationObject(
                        source_namespace=reference_member.source_namespace,
                        snapshot_id=reference_member.snapshot_id,
                        object=info,
                    )
                )
        return ReferenceSetRecord(
            dataset_id=dataset.dataset_id,
            owner_id=principal.owner_id,
            references=references,
            annotations=tuple(objects),
            created_at=datetime.now(UTC),
        )

    @staticmethod
    def _validate_annotations(export: ExportV2, annotations: DirectionAnnotations) -> None:
        # A structural view preserves original IDs and grid without inventing a seed.
        class Grid:
            experiment_id = export.source.snapshot_id
            step_count = export.time_grid.step_count
            instruments = export.instruments
            observations = export.observations

        annotations.validate_for(Grid())

    def get_reference_annotations(
        self, principal: Principal, reference_set_id: str
    ) -> tuple[DirectionAnnotations, ...]:
        record = self.get_reference_set(principal, reference_set_id)
        if record.references.partition == "test":
            raise Forbidden("Final reference contents remain sealed.")
        result = []
        with tempfile.TemporaryDirectory(
            prefix="reference-read-", dir=self.analysis_service.work_root
        ) as temporary:
            for index, item in enumerate(record.annotations):
                path = Path(temporary) / f"annotation-{index}.json"
                materialize_object(self.analysis_service.store, item.object, path)
                result.append(DirectionAnnotations.model_validate_json(path.read_bytes()))
        return tuple(result)

    def read_evaluation_report(self, principal: Principal, run_id: UUID) -> dict[str, JsonValue]:
        record = self.get_evaluation(principal, run_id)
        with tempfile.TemporaryDirectory(
            prefix="evaluation-read-", dir=self.analysis_service.work_root
        ) as temporary:
            path = Path(temporary) / "evaluation.json"
            materialize_object(self.analysis_service.store, record.report, path)
            import json

            return cast(dict[str, JsonValue], json.loads(path.read_bytes()))

    def _inputs(
        self, principal: Principal, spec: AdminSpec, root: Path, beat: Callable[[], None]
    ) -> _InputPlan:
        (root / "input").mkdir()
        if isinstance(spec, GenerationSpec):
            return _InputPlan({}, (), None)
        context: dict[str, JsonValue] = {}
        copies: list[tuple[ObjectInfo, Path]] = []
        model = None
        final_grant = None
        if isinstance(spec, EvaluationSpec) and spec.partition == "test":
            if self.authorize_final_evaluation is None or spec.benchmark_id is None:
                raise Forbidden(
                    "Final evaluation requires a current frozen benchmark authorization."
                )
            final_grant = self.authorize_final_evaluation(
                principal, spec.benchmark_id, spec.model_version_id
            )
            if (final_grant.dataset_id, final_grant.reference_set_id) != (
                spec.dataset_id,
                spec.reference_set_id,
            ):
                raise Forbidden("The queued evaluation differs from its frozen benchmark grant.")
            context["final_grant"] = final_grant.model_dump(mode="json")
        if isinstance(spec, (TrainingSpec, EvaluationSpec, ReferenceGenerationSpec)):
            dataset = self.get_dataset(principal, spec.dataset_id)
            context["dataset"] = dataset.model_dump(mode="json")
            partitions = (
                {spec.partition} if not isinstance(spec, TrainingSpec) else {"train", "validation"}
            )
            snapshots: list[JsonValue] = []
            for index, (snapshot_id, member) in enumerate(
                zip(dataset.snapshot_ids, dataset.manifest.members, strict=True)
            ):
                if member.partition not in partitions:
                    continue
                snapshot = self.analysis_service.get_snapshot(principal, snapshot_id)
                binding = self._binding(principal, snapshot)
                path = f"input/snapshot-{index}.json"
                copies.append((snapshot.object, root / path))
                snapshots.append({"binding": binding.model_dump(mode="json"), "path": path})
                beat()
            context["snapshots"] = snapshots
            if isinstance(spec, (TrainingSpec, EvaluationSpec)):
                reference_ids = (
                    (spec.labels_reference_set_id, spec.validation_reference_set_id)
                    if isinstance(spec, TrainingSpec)
                    else (spec.reference_set_id,)
                )
                sets = [
                    self.repository.get_reference_set(
                        principal, identity, allow_test=final_grant is not None
                    )
                    for identity in reference_ids
                ]
                context["reference_sets"] = [item.model_dump(mode="json") for item in sets]
                annotations: list[JsonValue] = []
                for references in sets:
                    if (
                        references.references.partition not in {"train", "validation"}
                        and final_grant is None
                    ):
                        raise Forbidden(
                            "Training and ordinary evaluation cannot materialize final references."
                        )
                    for item in references.annotations:
                        path = f"input/annotation-{len(annotations)}.json"
                        copies.append((item.object, root / path))
                        annotations.append(
                            {
                                "source_namespace": item.source_namespace,
                                "snapshot_id": item.snapshot_id,
                                "path": path,
                            }
                        )
                        beat()
                context["annotations"] = annotations
        if isinstance(spec, (EvaluationSpec, PublicationSpec)):
            model = self.analysis_service.repository.get_model(spec.model_version_id)
            context["model"] = model.model_dump(mode="json")
            beat()
        return _InputPlan(context, tuple(copies), model)

    def _materialize_inputs(
        self, plan: _InputPlan, payload: dict[str, JsonValue], root: Path, beat: Callable[[], None]
    ) -> None:
        model_objects = (
            () if plan.model is None else (plan.model.manifest, *plan.model.members.values())
        )
        used = len(json_bytes(payload))
        total = (
            used
            + sum(info.size_bytes for info, _ in plan.copies)
            + sum(info.size_bytes for info in model_objects)
        )
        if total > self.settings.admin_scratch_budget_bytes:
            raise CapacityExceeded(
                "Materialized inputs and request metadata exceed the scratch budget."
            )

        def before_copy(info: ObjectInfo) -> None:
            nonlocal used
            beat()
            if used + info.size_bytes > self.settings.admin_scratch_budget_bytes:
                raise CapacityExceeded("Input materialization exceeds the scratch budget.")
            used += info.size_bytes

        for info, path in plan.copies:
            before_copy(info)
            materialize_object(self.analysis_service.store, info, path)
            beat()
        if plan.model is not None:
            materialize_bundle(
                self.analysis_service.store,
                plan.model,
                root / "input/model",
                before_copy=before_copy,
            )
            beat()

    def run_admin(self, operation_id: UUID) -> AdminRunRecord | None:
        operation = self.analysis_service.repository.claim_operation(operation_id)
        if operation is None:
            return None
        token = operation.attempt_token
        assert token is not None
        run = self.repository.get_run_internal(operation_id)
        principal = Principal(owner_id=operation.owner_id, role="admin", token_id=UUID(int=0))
        last_heartbeat = 0.0
        deadline: float | None = None

        def beat() -> None:
            nonlocal last_heartbeat
            now = time.monotonic()
            if deadline is not None and now > deadline:
                raise CapacityExceeded("The administrative attempt exceeded its wall-time budget.")
            if now - last_heartbeat < min(1.0, self.settings.heartbeat_seconds):
                return
            current = self.analysis_service.repository.heartbeat_operation(
                operation_id, token, "administrative-computation"
            )
            last_heartbeat = now
            if current.state == "cancel_requested":
                raise ComputationCancelled()

        try:
            with tempfile.TemporaryDirectory(
                prefix=f"admin-{operation_id}-", dir=self.analysis_service.work_root
            ) as temporary:
                root = Path(temporary)
                if isinstance(run.spec, TrainingSpec):
                    runtime = run.spec.training_runtime
                    if (
                        runtime.time_budget_seconds > self.settings.admin_time_budget_seconds
                        or runtime.memory_budget_bytes > self.settings.admin_memory_budget_bytes
                    ):
                        raise CapacityExceeded(
                            "Queued training exceeds the current worker resource limits."
                        )
                    seconds, memory, threads = (
                        runtime.time_budget_seconds,
                        runtime.memory_budget_bytes,
                        runtime.threads,
                    )
                else:
                    seconds, memory, threads = (
                        self.settings.admin_time_budget_seconds,
                        self.settings.admin_memory_budget_bytes,
                        1,
                    )
                deadline = time.monotonic() + seconds
                inputs = self._inputs(principal, run.spec, root, beat)
                payload: dict[str, JsonValue] = {
                    "spec": run.spec.model_dump(mode="json"),
                    "run_id": str(run.run_id),
                    "owner_id": principal.owner_id,
                    "inputs": inputs.context,
                }
                self._materialize_inputs(inputs, payload, root, beat)
                beat()
                remaining = max(1, math.ceil(deadline - time.monotonic()))
                result = supervise_child(
                    payload,
                    root,
                    limits=ChildLimits(
                        remaining, memory, self.settings.admin_scratch_budget_bytes, threads
                    ),
                    heartbeat=beat,
                )
                return AdminResultCommitter(
                    analysis_service=self.analysis_service,
                    repository=self.repository,
                    run=run,
                    principal=principal,
                    root=root,
                    heartbeat=beat,
                ).commit(result)
        except StaleAttempt:
            return self.repository.get_run_internal(operation_id)
        except Exception as exc:
            if isinstance(exc, ComputationFailure):
                failure = exc.failure
            elif isinstance(exc, (CapacityExceeded, ObjectCapacityError)):
                failure = OperationFailure(
                    code="capacity_limit",
                    message="The administrative computation exceeds its configured budget.",
                )
            elif isinstance(exc, ServiceError):
                failure = OperationFailure(
                    code="unsupported_configuration"
                    if exc.code in {"unsupported_configuration", "forbidden"}
                    else "invalid_data",
                    message=exc.message[:1024],
                )
            elif isinstance(exc, ValueError):
                failure = OperationFailure(
                    code="invalid_data",
                    message="Administrative data or artifact validation failed.",
                )
            else:
                failure = OperationFailure(
                    code="infrastructure_failure",
                    message="The administrative worker could not complete its attempt.",
                    retryable=isinstance(
                        exc,
                        (
                            OSError,
                            psycopg.OperationalError,
                            psycopg.InterfaceError,
                            psycopg.errors.SerializationFailure,
                            psycopg.errors.DeadlockDetected,
                        ),
                    ),
                )
            with suppress(StaleAttempt):
                self.analysis_service.repository.fail_operation(operation_id, token, failure)
            return self.repository.get_run_internal(operation_id)
