"""Administrative HTTP workflows over explicit application operations."""

import hashlib
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Annotated, Literal, Protocol, Self
from uuid import UUID, uuid4

from fastapi import Depends, FastAPI, Header, Query, Request, Response
from pydantic import (
    AfterValidator,
    AwareDatetime,
    ConfigDict,
    Field,
    JsonValue,
    RootModel,
    field_validator,
)

from regimex import __version__
from regimex.admin_contracts import (
    MAX_ARCHIVE_SELECTION,
    MAX_PROGRESS_POINTS,
    AdminKind,
    AdminRunRecord,
    DatasetRecord,
    DatasetSelection,
    DatasetSpec,
    EvaluationRecord,
    EvaluationSpec,
    FinalReferenceImportSpec,
    GenerationSpec,
    HumanAnnotations,
    ImmutableModelVersion,
    ImportedTrainingSpec,
    PublicationSpec,
    RawGenerationPlan,
    ReferenceGenerationSpec,
    ReferenceImportSpec,
    ReferenceSetRecord,
    RequestDigest,
    RequestFingerprint,
    SplitPolicy,
    TrainingArchiveFilter,
    TrainingMetrics,
    TrainingProgress,
    TrainingRunName,
    TrainingSpec,
    distinct_snapshot_selections,
)
from regimex.api import OperationView, _json_dependency, _launch_source, _operation, _request_body
from regimex.contracts import Identifier, StrictModel
from regimex.data_catalog import (
    CURRENT_TRAINING_SEMANTICS,
    DatasetManifest,
    ReferenceSet,
    content_hash,
)
from regimex.deletion_contracts import DeletionPreview
from regimex.directions import (
    DEFAULT_ANALYSIS_PROFILE,
    AnalysisConfig,
    DirectionAnnotations,
)
from regimex.external_evaluations import ExternalEvaluation
from regimex.external_latency import ExternalLatencyEvidence
from regimex.history_contracts import DisplayName, DisplayNameUpdate
from regimex.model_registry import (
    LEGACY_SEGMENT_MOE_ID,
    SEGMENT_MOE_ID,
    ArchitectureId,
    InferenceRuntime,
    TrainingPolicySpec,
    TrainingRuntime,
    canonical_architecture_id,
    configured_parameter_count,
    resolve_training_policy,
)
from regimex.operations import LaunchSource
from regimex.preparation import (
    PreparationPage,
    PreparationRequest,
    PreparationRetry,
    PreparationView,
)
from regimex.service_contracts import NotFound, Principal, RegistryIdentifier
from regimex.storage import json_bytes
from regimex.training_statistics import TrainingRunStatistics


class AdminAPIService(Protocol):
    def start_preparation(
        self,
        principal: Principal,
        preparation_id: UUID,
        request: PreparationRequest,
        launch_source: LaunchSource,
    ) -> PreparationView: ...
    def get_preparation(self, principal: Principal, preparation_id: UUID) -> PreparationView: ...
    def list_preparations(
        self, principal: Principal, limit: int, offset: int
    ) -> PreparationPage: ...
    def retry_preparation(
        self, principal: Principal, preparation_id: UUID, operation_id: UUID
    ) -> PreparationView: ...

    @property
    def max_generation_experiments(self) -> int: ...

    def set_training_archive(
        self, principal: Principal, run_ids: tuple[UUID, ...], *, archived: bool
    ) -> None: ...

    def rename_training_run(
        self, principal: Principal, run_id: UUID, update: DisplayNameUpdate
    ) -> TrainingRunName: ...

    def preview_weight_deletion(self, principal: Principal, run_id: UUID) -> DeletionPreview: ...

    def training_actions(
        self, principal: Principal, run_id: UUID, *, include_final: bool, limit: int, offset: int
    ) -> tuple[AdminRunRecord, ...]: ...

    def training_metrics(self, principal: Principal, run_id: UUID) -> TrainingMetrics: ...

    def training_progress(
        self, principal: Principal, run_id: UUID, *, attempt: int | None, after: int
    ) -> TrainingProgress: ...

    def import_final_references(
        self,
        principal: Principal,
        spec: FinalReferenceImportSpec,
        *,
        idempotency_key: str | None = None,
        request_sha256: RequestDigest | None = None,
    ) -> ReferenceSetRecord: ...

    def submit_final_evaluation(
        self,
        principal: Principal,
        benchmark_id: str,
        model_version_id: str,
        *,
        idempotency_key: str,
        launch_source: LaunchSource | None = None,
    ) -> AdminRunRecord: ...

    def submit_generation(
        self,
        principal: Principal,
        spec: GenerationSpec,
        *,
        idempotency_key: str,
        launch_source: LaunchSource | None = None,
        request_sha256: str | None = None,
    ) -> AdminRunRecord: ...

    def create_dataset(
        self,
        principal: Principal,
        spec: DatasetSpec,
        *,
        idempotency_key: str | None = None,
        request_sha256: str | None = None,
    ) -> DatasetRecord: ...

    def generate_references(
        self,
        principal: Principal,
        spec: ReferenceGenerationSpec,
        *,
        idempotency_key: str,
        launch_source: LaunchSource | None = None,
        request_sha256: RequestDigest | None = None,
    ) -> AdminRunRecord: ...

    def import_references(
        self,
        principal: Principal,
        spec: ReferenceImportSpec,
        *,
        idempotency_key: str | None = None,
        request_sha256: RequestDigest | None = None,
    ) -> ReferenceSetRecord: ...

    def submit_training(
        self,
        principal: Principal,
        spec: TrainingSpec,
        *,
        idempotency_key: str,
        launch_source: LaunchSource | None = None,
        request_sha256: RequestDigest | None = None,
        display_name: str | None = None,
    ) -> AdminRunRecord: ...

    def submit_evaluation(
        self,
        principal: Principal,
        spec: EvaluationSpec,
        *,
        idempotency_key: str,
        launch_source: LaunchSource | None = None,
        request_sha256: str | None = None,
    ) -> AdminRunRecord: ...

    def submit_publication(
        self,
        principal: Principal,
        spec: PublicationSpec,
        *,
        idempotency_key: str,
        launch_source: LaunchSource | None = None,
        request_sha256: str | None = None,
    ) -> AdminRunRecord: ...

    def get_run(self, principal: Principal, run_id: UUID) -> AdminRunRecord: ...

    def list_runs(
        self,
        principal: Principal,
        *,
        kind: AdminKind | None = None,
        archive: TrainingArchiveFilter = "all",
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[AdminRunRecord, ...]: ...

    def get_dataset(self, principal: Principal, dataset_id: str) -> DatasetRecord: ...

    def list_datasets(
        self,
        principal: Principal,
        *,
        development_only: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[DatasetRecord, ...]: ...

    def get_reference_set(
        self, principal: Principal, reference_set_id: str
    ) -> ReferenceSetRecord: ...

    def list_reference_sets(
        self,
        principal: Principal,
        *,
        dataset_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[ReferenceSetRecord, ...]: ...

    def get_reference_annotations(
        self,
        principal: Principal,
        reference_set_id: str,
    ) -> tuple[DirectionAnnotations, ...]: ...

    def get_evaluation(self, principal: Principal, run_id: UUID) -> EvaluationRecord: ...

    def read_evaluation_report(
        self,
        principal: Principal,
        run_id: UUID,
    ) -> dict[str, JsonValue]: ...

    def get_metrics(
        self,
        principal: Principal,
        model_version_id: str = "active",
    ) -> dict[str, JsonValue]: ...


class GeneratedDatasetRequest(StrictModel):
    mode: Literal["generate"]
    plan: RawGenerationPlan
    splits: SplitPolicy = Field(default_factory=SplitPolicy)


class SnapshotDatasetRequest(StrictModel):
    mode: Literal["snapshots"]
    members: Annotated[
        tuple[DatasetSelection, ...],
        Field(min_length=3, max_length=1000),
        AfterValidator(distinct_snapshot_selections),
    ]


class DatasetRequest(
    RootModel[
        Annotated[GeneratedDatasetRequest | SnapshotDatasetRequest, Field(discriminator="mode")]
    ]
):
    model_config = ConfigDict(frozen=True, strict=True, allow_inf_nan=False)


class AutomaticReferenceRequest(StrictModel):
    mode: Literal["automatic"]
    dataset_id: RegistryIdentifier
    partition: Literal["train", "validation", "test"]
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    analysis_profile_id: Identifier = DEFAULT_ANALYSIS_PROFILE


class HumanReferenceRequest(StrictModel):
    mode: Literal["human_import"]
    dataset_id: RegistryIdentifier
    partition: Literal["train", "validation"]
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    analysis_profile_id: Identifier = DEFAULT_ANALYSIS_PROFILE
    reviewer: Identifier
    blind_review: bool
    independent: bool
    annotations: HumanAnnotations


class ReferenceRequest(
    RootModel[
        Annotated[AutomaticReferenceRequest | HumanReferenceRequest, Field(discriminator="mode")]
    ]
):
    model_config = ConfigDict(frozen=True, strict=True, allow_inf_nan=False)


class TrainingRunRequest(StrictModel):
    # Omitted and null names retain the request digest used by existing receipts.
    display_name: DisplayName | None = Field(default=None, exclude_if=lambda value: value is None)
    dataset_id: RegistryIdentifier
    labels_reference_set_id: RegistryIdentifier
    validation_reference_set_id: RegistryIdentifier
    architecture_id: ArchitectureId
    parameters: dict[str, JsonValue]
    policy: TrainingPolicySpec
    training_runtime: TrainingRuntime = Field(default_factory=TrainingRuntime)
    inference_runtime: InferenceRuntime = Field(default_factory=InferenceRuntime)
    analysis_profile_id: Identifier = DEFAULT_ANALYSIS_PROFILE
    benchmark_selection_id: RegistryIdentifier

    def resolve(self) -> tuple[Self, RequestFingerprint]:
        architecture = canonical_architecture_id(self.architecture_id)
        policy = resolve_training_policy(architecture, self.policy)
        resolved = self.model_copy(update={"architecture_id": architecture, "policy": policy})
        fingerprint = RequestFingerprint.for_profile_request(resolved)
        if architecture == SEGMENT_MOE_ID:
            previous = resolved.model_copy(update={"architecture_id": LEGACY_SEGMENT_MOE_ID})
        elif policy is not self.policy:
            # Earlier receipts expanded an omitted baseline seed list to three seeds.
            previous = self
        else:
            return resolved, fingerprint
        original = RequestFingerprint.for_profile_request(previous)
        return resolved, RequestFingerprint(
            sha256=fingerprint.sha256,
            previous_hashes=(
                *fingerprint.previous_hashes,
                original.sha256,
                *original.previous_hashes,
            ),
        )


class PublicationRequest(StrictModel):
    evaluation_run_id: UUID
    expected_previous: RegistryIdentifier | None = None


class EvaluationRequest(StrictModel):
    model_version_id: ImmutableModelVersion
    dataset_id: RegistryIdentifier
    reference_set_id: RegistryIdentifier
    partition: Literal["validation"] = "validation"


class ImportedTrainingView(ImportedTrainingSpec):
    external_evaluations: tuple[ExternalEvaluation, ...]
    external_latency: ExternalLatencyEvidence | None


class AdminRunView(StrictModel):
    run_id: UUID
    job: OperationView
    output: dict[str, JsonValue] | None
    weights_deletion_id: UUID | None = None
    archived_at: AwareDatetime | None = None
    polling_url: str
    training: TrainingSpec | ImportedTrainingView | None = None
    final_benchmark_id: Identifier | None = Field(
        default=None, exclude_if=lambda item: item is None
    )
    external_evaluations: tuple[ExternalEvaluation, ...] = Field(
        default=(), exclude_if=lambda items: not items
    )
    external_latency: ExternalLatencyEvidence | None = Field(
        default=None, exclude_if=lambda item: item is None
    )
    trainable_parameters: int | None = None
    statistics: TrainingRunStatistics | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    display_name: DisplayName | None = None
    name_revision: Annotated[int, Field(ge=0)] = 0


class AdminRunPage(StrictModel):
    items: tuple[AdminRunView, ...]
    limit: int
    offset: int


class TrainingValidationScore(StrictModel):
    model_version_id: Identifier
    validation_segment_macro_f1: Annotated[float, Field(ge=0, le=1)]


class TrainingArchiveUpdate(StrictModel):
    run_ids: Annotated[tuple[UUID, ...], Field(min_length=1, max_length=MAX_ARCHIVE_SELECTION)]
    archived: bool

    @field_validator("run_ids")
    @classmethod
    def unique_runs(cls, run_ids: tuple[UUID, ...]) -> tuple[UUID, ...]:
        if len(set(run_ids)) != len(run_ids):
            raise ValueError("Each training run must be listed only once.")
        return run_ids


class DatasetView(StrictModel):
    dataset_id: str
    content_sha256: str
    split_manifest_sha256: str
    snapshot_count: int
    partitions: dict[str, int]
    created_at: AwareDatetime
    manifest_url: str


class DatasetPage(StrictModel):
    items: tuple[DatasetView, ...]
    limit: int
    offset: int
    max_generation_experiments: int


class ReferenceView(StrictModel):
    reference_set_id: str
    dataset_id: str
    content_sha256: str
    partition: Literal["train", "validation", "test"]
    kind: Literal["automatic", "human"]
    policy: str
    independent: bool
    blind_review: bool
    reviewer: str | None
    created_at: AwareDatetime
    annotations_sealed: bool
    current_training_profile: bool
    manifest_url: str


class ReferencePage(StrictModel):
    items: tuple[ReferenceView, ...]
    limit: int
    offset: int


def _run(record: AdminRunRecord, *, include_final: bool = False) -> AdminRunView:
    allowed = {
        "dataset_id",
        "dataset_sha256",
        "reference_set_id",
        "reference_sha256",
        "model_version_id",
        "architecture_id",
        "bundle_sha256",
        "manifest_sha256",
        "quality_status",
        "validation_segment_macro_f1",
        "evaluation_run_id",
        "published_model_version_id",
    }
    training: TrainingSpec | ImportedTrainingView | None = None
    saved_count = (record.output or {}).get("trainable_parameters")
    count = None
    if isinstance(record.spec, ImportedTrainingSpec):
        training = ImportedTrainingView(
            **record.spec.model_dump(),
            external_evaluations=record.external_evaluations,
            external_latency=record.external_latency,
        )
        parameters = (training.configuration or {}).get("parameters")
        if isinstance(parameters, dict):
            count = configured_parameter_count(training.architecture_id, parameters)
    elif isinstance(record.spec, TrainingSpec):
        training = record.spec
        count = (
            saved_count
            if type(saved_count) is int
            else configured_parameter_count(training.architecture_id, training.parameters)
        )
    if training is not None:
        training = training.model_copy(
            update={"architecture_id": canonical_architecture_id(training.architecture_id)}
        )
    return AdminRunView(
        run_id=record.run_id,
        job=_operation(record.operation),
        training=training,
        final_benchmark_id=record.spec.benchmark_id
        if include_final and isinstance(record.spec, EvaluationSpec)
        else None,
        external_evaluations=record.external_evaluations
        if isinstance(training, TrainingSpec)
        else (),
        external_latency=record.external_latency if isinstance(training, TrainingSpec) else None,
        trainable_parameters=count,
        statistics=TrainingRunStatistics.from_operation(
            record.operation, record.output, now=datetime.now(UTC)
        )
        if isinstance(training, TrainingSpec)
        else None,
        display_name=record.display_name,
        name_revision=record.name_revision,
        output=None
        if record.output is None
        else {name: value for name, value in record.output.items() if name in allowed},
        weights_deletion_id=record.weights_deletion_id,
        archived_at=record.archived_at,
        polling_url=f"/api/v1/admin-runs/{record.run_id}",
    )


def _dataset(record: DatasetRecord) -> DatasetView:
    counts = dict.fromkeys(("train", "validation", "test"), 0)
    for member in record.manifest.members:
        counts[member.partition] += 1
    return DatasetView(
        dataset_id=record.dataset_id,
        content_sha256=record.manifest.content_sha256,
        split_manifest_sha256=record.manifest.split_manifest_sha256,
        snapshot_count=len(record.snapshot_ids),
        partitions=counts,
        created_at=record.created_at,
        manifest_url=f"/api/v1/datasets/{record.dataset_id}/manifest",
    )


def _reference(record: ReferenceSetRecord) -> ReferenceView:
    references = record.references
    return ReferenceView(
        reference_set_id=references.reference_set_id,
        dataset_id=record.dataset_id,
        content_sha256=references.content_sha256,
        partition=references.partition,
        kind=references.kind,
        policy=references.policy,
        independent=references.independent,
        blind_review=references.blind_review,
        reviewer=references.reviewer,
        created_at=record.created_at,
        annotations_sealed=references.partition == "test",
        current_training_profile=references.semantics == CURRENT_TRAINING_SEMANTICS,
        manifest_url=f"/api/v1/reference-sets/{references.reference_set_id}/manifest",
    )


def install_admin_routes(
    app: FastAPI, service: AdminAPIService, *, authenticate: Callable[..., Principal]
) -> None:
    Identity = Annotated[Principal, Depends(authenticate)]
    Key = Annotated[Identifier, Header(alias="Idempotency-Key")]
    Limit = Annotated[int, Query(ge=1, le=100)]
    Offset = Annotated[int, Query(ge=0, le=1_000_000)]
    DatasetBody = Annotated[DatasetRequest, Depends(_json_dependency(DatasetRequest))]
    ReferenceBody = Annotated[ReferenceRequest, Depends(_json_dependency(ReferenceRequest))]
    TrainingBody = Annotated[TrainingRunRequest, Depends(_json_dependency(TrainingRunRequest))]
    NameBody = Annotated[DisplayNameUpdate, Depends(_json_dependency(DisplayNameUpdate))]
    ArchiveBody = Annotated[TrainingArchiveUpdate, Depends(_json_dependency(TrainingArchiveUpdate))]
    EvaluationBody = Annotated[EvaluationRequest, Depends(_json_dependency(EvaluationRequest))]
    PublicationBody = Annotated[PublicationRequest, Depends(_json_dependency(PublicationRequest))]
    PreparationBody = Annotated[PreparationRequest, Depends(_json_dependency(PreparationRequest))]
    RetryBody = Annotated[PreparationRetry, Depends(_json_dependency(PreparationRetry))]
    app.state.request_models += (
        PreparationRequest,
        PreparationRetry,
        DatasetRequest,
        ReferenceRequest,
        TrainingRunRequest,
        DisplayNameUpdate,
        TrainingArchiveUpdate,
        EvaluationRequest,
        PublicationRequest,
    )

    @app.put(
        "/api/v1/training-preparations/{preparation_id}",
        status_code=202,
        openapi_extra=_request_body(PreparationRequest),
    )
    def prepare_data(
        preparation_id: UUID, identity: Identity, intake: PreparationBody, http_request: Request
    ) -> PreparationView:
        return service.start_preparation(
            identity, preparation_id, intake, _launch_source(http_request, identity)
        )

    @app.get("/api/v1/training-preparations/{preparation_id}")
    def preparation(preparation_id: UUID, identity: Identity) -> PreparationView:
        return service.get_preparation(identity, preparation_id)

    @app.get("/api/v1/training-preparations")
    def preparations(identity: Identity, limit: Limit = 25, offset: Offset = 0) -> PreparationPage:
        return service.list_preparations(identity, limit, offset)

    @app.post(
        "/api/v1/training-preparations/{preparation_id}/retry",
        openapi_extra=_request_body(PreparationRetry),
    )
    def retry_preparation(
        preparation_id: UUID, identity: Identity, retry: RetryBody
    ) -> PreparationView:
        return service.retry_preparation(identity, preparation_id, retry.operation_id)

    @app.post(
        "/api/v1/datasets",
        status_code=201,
        response_model=None,
        responses={
            201: {"model": DatasetView, "description": "Dataset created from snapshots"},
            202: {"model": AdminRunView, "description": "Dataset generation queued"},
        },
        openapi_extra=_request_body(DatasetRequest),
    )
    def create_dataset(
        identity: Identity,
        intake: DatasetBody,
        http_request: Request,
        response: Response,
        idempotency_key: Key,
    ) -> AdminRunView | DatasetView:
        request = intake.root
        digest = content_hash(request.model_dump(mode="json"))
        if isinstance(request, GeneratedDatasetRequest):
            record = service.submit_generation(
                identity,
                GenerationSpec(
                    dataset_id=f"dataset-{uuid4().hex}", plan=request.plan, splits=request.splits
                ),
                idempotency_key=idempotency_key,
                launch_source=_launch_source(http_request, identity),
                request_sha256=digest,
            )
            response.status_code = 202
            response.headers["Location"] = f"/api/v1/admin-runs/{record.run_id}"
            return _run(record)
        dataset_id = (
            "dataset-"
            + hashlib.sha256(
                json_bytes({"owner_id": identity.owner_id, "request": digest})
            ).hexdigest()[:40]
        )
        dataset = service.create_dataset(
            identity,
            DatasetSpec(dataset_id=dataset_id, members=request.members),
            idempotency_key=idempotency_key,
            request_sha256=digest,
        )
        response.headers["Location"] = f"/api/v1/datasets/{dataset.dataset_id}"
        return _dataset(dataset)

    @app.get("/api/v1/datasets")
    def list_datasets(
        identity: Identity, limit: Limit = 100, offset: Offset = 0, development_only: bool = False
    ) -> DatasetPage:
        return DatasetPage(
            items=tuple(
                _dataset(item)
                for item in service.list_datasets(
                    identity, development_only=development_only, limit=limit, offset=offset
                )
            ),
            limit=limit,
            offset=offset,
            max_generation_experiments=service.max_generation_experiments,
        )

    @app.get("/api/v1/datasets/{dataset_id}")
    def get_dataset(dataset_id: RegistryIdentifier, identity: Identity) -> DatasetView:
        return _dataset(service.get_dataset(identity, dataset_id))

    @app.get("/api/v1/datasets/{dataset_id}/manifest")
    def dataset_manifest(dataset_id: RegistryIdentifier, identity: Identity) -> DatasetManifest:
        return service.get_dataset(identity, dataset_id).manifest

    @app.post(
        "/api/v1/reference-sets",
        status_code=201,
        response_model=None,
        responses={
            201: {"model": ReferenceView, "description": "Human references imported"},
            202: {"model": AdminRunView, "description": "Automatic reference generation queued"},
        },
        openapi_extra=_request_body(ReferenceRequest),
    )
    def create_references(
        identity: Identity,
        intake: ReferenceBody,
        http_request: Request,
        response: Response,
        idempotency_key: Key,
    ) -> AdminRunView | ReferenceView:
        request = intake.root
        fingerprint = RequestFingerprint.for_profile_request(request)
        if isinstance(request, AutomaticReferenceRequest):
            record = service.generate_references(
                identity,
                ReferenceGenerationSpec(
                    dataset_id=request.dataset_id,
                    reference_set_id=f"reference-{uuid4().hex}",
                    partition=request.partition,
                    analysis=request.analysis,
                    analysis_profile_id=request.analysis_profile_id,
                ),
                idempotency_key=idempotency_key,
                launch_source=_launch_source(http_request, identity),
                request_sha256=fingerprint,
            )
            response.status_code = 202
            response.headers["Location"] = f"/api/v1/admin-runs/{record.run_id}"
            return _run(record)
        reference_id = (
            "reference-"
            + hashlib.sha256(
                json_bytes({"owner_id": identity.owner_id, "request": fingerprint.sha256})
            ).hexdigest()[:40]
        )
        references = service.import_references(
            identity,
            ReferenceImportSpec(
                **request.model_dump(exclude={"mode"}), reference_set_id=reference_id
            ),
            idempotency_key=idempotency_key,
            request_sha256=fingerprint,
        )
        response.headers["Location"] = (
            f"/api/v1/reference-sets/{references.references.reference_set_id}"
        )
        return _reference(references)

    @app.get("/api/v1/reference-sets")
    def list_references(
        identity: Identity,
        limit: Limit = 100,
        offset: Offset = 0,
        dataset_id: RegistryIdentifier | None = None,
    ) -> ReferencePage:
        return ReferencePage(
            items=tuple(
                _reference(item)
                for item in service.list_reference_sets(
                    identity, dataset_id=dataset_id, limit=limit, offset=offset
                )
            ),
            limit=limit,
            offset=offset,
        )

    @app.get("/api/v1/reference-sets/{reference_set_id}")
    def get_references(reference_set_id: RegistryIdentifier, identity: Identity) -> ReferenceView:
        return _reference(service.get_reference_set(identity, reference_set_id))

    @app.get("/api/v1/reference-sets/{reference_set_id}/manifest")
    def reference_manifest(
        reference_set_id: RegistryIdentifier, identity: Identity
    ) -> ReferenceSet:
        return service.get_reference_set(identity, reference_set_id).references

    @app.get("/api/v1/reference-sets/{reference_set_id}/annotations")
    def reference_annotations(
        reference_set_id: RegistryIdentifier, identity: Identity
    ) -> tuple[DirectionAnnotations, ...]:
        return service.get_reference_annotations(identity, reference_set_id)

    @app.post(
        "/api/v1/training-runs", status_code=202, openapi_extra=_request_body(TrainingRunRequest)
    )
    def train(
        identity: Identity,
        intake: TrainingBody,
        http_request: Request,
        response: Response,
        idempotency_key: Key,
    ) -> AdminRunView:
        resolved, fingerprint = intake.resolve()
        record = service.submit_training(
            identity,
            TrainingSpec(
                **resolved.model_dump(exclude={"display_name"}),
                model_version_id=f"model-{uuid4().hex}",
                build_id=__version__,
            ),
            idempotency_key=idempotency_key,
            request_sha256=fingerprint,
            display_name=intake.display_name,
            launch_source=_launch_source(http_request, identity, command="regimex submit-training"),
        )
        response.headers["Location"] = f"/api/v1/training-runs/{record.run_id}"
        return _run(record)

    @app.post(
        "/api/v1/evaluations", status_code=202, openapi_extra=_request_body(EvaluationRequest)
    )
    def evaluate(
        identity: Identity,
        intake: EvaluationBody,
        http_request: Request,
        response: Response,
        idempotency_key: Key,
    ) -> AdminRunView:
        record = service.submit_evaluation(
            identity,
            EvaluationSpec(**intake.model_dump()),
            idempotency_key=idempotency_key,
            launch_source=_launch_source(http_request, identity),
        )
        response.headers["Location"] = f"/api/v1/evaluations/{record.run_id}"
        return _run(record)

    @app.post(
        "/api/v1/models/publish", status_code=202, openapi_extra=_request_body(PublicationRequest)
    )
    @app.post(
        "/api/v1/models/{model_version_id}/publish",
        status_code=202,
        openapi_extra=_request_body(PublicationRequest),
    )
    def publish(
        model_version_id: ImmutableModelVersion,
        identity: Identity,
        intake: PublicationBody,
        http_request: Request,
        response: Response,
        idempotency_key: Key,
    ) -> AdminRunView:
        spec = PublicationSpec(model_version_id=model_version_id, **intake.model_dump())
        record = service.submit_publication(
            identity,
            spec,
            idempotency_key=idempotency_key,
            launch_source=_launch_source(http_request, identity),
        )
        response.headers["Location"] = f"/api/v1/admin-runs/{record.run_id}"
        return _run(record)

    @app.get("/api/v1/admin-runs/{run_id}")
    def get_run(run_id: UUID, identity: Identity) -> AdminRunView:
        return _run(service.get_run(identity, run_id))

    def checked_run(identity: Principal, run_id: UUID, kind: AdminKind) -> AdminRunView:
        record = service.get_run(identity, run_id)
        if record.operation.kind != kind:
            raise NotFound()
        return _run(record)

    @app.patch(
        "/api/v1/training-runs/archive",
        status_code=204,
        openapi_extra=_request_body(TrainingArchiveUpdate),
    )
    def set_training_archive(identity: Identity, edit: ArchiveBody) -> Response:
        service.set_training_archive(identity, edit.run_ids, archived=edit.archived)
        return Response(status_code=204)

    @app.get("/api/v1/training-runs/{run_id}")
    def get_training(run_id: UUID, identity: Identity) -> AdminRunView:
        return checked_run(identity, run_id, "training")

    @app.patch(
        "/api/v1/training-runs/{run_id}/name", openapi_extra=_request_body(DisplayNameUpdate)
    )
    def rename_training_run(run_id: UUID, identity: Identity, edit: NameBody) -> TrainingRunName:
        return service.rename_training_run(identity, run_id, edit)

    @app.get("/api/v1/training-runs/{run_id}/metrics")
    def training_metrics(run_id: UUID, identity: Identity) -> TrainingMetrics:
        return service.training_metrics(identity, run_id)

    @app.get("/api/v1/training-runs/{run_id}/validation-score")
    def training_validation_score(run_id: UUID, identity: Identity) -> TrainingValidationScore:
        metrics = service.training_metrics(identity, run_id)
        return TrainingValidationScore(
            model_version_id=metrics.model_version_id,
            validation_segment_macro_f1=metrics.validation_segment_macro_f1,
        )

    @app.post("/api/v1/training-runs/{run_id}/weights/deletion-preview")
    def preview_weight_deletion(run_id: UUID, identity: Identity) -> DeletionPreview:
        return service.preview_weight_deletion(identity, run_id)

    @app.get("/api/v1/training-runs/{run_id}/progress")
    def training_progress(
        run_id: UUID,
        identity: Identity,
        attempt: Annotated[int | None, Query(ge=0, le=10)] = None,
        after: Annotated[int, Query(ge=0, le=MAX_PROGRESS_POINTS)] = 0,
    ) -> TrainingProgress:
        return service.training_progress(identity, run_id, attempt=attempt, after=after)

    @app.get("/api/v1/training-runs/{run_id}/actions")
    def training_actions(
        run_id: UUID,
        identity: Identity,
        include_final: bool = False,
        limit: Limit = 100,
        offset: Offset = 0,
    ) -> AdminRunPage:
        records = service.training_actions(
            identity, run_id, include_final=include_final, limit=limit, offset=offset
        )
        return AdminRunPage(
            items=tuple(_run(item, include_final=include_final) for item in records),
            limit=limit,
            offset=offset,
        )

    @app.get("/api/v1/evaluations/{run_id}")
    def get_evaluation(run_id: UUID, identity: Identity) -> AdminRunView:
        return checked_run(identity, run_id, "evaluation")

    @app.get("/api/v1/evaluations/{run_id}/result")
    def evaluation_report(run_id: UUID, identity: Identity) -> dict[str, JsonValue]:
        return service.read_evaluation_report(identity, run_id)

    @app.get("/api/v1/training-runs")
    def list_training(
        identity: Identity,
        archive: TrainingArchiveFilter = "all",
        limit: Limit = 100,
        offset: Offset = 0,
    ) -> AdminRunPage:
        runs = service.list_runs(
            identity, kind="training", archive=archive, limit=limit, offset=offset
        )
        return AdminRunPage(
            items=tuple(_run(item) for item in runs),
            limit=limit,
            offset=offset,
        )

    @app.get("/api/v1/evaluations")
    def list_evaluations(
        identity: Identity, limit: Limit = 100, offset: Offset = 0
    ) -> AdminRunPage:
        return AdminRunPage(
            items=tuple(
                _run(item)
                for item in service.list_runs(
                    identity, kind="evaluation", limit=limit, offset=offset
                )
            ),
            limit=limit,
            offset=offset,
        )

    @app.get("/api/v1/metrics")
    def metrics(
        identity: Identity, model_version_id: RegistryIdentifier = "active"
    ) -> dict[str, JsonValue]:
        return service.get_metrics(identity, model_version_id)
