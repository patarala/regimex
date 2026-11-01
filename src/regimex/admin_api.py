"""Administrative HTTP workflows over explicit application operations."""

import hashlib
from collections.abc import Callable
from typing import Annotated, Literal, Protocol
from uuid import UUID, uuid4

from fastapi import Depends, FastAPI, Header, Query, Response
from pydantic import AwareDatetime, ConfigDict, Field, JsonValue, RootModel

from regimex import __version__
from regimex.admin_contracts import (
    AdminKind,
    AdminRunRecord,
    DatasetRecord,
    DatasetSelection,
    DatasetSpec,
    EvaluationRecord,
    EvaluationSpec,
    FinalReferenceImportSpec,
    GenerationSpec,
    HumanAnnotation,
    PublicationSpec,
    RawGenerationPlan,
    ReferenceGenerationSpec,
    ReferenceImportSpec,
    ReferenceSetRecord,
    SplitPolicy,
    TrainingSpec,
)
from regimex.api import OperationView, _json_dependency, _operation, _request_body
from regimex.contracts import Identifier, StrictModel
from regimex.data_catalog import DatasetManifest, ReferenceSet
from regimex.directions import AnalysisConfig, DirectionAnnotations
from regimex.model_registry import ArchitectureId, InferenceRuntime, TrainingPolicy, TrainingRuntime
from regimex.service_contracts import NotFound, Principal
from regimex.storage import json_bytes


class AdminAPIService(Protocol):
    def import_final_references(
        self,
        principal: Principal,
        spec: FinalReferenceImportSpec,
        *,
        idempotency_key: str | None = None,
        request_sha256: str | None = None,
    ) -> ReferenceSetRecord: ...

    def submit_final_evaluation(
        self,
        principal: Principal,
        benchmark_id: str,
        model_version_id: str,
        *,
        idempotency_key: str,
    ) -> AdminRunRecord: ...

    def submit_generation(
        self,
        principal: Principal,
        spec: GenerationSpec,
        *,
        idempotency_key: str,
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
        request_sha256: str | None = None,
    ) -> AdminRunRecord: ...

    def import_references(
        self,
        principal: Principal,
        spec: ReferenceImportSpec,
        *,
        idempotency_key: str | None = None,
        request_sha256: str | None = None,
    ) -> ReferenceSetRecord: ...

    def submit_training(
        self,
        principal: Principal,
        spec: TrainingSpec,
        *,
        idempotency_key: str,
        request_sha256: str | None = None,
    ) -> AdminRunRecord: ...

    def submit_evaluation(
        self,
        principal: Principal,
        spec: EvaluationSpec,
        *,
        idempotency_key: str,
        request_sha256: str | None = None,
    ) -> AdminRunRecord: ...

    def submit_publication(
        self,
        principal: Principal,
        spec: PublicationSpec,
        *,
        idempotency_key: str,
        request_sha256: str | None = None,
    ) -> AdminRunRecord: ...

    def get_run(self, principal: Principal, run_id: UUID) -> AdminRunRecord: ...

    def list_runs(
        self,
        principal: Principal,
        *,
        kind: AdminKind | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[AdminRunRecord, ...]: ...

    def get_dataset(self, principal: Principal, dataset_id: str) -> DatasetRecord: ...

    def list_datasets(
        self,
        principal: Principal,
        *,
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
    members: Annotated[tuple[DatasetSelection, ...], Field(min_length=3, max_length=1000)]


class DatasetRequest(
    RootModel[
        Annotated[GeneratedDatasetRequest | SnapshotDatasetRequest, Field(discriminator="mode")]
    ]
):
    model_config = ConfigDict(frozen=True, strict=True, allow_inf_nan=False)


class AutomaticReferenceRequest(StrictModel):
    mode: Literal["automatic"]
    dataset_id: Identifier
    partition: Literal["train", "validation", "test"]
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    analysis_profile_id: Identifier = "direction-32/2.0"


class HumanReferenceRequest(StrictModel):
    mode: Literal["human_import"]
    dataset_id: Identifier
    partition: Literal["train", "validation"]
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    analysis_profile_id: Identifier = "direction-32/2.0"
    reviewer: Identifier
    blind_review: bool
    independent: bool
    annotations: Annotated[tuple[HumanAnnotation, ...], Field(min_length=1, max_length=1000)]


class ReferenceRequest(
    RootModel[
        Annotated[AutomaticReferenceRequest | HumanReferenceRequest, Field(discriminator="mode")]
    ]
):
    model_config = ConfigDict(frozen=True, strict=True, allow_inf_nan=False)


class TrainingRunRequest(StrictModel):
    dataset_id: Identifier
    labels_reference_set_id: Identifier
    validation_reference_set_id: Identifier
    architecture_id: ArchitectureId
    parameters: dict[str, JsonValue]
    policy: TrainingPolicy
    training_runtime: TrainingRuntime = Field(default_factory=TrainingRuntime)
    inference_runtime: InferenceRuntime = Field(default_factory=InferenceRuntime)
    analysis_profile_id: Identifier = "direction-32/2.0"
    benchmark_selection_id: Identifier


class PublicationRequest(StrictModel):
    evaluation_run_id: UUID
    expected_previous: Identifier | None = None


class EvaluationRequest(StrictModel):
    model_version_id: Identifier
    dataset_id: Identifier
    reference_set_id: Identifier
    partition: Literal["validation"] = "validation"


class AdminRunView(StrictModel):
    run_id: UUID
    job: OperationView
    output: dict[str, JsonValue] | None
    polling_url: str


class AdminRunPage(StrictModel):
    items: tuple[AdminRunView, ...]
    limit: int
    offset: int


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
    manifest_url: str


class ReferencePage(StrictModel):
    items: tuple[ReferenceView, ...]
    limit: int
    offset: int


def _hash_request(value: StrictModel | DatasetRequest | ReferenceRequest) -> str:
    return hashlib.sha256(json_bytes(value.model_dump(mode="json"))).hexdigest()


def _run(record: AdminRunRecord) -> AdminRunView:
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
    return AdminRunView(
        run_id=record.run_id,
        job=_operation(record.operation),
        output=None
        if record.output is None
        else {name: value for name, value in record.output.items() if name in allowed},
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
    EvaluationBody = Annotated[EvaluationRequest, Depends(_json_dependency(EvaluationRequest))]
    PublicationBody = Annotated[PublicationRequest, Depends(_json_dependency(PublicationRequest))]
    app.state.request_models += (
        DatasetRequest,
        ReferenceRequest,
        TrainingRunRequest,
        EvaluationRequest,
        PublicationRequest,
    )

    @app.post("/api/v1/datasets", openapi_extra=_request_body(DatasetRequest))
    def create_dataset(
        identity: Identity,
        intake: DatasetBody,
        response: Response,
        idempotency_key: Key,
    ) -> AdminRunView | DatasetView:
        request = intake.root
        digest = _hash_request(intake)
        if isinstance(request, GeneratedDatasetRequest):
            record = service.submit_generation(
                identity,
                GenerationSpec(
                    dataset_id=f"dataset-{uuid4().hex}", plan=request.plan, splits=request.splits
                ),
                idempotency_key=idempotency_key,
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
        response.status_code = 201
        response.headers["Location"] = f"/api/v1/datasets/{dataset.dataset_id}"
        return _dataset(dataset)

    @app.get("/api/v1/datasets")
    def list_datasets(identity: Identity, limit: Limit = 100, offset: Offset = 0) -> DatasetPage:
        return DatasetPage(
            items=tuple(
                _dataset(item)
                for item in service.list_datasets(identity, limit=limit, offset=offset)
            ),
            limit=limit,
            offset=offset,
        )

    @app.get("/api/v1/datasets/{dataset_id}")
    def get_dataset(dataset_id: Identifier, identity: Identity) -> DatasetView:
        return _dataset(service.get_dataset(identity, dataset_id))

    @app.get("/api/v1/datasets/{dataset_id}/manifest")
    def dataset_manifest(dataset_id: Identifier, identity: Identity) -> DatasetManifest:
        return service.get_dataset(identity, dataset_id).manifest

    @app.post("/api/v1/reference-sets", openapi_extra=_request_body(ReferenceRequest))
    def create_references(
        identity: Identity,
        intake: ReferenceBody,
        response: Response,
        idempotency_key: Key,
    ) -> AdminRunView | ReferenceView:
        request = intake.root
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
                request_sha256=_hash_request(intake),
            )
            response.status_code = 202
            response.headers["Location"] = f"/api/v1/admin-runs/{record.run_id}"
            return _run(record)
        reference_id = (
            "reference-"
            + hashlib.sha256(
                json_bytes({"owner_id": identity.owner_id, "request": _hash_request(intake)})
            ).hexdigest()[:40]
        )
        references = service.import_references(
            identity,
            ReferenceImportSpec(
                **request.model_dump(exclude={"mode"}), reference_set_id=reference_id
            ),
            idempotency_key=idempotency_key,
            request_sha256=_hash_request(intake),
        )
        response.status_code = 201
        response.headers["Location"] = (
            f"/api/v1/reference-sets/{references.references.reference_set_id}"
        )
        return _reference(references)

    @app.get("/api/v1/reference-sets")
    def list_references(
        identity: Identity, limit: Limit = 100, offset: Offset = 0
    ) -> ReferencePage:
        return ReferencePage(
            items=tuple(
                _reference(item)
                for item in service.list_reference_sets(identity, limit=limit, offset=offset)
            ),
            limit=limit,
            offset=offset,
        )

    @app.get("/api/v1/reference-sets/{reference_set_id}")
    def get_references(reference_set_id: Identifier, identity: Identity) -> ReferenceView:
        return _reference(service.get_reference_set(identity, reference_set_id))

    @app.get("/api/v1/reference-sets/{reference_set_id}/manifest")
    def reference_manifest(reference_set_id: Identifier, identity: Identity) -> ReferenceSet:
        return service.get_reference_set(identity, reference_set_id).references

    @app.get("/api/v1/reference-sets/{reference_set_id}/annotations")
    def reference_annotations(
        reference_set_id: Identifier, identity: Identity
    ) -> tuple[DirectionAnnotations, ...]:
        return service.get_reference_annotations(identity, reference_set_id)

    @app.post(
        "/api/v1/training-runs", status_code=202, openapi_extra=_request_body(TrainingRunRequest)
    )
    def train(
        identity: Identity, intake: TrainingBody, response: Response, idempotency_key: Key
    ) -> AdminRunView:
        record = service.submit_training(
            identity,
            TrainingSpec(
                **intake.model_dump(exclude={"benchmark_selection_id"}),
                model_version_id=f"model-{uuid4().hex}",
                build_id=__version__,
                benchmark_selection_id=intake.benchmark_selection_id,
            ),
            idempotency_key=idempotency_key,
            request_sha256=_hash_request(intake),
        )
        response.headers["Location"] = f"/api/v1/training-runs/{record.run_id}"
        return _run(record)

    @app.post(
        "/api/v1/evaluations", status_code=202, openapi_extra=_request_body(EvaluationRequest)
    )
    def evaluate(
        identity: Identity, intake: EvaluationBody, response: Response, idempotency_key: Key
    ) -> AdminRunView:
        record = service.submit_evaluation(
            identity, EvaluationSpec(**intake.model_dump()), idempotency_key=idempotency_key
        )
        response.headers["Location"] = f"/api/v1/evaluations/{record.run_id}"
        return _run(record)

    @app.post(
        "/api/v1/models/{model_version_id}/publish",
        status_code=202,
        openapi_extra=_request_body(PublicationRequest),
    )
    def publish(
        model_version_id: Identifier,
        identity: Identity,
        intake: PublicationBody,
        response: Response,
        idempotency_key: Key,
    ) -> AdminRunView:
        spec = PublicationSpec(model_version_id=model_version_id, **intake.model_dump())
        record = service.submit_publication(identity, spec, idempotency_key=idempotency_key)
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

    @app.get("/api/v1/training-runs/{run_id}")
    def get_training(run_id: UUID, identity: Identity) -> AdminRunView:
        return checked_run(identity, run_id, "training")

    @app.get("/api/v1/evaluations/{run_id}")
    def get_evaluation(run_id: UUID, identity: Identity) -> AdminRunView:
        return checked_run(identity, run_id, "evaluation")

    @app.get("/api/v1/evaluations/{run_id}/result")
    def evaluation_report(run_id: UUID, identity: Identity) -> dict[str, JsonValue]:
        return service.read_evaluation_report(identity, run_id)

    @app.get("/api/v1/training-runs")
    def list_training(identity: Identity, limit: Limit = 100, offset: Offset = 0) -> AdminRunPage:
        return AdminRunPage(
            items=tuple(
                _run(item)
                for item in service.list_runs(identity, kind="training", limit=limit, offset=offset)
            ),
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
        identity: Identity, model_version_id: Identifier = "active"
    ) -> dict[str, JsonValue]:
        return service.get_metrics(identity, model_version_id)
