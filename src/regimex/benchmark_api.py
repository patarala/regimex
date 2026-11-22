"""Explicit administrative transitions for final-reference access."""

import hashlib
from collections.abc import Callable
from typing import Annotated, Literal, Protocol, Self
from urllib.parse import urlencode

from fastapi import Depends, FastAPI, Header, Response
from pydantic import Field, model_validator

from regimex.admin_api import AdminRunView, ReferenceView, _reference, _run
from regimex.admin_contracts import (
    AdminRunRecord,
    FinalReferenceImportSpec,
    HumanAnnotation,
    ReferenceSetRecord,
    RequestDigest,
    RequestFingerprint,
)
from regimex.api import _json_dependency, _request_body
from regimex.benchmark_contracts import (
    BenchmarkView,
    RegistryState,
    TrainingInventorySummary,
    TrainingPreparation,
)
from regimex.benchmarks import DatasetHistory
from regimex.contracts import Identifier, StrictModel
from regimex.directions import DEFAULT_ANALYSIS_PROFILE, AnalysisConfig
from regimex.service_contracts import Principal
from regimex.storage import json_bytes


class BenchmarkAPIService(Protocol):
    def get_registry(self, principal: Principal) -> RegistryState: ...

    def selection_benchmarks(self, principal: Principal, dataset_id: str) -> tuple[str, ...]: ...

    def preparation_inventory(
        self, principal: Principal, dataset_id: str, final_dataset_id: str
    ) -> TrainingInventorySummary: ...

    def prepare_selection(
        self, principal: Principal, request: TrainingPreparation
    ) -> BenchmarkView: ...

    def get_benchmark(self, principal: Principal, benchmark_id: str) -> BenchmarkView: ...

    def register_dataset_history(self, principal: Principal, dataset_id: str) -> RegistryState: ...

    def import_history(self, principal: Principal, history: DatasetHistory) -> RegistryState: ...

    def attest(
        self, principal: Principal, source_namespaces: tuple[str, ...], statement: str
    ) -> RegistryState: ...

    def reserve(
        self, principal: Principal, benchmark_id: str, dataset_id: str, reference_set_id: str
    ) -> BenchmarkView: ...

    def start_selection(self, principal: Principal, benchmark_id: str) -> BenchmarkView: ...

    def freeze(
        self, principal: Principal, benchmark_id: str, model_version_id: str
    ) -> BenchmarkView: ...

    def open(self, principal: Principal, benchmark_id: str) -> BenchmarkView: ...


class FinalEvaluationService(Protocol):
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
    ) -> AdminRunRecord: ...


class DatasetHistoryRequest(StrictModel):
    dataset_id: Identifier


class SelectionBenchmarks(StrictModel):
    benchmark_ids: tuple[Identifier, ...]
    analysis_profile_id: Identifier = DEFAULT_ANALYSIS_PROFILE


class PreparedSelection(StrictModel):
    benchmark_id: Identifier


class FinalReferenceRequest(StrictModel):
    """Human final annotations enter through a separate sealed access boundary."""

    dataset_id: Identifier
    partition: Literal["test"] = "test"
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    analysis_profile_id: Identifier = DEFAULT_ANALYSIS_PROFILE
    reviewer: Identifier
    blind_review: bool
    independent: bool
    annotations: Annotated[tuple[HumanAnnotation, ...], Field(min_length=1, max_length=1000)]


class InventoryRequest(StrictModel):
    source_namespaces: Annotated[tuple[Identifier, ...], Field(min_length=1, max_length=10000)]
    statement: Annotated[str, Field(min_length=1, max_length=4096)]


RegistryIdentifier = Annotated[Identifier, Field(pattern=r"^[^\x00]+$")]


class BenchmarkRequest(StrictModel):
    benchmark_id: RegistryIdentifier
    dataset_id: Identifier
    reference_set_id: Identifier


class FrozenModelRequest(StrictModel):
    model_version_id: Identifier

    @model_validator(mode="after")
    def immutable_model(self) -> Self:
        if self.model_version_id == "active":
            raise ValueError("frozen choices require a concrete model version")
        return self


def install_benchmark_routes(
    app: FastAPI,
    service: BenchmarkAPIService,
    administration: FinalEvaluationService,
    *,
    authenticate: Callable[..., Principal],
) -> None:
    Identity = Annotated[Principal, Depends(authenticate)]
    Key = Annotated[Identifier, Header(alias="Idempotency-Key")]
    DatasetBody = Annotated[DatasetHistoryRequest, Depends(_json_dependency(DatasetHistoryRequest))]
    HistoryBody = Annotated[DatasetHistory, Depends(_json_dependency(DatasetHistory))]
    InventoryBody = Annotated[InventoryRequest, Depends(_json_dependency(InventoryRequest))]
    ReserveBody = Annotated[BenchmarkRequest, Depends(_json_dependency(BenchmarkRequest))]
    PreparationBody = Annotated[TrainingPreparation, Depends(_json_dependency(TrainingPreparation))]
    ModelBody = Annotated[FrozenModelRequest, Depends(_json_dependency(FrozenModelRequest))]
    ReferenceBody = Annotated[
        FinalReferenceRequest, Depends(_json_dependency(FinalReferenceRequest))
    ]
    app.state.request_models += (
        DatasetHistoryRequest,
        DatasetHistory,
        InventoryRequest,
        BenchmarkRequest,
        TrainingPreparation,
        FrozenModelRequest,
        FinalReferenceRequest,
    )

    @app.get("/api/v1/benchmarks/registry")
    def registry(identity: Identity) -> RegistryState:
        return service.get_registry(identity)

    @app.get("/api/v1/training-options/benchmarks")
    def selection_benchmarks(
        identity: Identity, dataset_id: RegistryIdentifier
    ) -> SelectionBenchmarks:
        return SelectionBenchmarks(benchmark_ids=service.selection_benchmarks(identity, dataset_id))

    @app.get("/api/v1/training-options/inventory")
    def inventory_summary(
        identity: Identity, dataset_id: RegistryIdentifier, final_dataset_id: RegistryIdentifier
    ) -> TrainingInventorySummary:
        return service.preparation_inventory(identity, dataset_id, final_dataset_id)

    @app.post(
        "/api/v1/training-options/selection", openapi_extra=_request_body(TrainingPreparation)
    )
    def prepare_selection(identity: Identity, intake: PreparationBody) -> PreparedSelection:
        view = service.prepare_selection(identity, intake)
        return PreparedSelection(benchmark_id=view.binding.benchmark_id)

    @app.post(
        "/api/v1/benchmarks/reference-sets",
        status_code=201,
        openapi_extra=_request_body(FinalReferenceRequest),
    )
    def import_final_references(
        identity: Identity,
        intake: ReferenceBody,
        response: Response,
        idempotency_key: Key,
    ) -> ReferenceView:
        fingerprint = RequestFingerprint.for_profile_request(intake)
        reference_id = (
            "reference-"
            + hashlib.sha256(
                json_bytes({"owner_id": identity.owner_id, "request": fingerprint.sha256})
            ).hexdigest()[:40]
        )
        record = administration.import_final_references(
            identity,
            FinalReferenceImportSpec(**intake.model_dump(), reference_set_id=reference_id),
            idempotency_key=idempotency_key,
            request_sha256=fingerprint,
        )
        response.headers["Location"] = (
            f"/api/v1/reference-sets/{record.references.reference_set_id}"
        )
        return _reference(record)

    @app.post(
        "/api/v1/benchmarks/history/datasets", openapi_extra=_request_body(DatasetHistoryRequest)
    )
    def register_dataset(identity: Identity, intake: DatasetBody) -> RegistryState:
        return service.register_dataset_history(identity, intake.dataset_id)

    @app.post("/api/v1/benchmarks/history/import", openapi_extra=_request_body(DatasetHistory))
    def import_history(identity: Identity, intake: HistoryBody) -> RegistryState:
        return service.import_history(identity, intake)

    @app.post("/api/v1/benchmarks/attest", openapi_extra=_request_body(InventoryRequest))
    def attest(identity: Identity, intake: InventoryBody) -> RegistryState:
        return service.attest(identity, intake.source_namespaces, intake.statement)

    @app.post("/api/v1/benchmarks", status_code=201, openapi_extra=_request_body(BenchmarkRequest))
    def reserve(identity: Identity, intake: ReserveBody, response: Response) -> BenchmarkView:
        benchmark = service.reserve(
            identity, intake.benchmark_id, intake.dataset_id, intake.reference_set_id
        )
        query = urlencode({"benchmark_id": intake.benchmark_id})
        response.headers["Location"] = f"/api/v1/benchmarks?{query}"
        return benchmark

    @app.get("/api/v1/benchmarks")
    @app.get("/api/v1/benchmarks/{benchmark_id}")
    def get_benchmark(benchmark_id: RegistryIdentifier, identity: Identity) -> BenchmarkView:
        return service.get_benchmark(identity, benchmark_id)

    @app.post("/api/v1/benchmarks/selection-start")
    @app.post("/api/v1/benchmarks/{benchmark_id}/selection-start")
    def selection_start(benchmark_id: RegistryIdentifier, identity: Identity) -> BenchmarkView:
        return service.start_selection(identity, benchmark_id)

    @app.post("/api/v1/benchmarks/freeze", openapi_extra=_request_body(FrozenModelRequest))
    @app.post(
        "/api/v1/benchmarks/{benchmark_id}/freeze", openapi_extra=_request_body(FrozenModelRequest)
    )
    def freeze(
        benchmark_id: RegistryIdentifier, identity: Identity, intake: ModelBody
    ) -> BenchmarkView:
        return service.freeze(identity, benchmark_id, intake.model_version_id)

    @app.post("/api/v1/benchmarks/open")
    @app.post("/api/v1/benchmarks/{benchmark_id}/open")
    def open_references(benchmark_id: RegistryIdentifier, identity: Identity) -> BenchmarkView:
        return service.open(identity, benchmark_id)

    @app.post(
        "/api/v1/benchmarks/evaluations",
        status_code=202,
        openapi_extra=_request_body(FrozenModelRequest),
    )
    @app.post(
        "/api/v1/benchmarks/{benchmark_id}/evaluations",
        status_code=202,
        openapi_extra=_request_body(FrozenModelRequest),
    )
    def evaluate_final(
        benchmark_id: RegistryIdentifier,
        identity: Identity,
        intake: ModelBody,
        idempotency_key: Key,
        response: Response,
    ) -> AdminRunView:
        record = administration.submit_final_evaluation(
            identity, benchmark_id, intake.model_version_id, idempotency_key=idempotency_key
        )
        response.headers["Location"] = f"/api/v1/evaluations/{record.run_id}"
        return _run(record)
