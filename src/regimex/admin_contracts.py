"""Lightweight administrative requests and persisted resource metadata."""

from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import AwareDatetime, Field, JsonValue, model_validator

from regimex.contracts import MAX_STEPS, Identifier, Seed, StrictModel
from regimex.data_catalog import DatasetManifest, Partition, ReferenceSet, SplitGroup
from regimex.directions import AnalysisConfig, DirectionAnnotations
from regimex.infrastructure.object_store import ObjectInfo
from regimex.market_config import MarketConfig
from regimex.model_registry import (
    ArchitectureId,
    InferenceRuntime,
    TrainingPolicy,
    TrainingRuntime,
)
from regimex.operations import Operation, Sha256


class RawGenerationPlan(StrictModel):
    """Raw market generation has no annotation or model configuration."""

    seed: Seed
    experiment_count: Annotated[int, Field(ge=3, le=1000)]
    step_count: Annotated[int, Field(ge=64, le=MAX_STEPS)]
    market: MarketConfig | None = None


class SplitPolicy(StrictModel):
    seed: Seed = 1729
    train_fraction: Annotated[float, Field(gt=0, lt=1)] = 0.7
    validation_fraction: Annotated[float, Field(gt=0, lt=1)] = 0.15

    @model_validator(mode="after")
    def leave_test_partition(self) -> Self:
        if self.train_fraction + self.validation_fraction >= 1:
            raise ValueError("train and validation fractions must leave a test partition")
        return self


class GenerationSpec(StrictModel):
    kind: Literal["generation"] = "generation"
    dataset_id: Identifier
    plan: RawGenerationPlan
    splits: SplitPolicy = Field(default_factory=SplitPolicy)


class CatalogBinding(StrictModel):
    """Stable source/group declarations admitted by an administrator or generator."""

    snapshot_id: UUID
    source_namespace: Identifier
    split_group: SplitGroup


class DatasetSelection(StrictModel):
    snapshot_id: UUID
    partition: Partition


class DatasetSpec(StrictModel):
    dataset_id: Identifier
    members: Annotated[tuple[DatasetSelection, ...], Field(min_length=3, max_length=10000)]

    @model_validator(mode="after")
    def unique_snapshots(self) -> Self:
        if len({member.snapshot_id for member in self.members}) != len(self.members):
            raise ValueError("dataset snapshot selections must be unique")
        return self


class DatasetRecord(StrictModel):
    dataset_id: Identifier
    owner_id: Identifier
    manifest: DatasetManifest
    snapshot_ids: Annotated[tuple[UUID, ...], Field(min_length=3, max_length=10000)]
    created_at: AwareDatetime
    generation: GenerationSpec | None = None
    generation_provenance: ObjectInfo | None = None

    @model_validator(mode="after")
    def ordered_members(self) -> Self:
        if self.dataset_id != self.manifest.dataset_id or len(self.snapshot_ids) != len(
            self.manifest.members
        ):
            raise ValueError(
                "dataset identity and ordered stored snapshots must match its manifest"
            )
        if len(set(self.snapshot_ids)) != len(self.snapshot_ids):
            raise ValueError("dataset stored snapshots must be unique")
        if (self.generation is None) != (self.generation_provenance is None) or (
            self.generation is not None and self.generation.dataset_id != self.dataset_id
        ):
            raise ValueError("generated datasets require matching generation and provenance")
        return self


class AnnotationObject(StrictModel):
    source_namespace: Identifier
    snapshot_id: Identifier
    object: ObjectInfo


class ReferenceSetRecord(StrictModel):
    dataset_id: Identifier
    owner_id: Identifier
    references: ReferenceSet
    annotations: Annotated[tuple[AnnotationObject, ...], Field(min_length=1, max_length=10000)]
    created_at: AwareDatetime

    @model_validator(mode="after")
    def complete_annotation_objects(self) -> Self:
        expected = tuple(
            (member.source_namespace, member.snapshot_id, member.annotations_sha256)
            for member in self.references.members
        )
        actual = tuple(
            (member.source_namespace, member.snapshot_id, member.object.sha256)
            for member in self.annotations
        )
        if actual != expected:
            raise ValueError(
                "annotation objects must match ordered reference membership and hashes"
            )
        return self


class ReferenceGenerationSpec(StrictModel):
    kind: Literal["reference"] = "reference"
    dataset_id: Identifier
    reference_set_id: Identifier
    partition: Partition
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    analysis_profile_id: Identifier = "direction-32/2.0"


class HumanAnnotation(StrictModel):
    snapshot_id: UUID
    annotation: DirectionAnnotations


class HumanReferenceImportBase(StrictModel):
    dataset_id: Identifier
    reference_set_id: Identifier
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    analysis_profile_id: Identifier = "direction-32/2.0"
    reviewer: Identifier
    blind_review: bool
    independent: bool
    annotations: Annotated[tuple[HumanAnnotation, ...], Field(min_length=1, max_length=1000)]

    @model_validator(mode="after")
    def unique_annotations(self) -> Self:
        if len({item.snapshot_id for item in self.annotations}) != len(self.annotations):
            raise ValueError("human annotation snapshots must be unique")
        return self


class ReferenceImportSpec(HumanReferenceImportBase):
    partition: Literal["train", "validation"]


class FinalReferenceImportSpec(HumanReferenceImportBase):
    partition: Literal["test"] = "test"


class TrainingSpec(StrictModel):
    kind: Literal["training"] = "training"
    dataset_id: Identifier
    labels_reference_set_id: Identifier
    validation_reference_set_id: Identifier
    model_version_id: Identifier
    architecture_id: ArchitectureId
    parameters: dict[str, JsonValue]
    policy: TrainingPolicy
    training_runtime: TrainingRuntime
    inference_runtime: InferenceRuntime
    analysis_profile_id: Identifier
    build_id: Identifier
    benchmark_selection_id: Identifier | None = None

    @model_validator(mode="after")
    def immutable_model_version(self) -> Self:
        if self.model_version_id == "active":
            raise ValueError("the active alias cannot identify a new immutable model")
        return self


class EvaluationSpec(StrictModel):
    kind: Literal["evaluation"] = "evaluation"
    model_version_id: Identifier
    dataset_id: Identifier
    reference_set_id: Identifier
    partition: Literal["validation", "test"] = "validation"
    benchmark_id: Identifier | None = None

    @model_validator(mode="after")
    def fixed_model(self) -> Self:
        if self.model_version_id == "active":
            raise ValueError("evaluation requires a resolved immutable model version")
        if (self.partition == "test") != (self.benchmark_id is not None):
            raise ValueError("test evaluation requires an explicit benchmark; validation has none")
        return self


class PublicationSpec(StrictModel):
    kind: Literal["publication"] = "publication"
    model_version_id: Identifier
    evaluation_run_id: UUID
    expected_previous: Identifier | None = None

    @model_validator(mode="after")
    def immutable_model_version(self) -> Self:
        if self.model_version_id == "active":
            raise ValueError("publication requires a resolved immutable model version")
        return self


AdminSpec = Annotated[
    GenerationSpec | ReferenceGenerationSpec | TrainingSpec | EvaluationSpec | PublicationSpec,
    Field(discriminator="kind"),
]
AdminKind = Literal["generation", "reference", "training", "evaluation", "publication"]


class AdminRunRecord(StrictModel):
    run_id: UUID
    operation: Operation
    spec: AdminSpec
    output: dict[str, JsonValue] | None = None

    @model_validator(mode="after")
    def same_operation(self) -> Self:
        if self.run_id != self.operation.operation_id or self.operation.kind != self.spec.kind:
            raise ValueError("administrative run identity and kind must match its operation")
        if (self.operation.state == "succeeded") != (self.output is not None):
            raise ValueError("only a successful administrative run has committed output")
        return self


class EvaluationRecord(StrictModel):
    run_id: UUID
    owner_id: Identifier
    model_version_id: Identifier
    bundle_sha256: Sha256
    inference_policy_sha256: Sha256
    dataset_id: Identifier
    dataset_sha256: Sha256
    reference_set_id: Identifier
    reference_sha256: Sha256
    partition: Literal["validation", "test"]
    report: ObjectInfo
    metrics: dict[str, JsonValue]
    quality_status: Literal["validation_only"] = "validation_only"
    created_at: AwareDatetime
