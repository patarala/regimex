"""Lightweight administrative requests and persisted resource metadata."""

from typing import Annotated, ClassVar, Literal, Self
from uuid import UUID

from pydantic import (
    AfterValidator,
    AwareDatetime,
    Field,
    JsonValue,
    RootModel,
    ValidationError,
    model_validator,
)

from regimex.contracts import MAX_STEPS, Identifier, Seed, StrictModel
from regimex.data_catalog import DatasetManifest, Partition, ReferenceSet, SplitGroup, content_hash
from regimex.direction_quality import DirectionQualityEvidence
from regimex.directions import (
    DEFAULT_ANALYSIS_PROFILE,
    PREVIOUS_DIRECTION_POLICY,
    AnalysisConfig,
    AutomaticPolicy,
    DirectionAnnotations,
)
from regimex.external_evaluations import ExternalEvaluation
from regimex.external_latency import ExternalLatencyEvidence
from regimex.history_contracts import DisplayName
from regimex.infrastructure.object_store import ObjectInfo
from regimex.market_config import GeneratorVersion, MarketConfig
from regimex.model_bundles import BundleMember, RepeatOutcome, TrainingEvidence
from regimex.model_registry import (
    SEGMENT_MOE_ID,
    ArchitectureId,
    FixedUpdatePolicy,
    InferenceRuntime,
    SingleFitPolicy,
    TrainingPolicySpec,
    TrainingRuntime,
    TreeCount,
    TreeDepth,
    canonical_architecture_id,
)
from regimex.operations import Operation, Sha256
from regimex.service_contracts import RegistryIdentifier
from regimex.storage import json_bytes


class RequestFingerprint(StrictModel):
    """Preserve the resolved request digest and recognize earlier profile defaults."""

    sha256: Sha256
    previous_hashes: tuple[Sha256, ...]

    @classmethod
    def for_profile_request(cls, request: StrictModel) -> Self:
        expanded = request.model_dump(mode="json")
        current = content_hash(expanded)
        # Reconstruct exactly what earlier servers persisted for omitted fields.
        previous = []
        analysis = getattr(request, "analysis", None)
        for version in ("2.1", "2.0"):
            if "analysis_profile_id" not in request.model_fields_set:
                expanded["analysis_profile_id"] = f"direction-32/{version}"
            if isinstance(analysis, AnalysisConfig) and "version" not in analysis.model_fields_set:
                expanded["analysis"]["version"] = version
            digest = content_hash(expanded)
            if digest != current and digest not in previous:
                previous.append(digest)
        return cls(sha256=current, previous_hashes=tuple(previous))


type RequestDigest = Sha256 | RequestFingerprint


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
    generator_version: GeneratorVersion | None = Field(
        default=None, exclude_if=lambda version: version is None
    )

    @property
    def saved_generator_version(self) -> GeneratorVersion:
        # Queued jobs written before version pinning used generator 4.0.
        return self.generator_version or "4.0"


class CatalogBinding(StrictModel):
    """Stable source/group declarations admitted by an administrator or generator."""

    snapshot_id: UUID
    source_namespace: Identifier
    split_group: SplitGroup


class DatasetSelection(StrictModel):
    snapshot_id: UUID
    partition: Partition
    split_group: SplitGroup | None = Field(default=None, exclude_if=lambda group: group is None)


def distinct_snapshot_selections(
    members: tuple[DatasetSelection, ...],
) -> tuple[DatasetSelection, ...]:
    if len({member.snapshot_id for member in members}) != len(members):
        raise ValueError("dataset snapshot selections must be unique")
    return members


class DatasetSpec(StrictModel):
    dataset_id: Identifier
    members: Annotated[
        tuple[DatasetSelection, ...],
        Field(min_length=3, max_length=10000),
        AfterValidator(distinct_snapshot_selections),
    ]


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
    analysis_profile_id: Identifier = DEFAULT_ANALYSIS_PROFILE
    policy: AutomaticPolicy | None = Field(default=None, exclude_if=lambda value: value is None)

    @property
    def saved_policy(self) -> AutomaticPolicy:
        return self.policy or PREVIOUS_DIRECTION_POLICY


class HumanAnnotation(StrictModel):
    snapshot_id: UUID
    annotation: DirectionAnnotations


def _unique_annotations(items: tuple[HumanAnnotation, ...]) -> tuple[HumanAnnotation, ...]:
    if len({item.snapshot_id for item in items}) != len(items):
        raise ValueError("human annotation snapshots must be unique")
    return items


HumanAnnotations = Annotated[
    tuple[HumanAnnotation, ...],
    Field(min_length=1, max_length=1000),
    AfterValidator(_unique_annotations),
]


class HumanReferenceImportBase(StrictModel):
    dataset_id: Identifier
    reference_set_id: Identifier
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    analysis_profile_id: Identifier = DEFAULT_ANALYSIS_PROFILE
    reviewer: Identifier
    blind_review: bool
    independent: bool
    annotations: HumanAnnotations


class ReferenceImportSpec(HumanReferenceImportBase):
    partition: Literal["train", "validation"]


class FinalReferenceImportSpec(HumanReferenceImportBase):
    partition: Literal["test"] = "test"


def _immutable_model_version(value: str) -> str:
    if value == "active":
        raise ValueError("Use an immutable model version ID.")
    return value


ImmutableModelVersion = Annotated[
    RegistryIdentifier,
    AfterValidator(_immutable_model_version),
    Field(json_schema_extra={"not": {"const": "active"}}),
]


class TrainingSpec(StrictModel):
    kind: Literal["training"] = "training"
    dataset_id: Identifier
    labels_reference_set_id: Identifier
    validation_reference_set_id: Identifier
    model_version_id: ImmutableModelVersion
    architecture_id: ArchitectureId
    parameters: dict[str, JsonValue]
    policy: TrainingPolicySpec
    training_runtime: TrainingRuntime
    inference_runtime: InferenceRuntime
    analysis_profile_id: Identifier
    build_id: Identifier
    benchmark_selection_id: Identifier | None = None


class EvaluationSpec(StrictModel):
    kind: Literal["evaluation"] = "evaluation"
    model_version_id: ImmutableModelVersion
    dataset_id: Identifier
    reference_set_id: Identifier
    partition: Literal["validation", "test"] = "validation"
    benchmark_id: Identifier | None = None

    @model_validator(mode="after")
    def benchmark_matches_partition(self) -> Self:
        if (self.partition == "test") != (self.benchmark_id is not None):
            raise ValueError("test evaluation requires an explicit benchmark; validation has none")
        return self


class ImportedTrainingSpec(StrictModel):
    """Recorded bundle provenance for a completed fit outside the service queue."""

    kind: Literal["training"] = "training"
    origin: Literal["external"]
    model_version_id: Identifier
    architecture_id: ArchitectureId
    implementation_version: Identifier
    build_id: Identifier
    analysis_profile_id: Identifier
    evidence: TrainingEvidence
    weights: BundleMember
    configuration: dict[str, JsonValue] | None
    inference_runtime: InferenceRuntime

    @property
    def recorded_policy(self) -> TrainingPolicySpec | None:
        settings = (self.configuration or {}).get("training_policy")
        if not isinstance(settings, dict):
            return None
        try:
            policy = RootModel[TrainingPolicySpec].model_validate_json(json_bytes(settings)).root
        except ValidationError:
            return None
        return policy if policy.seeds == self.evidence.seeds else None


class PublicationSpec(StrictModel):
    kind: Literal["publication"] = "publication"
    model_version_id: ImmutableModelVersion
    evaluation_run_id: UUID
    expected_previous: Identifier | None = None


AdminSpec = Annotated[
    GenerationSpec | ReferenceGenerationSpec | TrainingSpec | EvaluationSpec | PublicationSpec,
    Field(discriminator="kind"),
]
AdminKind = Literal["generation", "reference", "training", "evaluation", "publication"]
TrainingArchiveFilter = Literal["active", "archived", "all"]
MAX_ARCHIVE_SELECTION = 100


def admin_request_hash(spec: AdminSpec, display_name: str | None) -> str:
    request = spec.model_dump(mode="json")
    if display_name is not None:
        request["display_name"] = display_name
    return content_hash(request)


class TrainingRunName(StrictModel):
    run_id: UUID
    display_name: DisplayName | None
    revision: Annotated[int, Field(ge=0)]


class AdminRunRecord(StrictModel):
    run_id: UUID
    operation: Operation
    spec: AdminSpec | ImportedTrainingSpec
    output: dict[str, JsonValue] | None = None
    display_name: DisplayName | None = None
    name_revision: Annotated[int, Field(ge=0)] = 0
    weights_deletion_id: UUID | None = None
    archived_at: AwareDatetime | None = None
    external_evaluations: tuple[ExternalEvaluation, ...] = ()
    external_latency: ExternalLatencyEvidence | None = None

    @model_validator(mode="after")
    def same_operation(self) -> Self:
        if self.run_id != self.operation.operation_id or self.operation.kind != self.spec.kind:
            raise ValueError("administrative run identity and kind must match its operation")
        if (self.operation.state == "succeeded") != (self.output is not None):
            raise ValueError("only a successful administrative run has committed output")
        if isinstance(self.spec, ImportedTrainingSpec) != (self.operation.stage == "imported"):
            raise ValueError("imported training provenance requires a completed import")
        return self


def legacy_repeat_outcomes(
    spec: TrainingSpec | ImportedTrainingSpec, history: JsonValue
) -> tuple[RepeatOutcome, ...]:
    """Recover stopping reasons for verified histories written before backends recorded them."""
    if canonical_architecture_id(spec.architecture_id) not in ("tcn", SEGMENT_MOE_ID):
        return ()
    if not isinstance(history, dict):
        return ()
    policy = spec.recorded_policy if isinstance(spec, ImportedTrainingSpec) else spec.policy
    if policy is None or isinstance(policy, SingleFitPolicy):
        return ()
    unit = "update" if isinstance(policy, FixedUpdatePolicy) else "epoch"
    rows = history.get(unit + "s")
    if not isinstance(rows, list):
        return ()
    # The native codec verifies every seed's complete history before publication.
    # An epoch repeat that finishes below its budget can only have exhausted patience.
    last_step: dict[int, int] = {}
    for row in rows:
        if not isinstance(row, dict):
            return ()
        seed, step = row.get("seed"), row.get(unit)
        if type(seed) is not int or type(step) is not int:
            return ()
        last_step[seed] = step
    if tuple(last_step) != policy.seeds:
        return ()
    if isinstance(policy, FixedUpdatePolicy):
        return tuple(
            RepeatOutcome(seed=seed, step=step, reason="update_limit")
            for seed, step in last_step.items()
            if step == policy.updates
        )
    return tuple(
        RepeatOutcome(
            seed=seed,
            step=step,
            reason="epoch_limit" if step == policy.epochs else "early_stopping",
        )
        for seed, step in last_step.items()
    )


class ForestValidationPoint(StrictModel):
    trees: TreeCount
    validation_segment_macro_f1: Annotated[float, Field(ge=0, le=1)]


class ForestValidationCurve(StrictModel):
    """Validation measurements reconstructed from the first trees of a saved forest."""

    bundle_sha256: Sha256
    dataset_sha256: Sha256
    validation_reference_sha256: Sha256
    metric: Literal["decoded_segment_macro_f1/iou-0.50"]
    points: Annotated[tuple[ForestValidationPoint, ...], Field(min_length=2, max_length=128)]

    @model_validator(mode="after")
    def increasing_tree_counts(self) -> Self:
        counts = [point.trees for point in self.points]
        if counts != sorted(set(counts)):
            raise ValueError("Forest measurements require increasing tree counts")
        return self

    def matches_original_fit(
        self, spec: ImportedTrainingSpec, bundle_sha256: str, history: JsonValue
    ) -> bool:
        fits = history.get("fits") if isinstance(history, dict) else None
        if not isinstance(fits, list) or len(fits) != 1 or not isinstance(fits[0], dict):
            return False
        evidence = spec.evidence
        last = self.points[-1]
        return (
            spec.architecture_id == "adaptive"
            and (spec.configuration or {}).get("codec_version") == "adaptive-forest/1.0"
            and self.bundle_sha256 == bundle_sha256
            and self.dataset_sha256
            == (evidence.validation_dataset_sha256 or evidence.dataset_sha256)
            and self.validation_reference_sha256 == evidence.validation_references.content_sha256
            and fits[0].get("trees") == last.trees
            and fits[0].get("seed") == evidence.selected_seed
            and fits[0].get("validation_segment_macro_f1") == last.validation_segment_macro_f1
            and last.validation_segment_macro_f1 == evidence.validation_segment_macro_f1
        )


class TrainingMetrics(StrictModel):
    model_version_id: Identifier
    selected_seed: Seed
    validation_segment_macro_f1: Annotated[float, Field(ge=0, le=1)]
    history: JsonValue
    history_note: str | None = None
    repeat_outcomes: tuple[RepeatOutcome, ...] = ()
    direction_quality: DirectionQualityEvidence | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    forest_validation_curve: ForestValidationCurve | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


MAX_PROGRESS_BATCH = 512
# The largest epoch schedule has 64 seeds with 1,000 epochs each.
MAX_PROGRESS_POINTS = 64_000
type HistorySeries = Literal["candidates", "epochs", "updates", "fits"]


class CandidateProgress(StrictModel):
    series: ClassVar[HistorySeries] = "candidates"
    regularization: Annotated[float, Field(gt=0)]
    validation_segment_macro_f1: Annotated[float, Field(ge=0, le=1)]


class ForestFitProgress(StrictModel):
    series: ClassVar[HistorySeries] = "fits"
    seed: Seed
    trees: TreeCount
    max_depth: TreeDepth
    validation_segment_macro_f1: Annotated[float, Field(ge=0, le=1)]


class TemporalProgress(StrictModel):
    seed: Seed
    best_step: int | None = Field(default=None, ge=1, exclude_if=lambda value: value is None)
    training_loss: Annotated[float, Field(ge=0)]
    validation_segment_macro_f1: float | None = Field(
        default=None, ge=0, le=1, exclude_if=lambda value: value is None
    )


class EpochProgress(TemporalProgress):
    series: ClassVar[HistorySeries] = "epochs"
    epoch: Annotated[int, Field(ge=1, le=1000)]
    stop_reason: Literal["early_stopping", "epoch_limit"] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


class UpdateProgress(TemporalProgress):
    series: ClassVar[HistorySeries] = "updates"
    update: Annotated[int, Field(ge=1, le=32_000)]
    stop_reason: Literal["update_limit"] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


type ProgressPoint = CandidateProgress | EpochProgress | UpdateProgress | ForestFitProgress


class TrainingProgress(StrictModel):
    attempt: Annotated[int, Field(ge=0)]
    cursor: Annotated[int, Field(ge=0, le=MAX_PROGRESS_POINTS)]
    has_more: bool
    history: dict[HistorySeries, list[ProgressPoint]]


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
