"""Immutable raw membership and independent supervision, before backend features.

These contracts contain no filesystem paths or storage implementation. Readers
are explicit capabilities supplied by the application. The training loader
requests only training labels and validation references; a final evaluator needs
a separate, authorized path. Content hashes establish identity, not authenticity.
"""

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Literal, Self

from pydantic import ConfigDict, Field, model_validator

from regimex.benchmarks import DirectionSemantics, ReferenceProvenance
from regimex.contracts import MAX_STEPS, Identifier, Observation, Seed, StrictModel
from regimex.directions import (
    DIRECTION_POLICY,
    AnalysisConfig,
    DirectionAnnotations,
    DirectionInterval,
)
from regimex.exports import SHA256, ExportInstrument, ExportObservation, ExportV2
from regimex.model_bundles import ReferenceIdentity
from regimex.storage import json_bytes

Partition = Literal["train", "validation", "test"]


class CatalogRecord(StrictModel):
    model_config = ConfigDict(revalidate_instances="always")


def content_hash(value: object) -> str:
    """SHA-256 of the repository's canonical JSON encoding, including its newline."""
    return hashlib.sha256(json_bytes(value)).hexdigest()


def analysis_profile_hash(profile: AnalysisConfig) -> str:
    return content_hash(profile.model_dump(mode="json"))


class SplitGroup(CatalogRecord):
    """Source-controlled dependence group, separate from prediction observations.

    A seed is meaningful only within its generator namespace. The namespace must
    remain stable across dataset versions and source aliases. Missing seed means
    an external run, not an observed seed of zero.
    """

    namespace: Identifier
    group_id: Identifier
    seed: Seed | None = None


class SourceSnapshot(CatalogRecord):
    version: Literal["raw-snapshot/1.0"] = "raw-snapshot/1.0"
    source_namespace: Identifier
    split_group: SplitGroup
    export: ExportV2


def observation_hashes(export: ExportV2) -> tuple[str, ...]:
    """Conservative whole-instrument fingerprints preserving recorded exposures.

    IDs, source declarations, seeds and annotations cannot disguise duplicate
    observations. Missing values remain distinct from observed zero. This detects
    exact complete sequences, not shifted or differently sampled partial windows;
    source run/group identities prevent splitting windows of the same run.
    """
    hashes = []
    for instrument in export.instruments:
        rows = []
        for row in export.observations:
            if row.instrument_id != instrument.instrument_id:
                continue
            values = {name: getattr(row, name) for name in Observation.model_fields}
            del values["instrument_id"]
            normalized = {
                name: (0.0 if value == 0 else value) if isinstance(value, float) else value
                for name, value in values.items()
            }
            rows.append(normalized)
        # Names and external IDs do not contribute to the observation fingerprint.
        hashes.append(content_hash({"version": "observations/1.0", "rows": rows}))
    return tuple(sorted(hashes))


class SnapshotMember(CatalogRecord):
    source_namespace: Identifier
    snapshot_id: Identifier
    source_id: Identifier
    run_id: Identifier
    export_sha256: SHA256
    observation_hashes: Annotated[tuple[SHA256, ...], Field(min_length=1, max_length=2)]
    instrument_ids: Annotated[tuple[Identifier, ...], Field(min_length=1, max_length=2)]
    step_count: Annotated[int, Field(ge=1, le=MAX_STEPS)]
    split_group: SplitGroup
    partition: Partition

    @model_validator(mode="after")
    def complete_instruments(self) -> Self:
        if len(set(self.instrument_ids)) != len(self.instrument_ids):
            raise ValueError("snapshot instrument identifiers must be unique")
        if len(self.observation_hashes) != len(self.instrument_ids):
            raise ValueError("snapshot requires one fingerprint per complete instrument")
        return self


def create_snapshot_member(snapshot: SourceSnapshot, partition: Partition) -> SnapshotMember:
    # Reparse nested instances too: model_copy/model_construct are not boundary validation.
    snapshot = SourceSnapshot.model_validate_json(snapshot.model_dump_json())
    export = snapshot.export
    return SnapshotMember(
        source_namespace=snapshot.source_namespace,
        snapshot_id=export.source.snapshot_id,
        source_id=export.source.source_id,
        run_id=export.source.run_id,
        export_sha256=export.content_sha256,
        observation_hashes=observation_hashes(export),
        instrument_ids=tuple(item.instrument_id for item in export.instruments),
        step_count=export.time_grid.step_count,
        split_group=snapshot.split_group,
        partition=partition,
    )


class DatasetPayload(CatalogRecord):
    version: Literal["raw-dataset/1.0"] = "raw-dataset/1.0"
    dataset_id: Identifier
    members: Annotated[tuple[SnapshotMember, ...], Field(min_length=3)]

    @model_validator(mode="after")
    def isolated_partitions(self) -> Self:
        if {member.partition for member in self.members} != {"train", "validation", "test"}:
            raise ValueError("dataset requires nonempty train, validation and test partitions")
        snapshots: set[tuple[str, str]] = set()
        partition_by_key: dict[tuple[str, ...], Partition] = {}
        for member in self.members:
            identity = (member.source_namespace, member.snapshot_id)
            if identity in snapshots:
                raise ValueError("dataset contains duplicate snapshot membership")
            snapshots.add(identity)
            group = member.split_group
            keys = [
                ("run", member.source_namespace, member.source_id, member.run_id),
                ("group", group.namespace, group.group_id),
                *(("observation", value) for value in member.observation_hashes),
            ]
            if group.seed is not None:
                keys.append(("seed", group.namespace, str(group.seed)))
            for key in keys:
                if key in partition_by_key and partition_by_key[key] != member.partition:
                    raise ValueError(
                        "dataset partitions overlap by run, group, seed or observations"
                    )
                partition_by_key[key] = member.partition
        return self


class DatasetManifest(DatasetPayload):
    content_sha256: SHA256

    @model_validator(mode="after")
    def immutable_identity(self) -> Self:
        if self.content_sha256 != content_hash(
            self.model_dump(mode="json", exclude={"content_sha256"})
        ):
            raise ValueError("dataset manifest content SHA-256 mismatch")
        return self

    @property
    def split_manifest_sha256(self) -> str:
        # Bind order, source identities and every grouping declaration, not names alone.
        return content_hash(
            {
                "version": "raw-splits/1.0",
                "members": [m.model_dump(mode="json") for m in self.members],
            }
        )


def seal_dataset(dataset_id: str, members: tuple[SnapshotMember, ...]) -> DatasetManifest:
    payload = DatasetPayload(dataset_id=dataset_id, members=members)
    data = payload.model_dump(mode="json")
    return DatasetManifest.model_validate_json(
        json_bytes({**data, "content_sha256": content_hash(data)})
    )


class ReferenceMember(CatalogRecord):
    source_namespace: Identifier
    snapshot_id: Identifier
    export_sha256: SHA256
    annotations_sha256: SHA256


def create_reference_member(
    member: SnapshotMember, annotations: DirectionAnnotations
) -> ReferenceMember:
    annotations = DirectionAnnotations.model_validate_json(annotations.model_dump_json())
    if annotations.experiment_id != member.snapshot_id:
        raise ValueError("annotations reference a different snapshot")
    return ReferenceMember(
        source_namespace=member.source_namespace,
        snapshot_id=member.snapshot_id,
        export_sha256=member.export_sha256,
        annotations_sha256=content_hash(annotations.model_dump(mode="json")),
    )


class ReferencePayload(CatalogRecord):
    version: Literal["reference-set/1.0"] = "reference-set/1.0"
    reference_set_id: Identifier
    dataset_sha256: SHA256
    partition: Partition
    kind: Literal["automatic", "human"]
    policy: Identifier
    semantics: DirectionSemantics
    analysis_profile: AnalysisConfig
    reviewer: Identifier | None = None
    blind_review: bool = False
    independent: bool = False
    members: Annotated[tuple[ReferenceMember, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def compatible_profile_and_provenance(self) -> Self:
        if self.semantics.profile_sha256 != analysis_profile_hash(self.analysis_profile):
            raise ValueError("reference direction semantics differ from the annotation profile")
        ReferenceProvenance(
            reference_set_sha256=content_hash(
                self.model_dump(mode="json", exclude={"content_sha256"})
            ),
            kind=self.kind,
            policy=self.policy,
            semantics=self.semantics,
            reviewer=self.reviewer,
            blind_review=self.blind_review,
            independent=self.independent,
        )
        if self.policy != ("human-directions/1.0" if self.kind == "human" else DIRECTION_POLICY):
            raise ValueError("reference policy must match its declared annotation kind")
        if len({(m.source_namespace, m.snapshot_id) for m in self.members}) != len(self.members):
            raise ValueError("reference set contains duplicate snapshot membership")
        return self


class ReferenceSet(ReferencePayload):
    content_sha256: SHA256

    @model_validator(mode="after")
    def immutable_identity(self) -> Self:
        if self.content_sha256 != content_hash(
            self.model_dump(mode="json", exclude={"content_sha256"})
        ):
            raise ValueError("reference set content SHA-256 mismatch")
        return self

    def identity(self) -> ReferenceIdentity:
        return ReferenceIdentity(
            reference_set_id=self.reference_set_id,
            kind=self.kind,
            policy=self.policy,
            content_sha256=self.content_sha256,
            direction_semantics_sha256=content_hash(self.semantics.model_dump(mode="json")),
        )


def seal_reference_set(payload: ReferencePayload) -> ReferenceSet:
    data = payload.model_dump(mode="json")
    return ReferenceSet.model_validate_json(
        json_bytes({**data, "content_sha256": content_hash(data)})
    )


class ValidatedAnnotations(CatalogRecord):
    """Deeply immutable annotation record with a validated analysis profile."""

    experiment_id: Identifier
    policy: Identifier
    analysis_profile: AnalysisConfig
    reference_kind: Literal["automatic", "human"]
    reviewer: Identifier | None
    intervals: tuple[DirectionInterval, ...]


class LabeledSnapshot(CatalogRecord):
    source_namespace: Identifier
    snapshot_id: Identifier
    export: ExportV2
    annotations: ValidatedAnnotations


class RawTrainingData(CatalogRecord):
    """Checked train/validation values, without proof of their original allocation.

    Group/seed isolation, complete dataset membership, test exclusion and unseen
    history require the checked DatasetManifest and official loader. These checks
    additionally reject a mixed-up raw value at any backend's intake boundary;
    opaque manifest/reference digests alone cannot reconstruct that allocation.
    """

    dataset_sha256: SHA256
    split_manifest_sha256: SHA256
    labels: ReferenceIdentity
    validation_references: ReferenceIdentity
    semantics: DirectionSemantics
    analysis_profile: AnalysisConfig
    train: Annotated[tuple[LabeledSnapshot, ...], Field(min_length=1)]
    validation: Annotated[tuple[LabeledSnapshot, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def consistent_supervision_and_isolation(self) -> Self:
        profile = AnalysisConfig.model_validate_json(self.analysis_profile.model_dump_json())
        semantics = DirectionSemantics.model_validate_json(self.semantics.model_dump_json())
        if semantics.profile_sha256 != analysis_profile_hash(profile):
            raise ValueError("training direction semantics differ from the analysis profile")
        semantics_sha256 = content_hash(semantics.model_dump(mode="json"))
        identities: set[tuple[str, str]] = set()
        partition_by_key: dict[tuple[str, ...], str] = {}
        for partition, samples, identity in (
            ("train", self.train, self.labels),
            ("validation", self.validation, self.validation_references),
        ):
            identity = ReferenceIdentity.model_validate_json(identity.model_dump_json())
            if identity.direction_semantics_sha256 != semantics_sha256:
                raise ValueError("training reference identity has incompatible direction semantics")
            for sample in samples:
                export = ExportV2.model_validate_json(sample.export.model_dump_json())
                annotation_data = sample.annotations.model_dump(mode="json")
                annotation_data["policy_config"] = annotation_data.pop("analysis_profile")
                annotation = DirectionAnnotations.model_validate_json(json_bytes(annotation_data))
                if (
                    sample.snapshot_id != export.source.snapshot_id
                    or annotation.experiment_id != sample.snapshot_id
                ):
                    raise ValueError("training snapshot/export/annotation identities differ")
                if (
                    annotation.reference_kind != identity.kind
                    or annotation.policy != identity.policy
                    or sample.annotations.analysis_profile != profile
                ):
                    raise ValueError(
                        "training annotations differ from reference provenance or profile"
                    )
                annotation.validate_for(
                    SnapshotGrid(
                        experiment_id=sample.snapshot_id,
                        step_count=export.time_grid.step_count,
                        instruments=export.instruments,
                        observations=export.observations,
                    )
                )
                snapshot_identity = (sample.source_namespace, sample.snapshot_id)
                if snapshot_identity in identities:
                    raise ValueError("training values contain duplicate snapshot identities")
                identities.add(snapshot_identity)
                keys = (
                    ("run", sample.source_namespace, export.source.source_id, export.source.run_id),
                    *(("observation", digest) for digest in observation_hashes(export)),
                )
                for key in keys:
                    if key in partition_by_key and partition_by_key[key] != partition:
                        raise ValueError(
                            "training and validation overlap by source run or observations"
                        )
                    partition_by_key[key] = partition
        return self


@dataclass(frozen=True)
class SnapshotGrid:
    experiment_id: str
    step_count: int
    instruments: tuple[ExportInstrument, ...]
    observations: tuple[ExportObservation, ...]

    @classmethod
    def from_export(cls, export: ExportV2) -> SnapshotGrid:
        return cls(
            experiment_id=export.source.snapshot_id,
            step_count=export.time_grid.step_count,
            instruments=export.instruments,
            observations=export.observations,
        )


def validate_reference_membership(dataset: DatasetManifest, references: ReferenceSet) -> None:
    """Validate reference metadata without reading labels, including final references."""
    dataset = DatasetManifest.model_validate_json(dataset.model_dump_json())
    references = ReferenceSet.model_validate_json(references.model_dump_json())
    if references.dataset_sha256 != dataset.content_sha256:
        raise ValueError("reference set belongs to a different dataset identity")
    expected = tuple(
        (member.source_namespace, member.snapshot_id, member.export_sha256)
        for member in dataset.members
        if member.partition == references.partition
    )
    actual = tuple(
        (member.source_namespace, member.snapshot_id, member.export_sha256)
        for member in references.members
    )
    if actual != expected:
        raise ValueError("reference set must cover its complete ordered dataset partition")


def _load_partition(
    dataset: DatasetManifest,
    references: ReferenceSet,
    *,
    snapshot_reader: Callable[[SnapshotMember], SourceSnapshot],
    reference_reader: Callable[[ReferenceSet, ReferenceMember], DirectionAnnotations],
) -> tuple[LabeledSnapshot, ...]:
    result = []
    members = tuple(m for m in dataset.members if m.partition == references.partition)
    for member, reference_member in zip(members, references.members, strict=True):
        snapshot = snapshot_reader(member)
        if create_snapshot_member(snapshot, references.partition) != member:
            raise ValueError("loaded snapshot differs from its dataset identity")
        # Keep no group/seed metadata in the value passed on to model backends.
        export = ExportV2.model_validate_json(snapshot.export.model_dump_json())
        annotations = reference_reader(references, reference_member)
        annotations = DirectionAnnotations.model_validate_json(annotations.model_dump_json())
        if content_hash(annotations.model_dump(mode="json")) != reference_member.annotations_sha256:
            raise ValueError("loaded annotations content SHA-256 mismatch")
        profile = AnalysisConfig.model_validate(annotations.policy_config)
        if (
            annotations.reference_kind != references.kind
            or annotations.policy != references.policy
            or annotations.reviewer != references.reviewer
            or profile != references.analysis_profile
        ):
            raise ValueError("loaded annotations differ from their reference provenance or profile")
        annotations.validate_for(
            SnapshotGrid(
                experiment_id=export.source.snapshot_id,
                step_count=export.time_grid.step_count,
                instruments=export.instruments,
                observations=export.observations,
            )
        )
        result.append(
            LabeledSnapshot(
                source_namespace=member.source_namespace,
                snapshot_id=member.snapshot_id,
                export=export,
                annotations=ValidatedAnnotations(
                    experiment_id=annotations.experiment_id,
                    policy=annotations.policy,
                    analysis_profile=profile,
                    reference_kind=annotations.reference_kind,
                    reviewer=annotations.reviewer,
                    intervals=annotations.intervals,
                ),
            )
        )
    return tuple(result)


def load_training_data(
    dataset: DatasetManifest,
    labels: ReferenceSet,
    validation_references: ReferenceSet,
    *,
    snapshot_reader: Callable[[SnapshotMember], SourceSnapshot],
    reference_reader: Callable[[ReferenceSet, ReferenceMember], DirectionAnnotations],
) -> RawTrainingData:
    """Load raw train/validation data only, before backend-specific feature creation.

    Metadata is checked before either reader runs. Reader capabilities are local
    application dependencies; this function does not grant final-reference access
    or claim to sandbox a callback with broader external permissions.
    """
    dataset = DatasetManifest.model_validate_json(dataset.model_dump_json())
    labels = ReferenceSet.model_validate_json(labels.model_dump_json())
    validation_references = ReferenceSet.model_validate_json(
        validation_references.model_dump_json()
    )
    if labels.partition != "train" or validation_references.partition != "validation":
        raise ValueError("training may read only train labels and validation references")
    for references in (labels, validation_references):
        validate_reference_membership(dataset, references)
    if labels.semantics != validation_references.semantics:
        raise ValueError("training labels and validation references have incompatible semantics")
    return RawTrainingData(
        dataset_sha256=dataset.content_sha256,
        split_manifest_sha256=dataset.split_manifest_sha256,
        labels=labels.identity(),
        validation_references=validation_references.identity(),
        semantics=labels.semantics,
        analysis_profile=labels.analysis_profile,
        train=_load_partition(
            dataset, labels, snapshot_reader=snapshot_reader, reference_reader=reference_reader
        ),
        validation=_load_partition(
            dataset,
            validation_references,
            snapshot_reader=snapshot_reader,
            reference_reader=reference_reader,
        ),
    )
