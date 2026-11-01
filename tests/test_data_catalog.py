"""Raw membership identities and supervision access remain independent of features."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from regimex.benchmarks import DirectionSemantics
from regimex.data_catalog import (
    DatasetManifest,
    Partition,
    RawTrainingData,
    ReferenceMember,
    ReferencePayload,
    ReferenceSet,
    SnapshotMember,
    SourceSnapshot,
    SplitGroup,
    analysis_profile_hash,
    content_hash,
    create_reference_member,
    create_snapshot_member,
    load_training_data,
    seal_dataset,
    seal_reference_set,
    validate_reference_membership,
)
from regimex.directions import (
    AnalysisConfig,
    Direction,
    DirectionAnnotations,
    DirectionInterval,
    summarize_interval,
)
from regimex.exports import ExportPayloadV2, seal_export

PROFILE = AnalysisConfig(min_duration=2, minimum_observations=2, minimum_price_coverage=0.5)
SEMANTICS = DirectionSemantics(
    profile_sha256=analysis_profile_hash(PROFILE), decoder_version="shared-decoder/1.0"
)


def snapshot(index: int, *, instrument_count: int = 1) -> SourceSnapshot:
    value = json.loads((Path(__file__).parent / "fixtures" / "experiment-v2.json").read_bytes())
    value.pop("content_sha256")
    for name in ("snapshot_id", "run_id", "configuration_id"):
        value["source"][name] = f"{name}-{index}"
    value["time_grid"]["step_count"] = 6
    value["time_grid"]["completed_horizon"] = 103.0
    template_instrument = value["instruments"][0]
    template_row = value["observations"][0]
    value["instruments"] = []
    value["observations"] = []
    for instrument_index in range(instrument_count):
        instrument_id = f"asset-{instrument_index}/Q"
        initial = float(100 + index * 10 + instrument_index * 50)
        value["instruments"].append(
            {
                **template_instrument,
                "instrument_id": instrument_id,
                "base_asset": f"asset-{instrument_index}",
                "initial_price": initial,
            }
        )
        for step in range(6):
            value["observations"].append(
                {
                    **template_row,
                    "instrument_id": instrument_id,
                    "step": step,
                    "last_price": None if step == 2 else initial + step,
                    "trade_count": None if step == 2 else (0 if step == 0 else 1),
                    "trade_volume": None if step == 2 else (0.0 if step == 0 else 0.125),
                    "price_state": "unknown"
                    if step == 2
                    else ("carried" if step == 0 else "observed"),
                }
            )
    return SourceSnapshot(
        source_namespace="simulator-integration",
        split_group=SplitGroup(namespace="generator-v1", group_id=f"seed-{index}", seed=index),
        export=seal_export(ExportPayloadV2.model_validate_json(json.dumps(value))),
    )


def annotations(
    source: SourceSnapshot, *, human: bool = False, direction: Direction = Direction.GROWTH
) -> DirectionAnnotations:
    export = source.export
    return DirectionAnnotations(
        experiment_id=export.source.snapshot_id,
        policy="human-directions/1.0" if human else "retrospective-directions/3.0",
        policy_config=PROFILE.model_dump(mode="json"),
        reference_kind="human" if human else "automatic",
        reviewer="independent-reviewer" if human else None,
        intervals=tuple(
            DirectionInterval(
                instrument_id=instrument.instrument_id,
                start=0,
                stop=export.time_grid.step_count,
                direction=direction,
                status="observed",
                statistics=summarize_interval(
                    tuple(
                        row
                        for row in export.observations
                        if row.instrument_id == instrument.instrument_id
                    ),
                    0,
                    export.time_grid.step_count,
                ),
            )
            for instrument in export.instruments
        ),
    )


def dataset() -> tuple[DatasetManifest, tuple[SourceSnapshot, ...]]:
    sources = tuple(snapshot(index, instrument_count=2) for index in range(3))
    partitions: tuple[Partition, ...] = ("train", "validation", "test")
    return seal_dataset(
        "raw-fixture",
        tuple(create_snapshot_member(s, p) for s, p in zip(sources, partitions, strict=True)),
    ), sources


def reference_set(
    manifest: DatasetManifest,
    sources: tuple[SourceSnapshot, ...],
    partition: Partition,
    *,
    human: bool = False,
    documents: tuple[DirectionAnnotations, ...] | None = None,
) -> ReferenceSet:
    members = tuple(m for m in manifest.members if m.partition == partition)
    if documents is None:
        documents = tuple(
            annotations(
                next(s for s in sources if s.export.source.snapshot_id == m.snapshot_id),
                human=human,
            )
            for m in members
        )
    return seal_reference_set(
        ReferencePayload(
            reference_set_id=f"{partition}-{'human' if human else 'automatic'}",
            dataset_sha256=manifest.content_sha256,
            partition=partition,
            kind="human" if human else "automatic",
            policy="human-directions/1.0" if human else "retrospective-directions/3.0",
            semantics=SEMANTICS,
            analysis_profile=PROFILE,
            reviewer="independent-reviewer" if human else None,
            blind_review=human,
            independent=human,
            members=tuple(
                create_reference_member(m, a) for m, a in zip(members, documents, strict=True)
            ),
        )
    )


@pytest.mark.parametrize("previous", ["1.0", "2.0"])
def test_previous_automatic_reference_policy_is_rejected_with_a_valid_digest(previous: str) -> None:
    manifest, sources = dataset()
    current = reference_set(manifest, sources, "train")
    payload = current.model_dump(mode="json", exclude={"content_sha256"})
    assert payload["policy"] == "retrospective-directions/3.0"
    payload["policy"] = f"retrospective-directions/{previous}"
    previous_document = json.dumps({**payload, "content_sha256": content_hash(payload)})
    with pytest.raises(ValidationError, match="reference policy must match"):
        ReferenceSet.model_validate_json(previous_document)
    assert current.policy == "retrospective-directions/3.0"


def test_membership_identity_binds_order_partitions_groups_and_raw_content() -> None:
    manifest, sources = dataset()
    assert DatasetManifest.model_validate_json(manifest.model_dump_json()) == manifest
    reversed_members = seal_dataset(manifest.dataset_id, tuple(reversed(manifest.members)))
    assert reversed_members.content_sha256 != manifest.content_sha256
    assert reversed_members.split_manifest_sha256 != manifest.split_manifest_sha256
    swapped = tuple(
        m.model_copy(update={"partition": "validation" if m.partition == "train" else "train"})
        if m.partition != "test"
        else m
        for m in manifest.members
    )
    assert seal_dataset(manifest.dataset_id, swapped).content_sha256 != manifest.content_sha256
    changed = sources[0].model_copy(
        update={"split_group": SplitGroup(namespace="generator-v1", group_id="new-group", seed=10)}
    )
    updated = (create_snapshot_member(changed, "train"), *manifest.members[1:])
    assert seal_dataset(manifest.dataset_id, updated).content_sha256 != manifest.content_sha256


@pytest.mark.parametrize("kind", ["run", "group", "seed", "observation", "snapshot"])
def test_partitions_reject_overlapping_experiments_groups_and_observations(kind: str) -> None:
    manifest, sources = dataset()
    train, validation, test = manifest.members
    if kind == "run":
        validation = validation.model_copy(update={"run_id": train.run_id})
    elif kind == "group":
        validation = validation.model_copy(
            update={
                "split_group": validation.split_group.model_copy(
                    update={"group_id": train.split_group.group_id}
                )
            }
        )
    elif kind == "seed":
        validation = validation.model_copy(
            update={
                "split_group": validation.split_group.model_copy(
                    update={"seed": train.split_group.seed}
                )
            }
        )
    elif kind == "observation":
        # A single duplicate instrument rejects the complete experiment allocation.
        validation = validation.model_copy(
            update={
                "observation_hashes": (
                    train.observation_hashes[0],
                    validation.observation_hashes[1],
                )
            }
        )
    else:
        validation = create_snapshot_member(sources[0], "validation")
    with pytest.raises(ValueError, match=r"duplicate snapshot|partitions overlap"):
        seal_dataset("invalid", (train, validation, test))


def test_seed_namespaces_do_not_conflate_unrelated_generators() -> None:
    manifest, _ = dataset()
    train, validation, test = manifest.members
    validation = validation.model_copy(
        update={"split_group": SplitGroup(namespace="other-generator", group_id="seed-0", seed=0)}
    )
    assert seal_dataset("independent-generators", (train, validation, test))


def test_raw_hash_detects_duplicates_after_renaming_source_and_instrument() -> None:
    first = create_snapshot_member(snapshot(0), "train")
    renamed = snapshot(0).model_dump(mode="json")
    renamed["export"]["source"]["run_id"] = "renamed-run"
    renamed["export"]["source"]["snapshot_id"] = "renamed-snapshot"
    renamed["export"]["instruments"][0]["instrument_id"] = "renamed/Q"
    for row in renamed["export"]["observations"]:
        row["instrument_id"] = "renamed/Q"
    renamed["export"].pop("content_sha256")
    renamed["export"] = seal_export(
        ExportPayloadV2.model_validate_json(json.dumps(renamed["export"]))
    ).model_dump(mode="json")
    second = create_snapshot_member(
        SourceSnapshot.model_validate_json(json.dumps(renamed)), "validation"
    )
    assert first.export_sha256 != second.export_sha256
    assert first.observation_hashes == second.observation_hashes


def test_missing_aggregates_and_observed_zero_have_distinct_hashes() -> None:
    source = snapshot(0)
    payload = source.export.model_dump(mode="json", exclude={"content_sha256"})
    payload["observations"][2].update(
        last_price=101.0, trade_count=0, trade_volume=0.0, price_state="carried"
    )
    export = seal_export(ExportPayloadV2.model_validate_json(json.dumps(payload)))
    changed = source.model_copy(update={"export": export})
    assert (
        create_snapshot_member(source, "train").observation_hashes
        != create_snapshot_member(changed, "train").observation_hashes
    )


@pytest.mark.parametrize("missing", ["train", "validation", "test"])
def test_every_partition_is_required(missing: str) -> None:
    manifest, _ = dataset()
    remaining = tuple(m for m in manifest.members if m.partition != missing)
    with pytest.raises(ValueError):
        seal_dataset("incomplete", remaining)


def test_two_instruments_cannot_be_assigned_separately() -> None:
    manifest, _ = dataset()
    first = manifest.members[0]
    with pytest.raises(ValueError, match="one fingerprint per complete instrument"):
        seal_dataset(
            "missing-instrument",
            (
                first.model_copy(update={"instrument_ids": first.instrument_ids[:1]}),
                *manifest.members[1:],
            ),
        )


def test_training_loader_reads_only_train_and_validation_with_independent_references() -> None:
    manifest, sources = dataset()
    labels = reference_set(manifest, sources, "train")
    validation = reference_set(
        manifest,
        sources,
        "validation",
        human=True,
        documents=(annotations(sources[1], human=True, direction=Direction.SIDEWAYS),),
    )
    reads: list[tuple[str, str]] = []

    def read_snapshot(member: SnapshotMember) -> SourceSnapshot:
        reads.append(("snapshot", member.partition))
        assert member.partition != "test"
        return next(s for s in sources if s.export.source.snapshot_id == member.snapshot_id)

    documents: list[DirectionAnnotations] = []

    def read_reference(reference: ReferenceSet, member: ReferenceMember) -> DirectionAnnotations:
        reads.append(("reference", reference.partition))
        assert reference.partition != "test"
        source = next(s for s in sources if s.export.source.snapshot_id == member.snapshot_id)
        document = annotations(
            source,
            human=reference.kind == "human",
            direction=Direction.SIDEWAYS if reference.kind == "human" else Direction.GROWTH,
        )
        documents.append(document)
        return document

    result = load_training_data(
        manifest, labels, validation, snapshot_reader=read_snapshot, reference_reader=read_reference
    )
    assert reads == [
        ("snapshot", "train"),
        ("reference", "train"),
        ("snapshot", "validation"),
        ("reference", "validation"),
    ]
    assert result.dataset_sha256 == manifest.content_sha256
    assert result.split_manifest_sha256 == manifest.split_manifest_sha256
    assert result.labels.kind == "automatic"
    assert result.validation_references.kind == "human"
    assert result.train[0].annotations.intervals[0].direction == Direction.GROWTH
    assert result.validation[0].annotations.intervals[0].direction == Direction.SIDEWAYS
    assert (
        result.labels.direction_semantics_sha256
        == result.validation_references.direction_semantics_sha256
    )
    assert not hasattr(result, "test")
    assert len(result.train[0].export.instruments) == 2
    assert "seed" not in result.model_dump_json()
    assert "split_group" not in result.model_dump_json()
    assert "generation" not in result.model_dump_json()
    assert result.train[0].export.observations[0].trade_count == 0
    assert result.train[0].export.observations[2].trade_count is None
    assert result.train[0].export.observations[0].last_price is not None
    assert result.train[0].export.observations[2].last_price is None
    documents[0].policy_config["minimum_observations"] = 99
    assert result.train[0].annotations.analysis_profile.minimum_observations == 2
    with pytest.raises(ValidationError, match="frozen"):
        result.train[0].annotations.analysis_profile.min_duration = 3


@pytest.mark.parametrize("position", ["labels", "validation"])
def test_test_reference_access_is_denied_before_any_reader(position: str) -> None:
    manifest, sources = dataset()
    labels = reference_set(manifest, sources, "train")
    validation = reference_set(manifest, sources, "validation")
    final = reference_set(manifest, sources, "test")

    def forbidden_snapshot(member: SnapshotMember) -> SourceSnapshot:
        raise AssertionError("reader was called before access validation")

    def forbidden_reference(ref: ReferenceSet, member: ReferenceMember) -> DirectionAnnotations:
        raise AssertionError("reference payload was accessed")

    with pytest.raises(ValueError, match="only train labels and validation references"):
        load_training_data(
            manifest,
            final if position == "labels" else labels,
            final if position == "validation" else validation,
            snapshot_reader=forbidden_snapshot,
            reference_reader=forbidden_reference,
        )


def test_reference_membership_is_complete_and_binds_dataset_content() -> None:
    manifest, sources = dataset()
    refs = reference_set(manifest, sources, "validation")
    changed = seal_reference_set(
        ReferencePayload.model_validate(
            {
                **refs.model_dump(exclude={"content_sha256"}),
                "members": (refs.members[0].model_copy(update={"snapshot_id": "unknown"}),),
            }
        )
    )
    with pytest.raises(ValueError, match="complete ordered dataset partition"):
        validate_reference_membership(manifest, changed)
    different_dataset = seal_dataset("different-dataset", manifest.members)
    with pytest.raises(ValueError, match="different dataset identity"):
        validate_reference_membership(different_dataset, refs)


@pytest.mark.parametrize(
    "corruption", ["raw_hash", "reference_hash", "coverage", "availability", "statistics"]
)
def test_loader_rejects_corruption_and_invalid_annotations(corruption: str) -> None:
    manifest, sources = dataset()
    document = annotations(sources[0])
    interval = document.intervals[0]
    if corruption == "coverage":
        document = document.model_copy(
            update={"intervals": (interval.model_copy(update={"stop": 5}), *document.intervals[1:])}
        )
    elif corruption == "availability":
        document = document.model_copy(
            update={
                "intervals": (
                    interval.model_copy(update={"direction": None, "status": "no_trades"}),
                    *document.intervals[1:],
                )
            }
        )
    elif corruption == "statistics":
        statistics = interval.statistics.model_copy(update={"trade_count": 0})
        document = document.model_copy(
            update={
                "intervals": (
                    interval.model_copy(update={"statistics": statistics}),
                    *document.intervals[1:],
                )
            }
        )
    labels = reference_set(manifest, sources, "train", documents=(document,))
    validation = reference_set(manifest, sources, "validation")

    def read_snapshot(member: SnapshotMember) -> SourceSnapshot:
        source = next(s for s in sources if s.export.source.snapshot_id == member.snapshot_id)
        if corruption == "raw_hash":
            return source.model_copy(
                update={"export": source.export.model_copy(update={"content_sha256": "a" * 64})}
            )
        return source

    def read_reference(ref: ReferenceSet, member: ReferenceMember) -> DirectionAnnotations:
        if ref.partition == "train":
            if corruption == "reference_hash":
                return document.model_copy(
                    update={
                        "intervals": (
                            interval.model_copy(update={"direction": Direction.SIDEWAYS}),
                            *document.intervals[1:],
                        )
                    }
                )
            return document
        return annotations(sources[1])

    with pytest.raises(ValueError):
        load_training_data(
            manifest,
            labels,
            validation,
            snapshot_reader=read_snapshot,
            reference_reader=read_reference,
        )


def test_human_references_require_honest_independence_and_matching_profile() -> None:
    manifest, sources = dataset()
    refs = reference_set(manifest, sources, "validation", human=True)
    payload = refs.model_dump(exclude={"content_sha256"})
    with pytest.raises(ValueError, match="blind-review"):
        seal_reference_set(ReferencePayload.model_validate({**payload, "blind_review": False}))
    with pytest.raises(ValueError, match="annotation profile"):
        seal_reference_set(
            ReferencePayload.model_validate(
                {**payload, "analysis_profile": PROFILE.model_copy(update={"min_duration": 3})}
            )
        )


def test_corrupt_manifest_identity_fails_without_materializing_observations() -> None:
    manifest, sources = dataset()
    references = reference_set(manifest, sources, "validation")
    for bad_dataset, bad_references in (
        (manifest.model_copy(update={"content_sha256": "a" * 64}), references),
        (manifest, references.model_copy(update={"content_sha256": "b" * 64})),
    ):
        with pytest.raises(ValueError, match="content SHA-256 mismatch"):
            validate_reference_membership(bad_dataset, bad_references)


def test_incompatible_reference_semantics_fail_before_materialization() -> None:
    manifest, sources = dataset()
    labels = reference_set(manifest, sources, "train")
    validation = reference_set(manifest, sources, "validation", human=True)
    validation = seal_reference_set(
        ReferencePayload.model_validate(
            {
                **validation.model_dump(exclude={"content_sha256"}),
                "semantics": SEMANTICS.model_copy(update={"decoder_version": "other-decoder/2.0"}),
            }
        )
    )

    def forbidden_snapshot(member: SnapshotMember) -> SourceSnapshot:
        raise AssertionError("observations should not be materialized")

    def forbidden_reference(ref: ReferenceSet, member: ReferenceMember) -> DirectionAnnotations:
        raise AssertionError("references should not be materialized")

    with pytest.raises(ValueError, match="incompatible semantics"):
        load_training_data(
            manifest,
            labels,
            validation,
            snapshot_reader=forbidden_snapshot,
            reference_reader=forbidden_reference,
        )


def test_catalog_manifests_and_imports_do_not_require_scientific_dependencies(
    tmp_path: Path,
) -> None:
    code = """
import sys
import regimex.data_catalog
assert not {'numpy', 'torch', 'sklearn'} & set(sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], cwd=tmp_path, check=True)


def raw_training_data() -> RawTrainingData:
    manifest, sources = dataset()
    labels = reference_set(manifest, sources, "train")
    validation = reference_set(manifest, sources, "validation", human=True)

    def read_snapshot(member: SnapshotMember) -> SourceSnapshot:
        return next(s for s in sources if s.export.source.snapshot_id == member.snapshot_id)

    def read_reference(ref: ReferenceSet, member: ReferenceMember) -> DirectionAnnotations:
        source = next(s for s in sources if s.export.source.snapshot_id == member.snapshot_id)
        return annotations(source, human=ref.kind == "human")

    return load_training_data(
        manifest, labels, validation, snapshot_reader=read_snapshot, reference_reader=read_reference
    )


@pytest.mark.parametrize("boundary", ["constructor", "instance", "json"])
@pytest.mark.parametrize(
    "corruption",
    [
        "validation_is_train",
        "source_run",
        "renamed_duplicate_observations",
        "snapshot_identity",
        "annotation_identity",
        "annotation_profile",
        "reference_provenance",
        "reference_policy",
        "coverage",
        "availability",
        "statistics",
        "profile",
        "semantics",
        "label_semantics",
        "validation_semantics",
        "export_digest",
        "human_reviewer",
    ],
)
def test_raw_training_intake_rejects_constructed_or_copied_mistakes(
    boundary: str, corruption: str
) -> None:
    original = raw_training_data()
    changed = original
    sample = original.validation[0]
    annotation = sample.annotations
    interval = annotation.intervals[0]
    if corruption == "validation_is_train":
        changed = original.model_copy(
            update={"validation": original.train, "validation_references": original.labels}
        )
    elif corruption in {"source_run", "renamed_duplicate_observations"}:
        export = (
            original.train[0].export
            if corruption == "renamed_duplicate_observations"
            else sample.export
        )
        payload = export.model_dump(mode="json", exclude={"content_sha256"})
        payload["source"]["snapshot_id"] = sample.snapshot_id
        payload["source"]["run_id"] = (
            "independent-run"
            if corruption == "renamed_duplicate_observations"
            else original.train[0].export.source.run_id
        )
        export = seal_export(ExportPayloadV2.model_validate_json(json.dumps(payload)))
        sample = sample.model_copy(update={"export": export})
        if corruption == "renamed_duplicate_observations":
            # Rebuild valid human statistics and rename both identities so only
            # the ID-independent raw fingerprint can identify the overlap.
            source = snapshot(1).model_copy(update={"export": export})
            document = annotations(source, human=True)
            annotation = annotation.model_copy(update={"intervals": document.intervals})
            sample = sample.model_copy(update={"source_namespace": "different-source"})
    elif corruption == "snapshot_identity":
        sample = sample.model_copy(update={"snapshot_id": "different-snapshot"})
    elif corruption == "annotation_identity":
        annotation = annotation.model_copy(update={"experiment_id": "different-snapshot"})
    elif corruption == "annotation_profile":
        annotation = annotation.model_copy(
            update={"analysis_profile": PROFILE.model_copy(update={"min_duration": 3})}
        )
    elif corruption == "reference_provenance":
        changed = original.model_copy(update={"validation_references": original.labels})
    elif corruption == "reference_policy":
        changed = original.model_copy(
            update={
                "validation_references": original.validation_references.model_copy(
                    update={"policy": "other-policy/1.0"}
                )
            }
        )
    elif corruption == "coverage":
        annotation = annotation.model_copy(
            update={
                "intervals": (interval.model_copy(update={"stop": 5}), *annotation.intervals[1:])
            }
        )
    elif corruption == "availability":
        annotation = annotation.model_copy(
            update={
                "intervals": (
                    interval.model_copy(update={"direction": None, "status": "no_trades"}),
                    *annotation.intervals[1:],
                )
            }
        )
    elif corruption == "statistics":
        statistics = interval.statistics.model_copy(update={"trade_count": 0})
        annotation = annotation.model_copy(
            update={
                "intervals": (
                    interval.model_copy(update={"statistics": statistics}),
                    *annotation.intervals[1:],
                )
            }
        )
    elif corruption == "profile":
        changed = original.model_copy(
            update={"analysis_profile": PROFILE.model_copy(update={"min_duration": 3})}
        )
    elif corruption == "semantics":
        changed = original.model_copy(
            update={
                "semantics": SEMANTICS.model_copy(
                    update={"decoder_version": "different-decoder/2.0"}
                )
            }
        )
    elif corruption in {"label_semantics", "validation_semantics"}:
        name = "labels" if corruption == "label_semantics" else "validation_references"
        identity = getattr(original, name).model_copy(
            update={"direction_semantics_sha256": "f" * 64}
        )
        changed = original.model_copy(update={name: identity})
    elif corruption == "export_digest":
        sample = sample.model_copy(
            update={"export": sample.export.model_copy(update={"content_sha256": "a" * 64})}
        )
    elif corruption == "human_reviewer":
        annotation = annotation.model_copy(update={"reviewer": None})
    if sample != original.validation[0] or annotation != original.validation[0].annotations:
        sample = sample.model_copy(update={"annotations": annotation})
        changed = changed.model_copy(update={"validation": (sample,)})
    with pytest.raises(ValueError):
        if boundary == "constructor":
            RawTrainingData(**changed.model_dump())
        elif boundary == "instance":
            RawTrainingData.model_validate(changed)
        else:
            RawTrainingData.model_validate_json(changed.model_dump_json())


def test_valid_raw_training_data_preserves_independent_reference_sources_on_revalidation() -> None:
    original = raw_training_data()
    revalidated = RawTrainingData.model_validate(original)
    assert revalidated.labels.kind == "automatic"
    assert revalidated.validation_references.kind == "human"
    assert RawTrainingData.model_validate_json(original.model_dump_json()) == original
