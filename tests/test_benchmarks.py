"""Historical isolation, durable reservations, and the final-reference boundary."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from regimex.benchmarks import (
    BenchmarkMember,
    BenchmarkRegistry,
    DatasetHistory,
    DirectionSemantics,
    Partition,
    PartitionMembership,
    ReferenceProvenance,
    _preserves_history,
    attest_history,
    freeze_choices,
    open_benchmark,
    record_selection_started,
    register_history,
    reserve_benchmark,
)
from regimex.data_catalog import observation_hashes
from regimex.exports import ExportPayloadV2, ExportV2, load_export_v2, seal_export

START = datetime(2026, 10, 21, tzinfo=UTC)
SEMANTICS = DirectionSemantics(profile_sha256="a" * 64, decoder_version="shared-decoder/1.0")


def at(offset: int) -> datetime:
    return START + timedelta(seconds=offset)


def snapshot(name: str, price: float) -> ExportV2:
    source = load_export_v2(Path(__file__).parent / "fixtures" / "experiment-v2.json")
    data = source.model_dump(mode="json", exclude={"content_sha256"})
    data["source"]["snapshot_id"] = name
    data["instruments"][0]["initial_price"] = price
    data["observations"][0]["last_price"] = price
    return seal_export(ExportPayloadV2.model_validate_json(json.dumps(data)))


def snapshot_member(namespace: str, source: ExportV2, seed: int) -> BenchmarkMember:
    return BenchmarkMember(
        source_namespace=namespace,
        experiment_id=source.source.snapshot_id,
        group_id=f"fixture-seed:{seed}",
        seed=seed,
        observation_hashes=observation_hashes(source),
    )


def member(name: str, seed: int, *, namespace: str = "recorded") -> BenchmarkMember:
    return snapshot_member(namespace, snapshot(name, 100.0 + seed), seed)


def reference(*, human: bool = False) -> ReferenceProvenance:
    return ReferenceProvenance(
        reference_set_sha256=("b" if human else "c") * 64,
        kind="human" if human else "automatic",
        policy="human-directions/1.0" if human else "retrospective-directions/3.0",
        semantics=SEMANTICS,
        reviewer="fixture-reviewer" if human else None,
        blind_review=human,
        independent=human,
    )


def history(
    name: str,
    identity: str,
    memberships: tuple[tuple[Partition, BenchmarkMember], ...],
) -> DatasetHistory:
    return DatasetHistory(
        dataset_id=name,
        dataset_sha256=identity * 64,
        recorded_at=at(0),
        memberships=tuple(
            PartitionMembership(partition=partition, member=item) for partition, item in memberships
        ),
    )


def registry() -> BenchmarkRegistry:
    result = register_history(
        BenchmarkRegistry(),
        history(
            "original",
            "1",
            (
                ("train", member("old-train", 11)),
                ("validation", member("old-validation", 12)),
                ("test", member("old-test", 13)),
            ),
        ),
    )
    result = register_history(
        result,
        history("newer", "2", (("train", member("new-train", 21, namespace="export")),)),
    )
    return attest_history(
        result,
        source_namespaces=("recorded", "export", "candidate"),
        attested_by="fixture-inventory",
        statement="Fixture inventory only; not an assertion about project history.",
        at=at(1),
    )


def reserved() -> BenchmarkRegistry:
    return reserve_benchmark(
        registry(),
        benchmark_id="fixture-final",
        members=(member("unseen", 31, namespace="candidate"),),
        reference=reference(human=True),
        at=at(2),
    )


def frozen() -> BenchmarkRegistry:
    return freeze_choices(
        record_selection_started(reserved(), "fixture-final", at=at(3)),
        "fixture-final",
        at=at(4),
        choices_sha256="d" * 64,
        model_artifact_sha256="e" * 64,
        training_dataset_hashes=("1" * 64,),
        training_reference=reference(),
    )


def test_observation_fingerprints_ignore_names_but_preserve_missing_values() -> None:
    original = snapshot("first", 100.0)
    data = original.model_dump(mode="json", exclude={"content_sha256"})
    data["source"]["snapshot_id"] = "renamed"
    data["producer"] = {"name": "other", "version": "2"}
    instruments = data["instruments"]
    rows = data["observations"]
    instruments[0]["instrument_id"] = "renamed/Q"
    for row in rows:
        row["instrument_id"] = "renamed/Q"
    renamed = seal_export(ExportPayloadV2.model_validate_json(json.dumps(data)))
    assert observation_hashes(original) == observation_hashes(renamed)
    rows[1].update(trade_count=0, trade_volume=0.0)
    observed_zero = seal_export(ExportPayloadV2.model_validate_json(json.dumps(data)))
    assert observation_hashes(original) != observation_hashes(observed_zero)


def test_each_instrument_is_hashed_to_detect_partially_copied_experiments() -> None:
    source = snapshot("source", 100.0)
    data = source.model_dump(mode="json", exclude={"content_sha256"})
    data["source"]["snapshot_id"] = "partially-copied"
    instruments = data["instruments"]
    rows = data["observations"]
    instruments.append(
        {**instruments[0], "instrument_id": "B/Q", "base_asset": "B", "initial_price": 300.0}
    )
    extra: list[dict[str, Any]] = []
    for row in rows:
        extra.append(
            {
                **row,
                "instrument_id": "B/Q",
                "last_price": 300.0 if row["last_price"] is not None else None,
            }
        )
    rows.extend(extra)
    combined = seal_export(ExportPayloadV2.model_validate_json(json.dumps(data)))
    original = snapshot_member("recorded", source, 11)
    candidate = snapshot_member("candidate", combined, 99)
    assert set(original.observation_hashes) < set(candidate.observation_hashes)
    current = registry()
    original_history = history("copy-source", "3", (("development", original),))
    current = register_history(current, original_history)
    current = attest_history(
        current,
        source_namespaces=("recorded", "export", "candidate"),
        attested_by="fixture",
        statement="Complete fixture inventory.",
        at=at(1),
    )
    with pytest.raises(ValueError, match="overlaps a prior partition"):
        reserve_benchmark(
            current,
            benchmark_id="copied-instrument",
            members=(candidate,),
            reference=reference(),
            at=at(2),
        )


@pytest.mark.parametrize("seed", [11, 12, 13, 21])
def test_reservation_checks_every_historical_partition_and_namespace(seed: int) -> None:
    # Distinct IDs, namespace and prices cannot hide a reused seed group.
    candidate = snapshot_member("candidate", snapshot("renamed", 900.0), seed)
    with pytest.raises(ValueError, match="overlaps a prior partition"):
        reserve_benchmark(
            registry(),
            benchmark_id="overlap",
            members=(candidate,),
            reference=reference(),
            at=at(2),
        )


def test_copied_observations_cannot_hide_behind_different_seeds_and_groups() -> None:
    copied = snapshot_member("candidate", snapshot("renamed", 113.0), 99)
    with pytest.raises(ValueError, match="overlaps a prior partition"):
        reserve_benchmark(
            registry(),
            benchmark_id="copied-test-observations",
            members=(copied,),
            reference=reference(),
            at=at(2),
        )


@pytest.mark.parametrize("reuse", ["seed", "group"])
def test_group_and_seed_identity_cannot_be_changed_to_disguise_the_other(reuse: str) -> None:
    original = snapshot_member("candidate", snapshot("different-observations", 900.0), 13)
    data = original.model_dump()
    if reuse == "seed":
        data["group_id"] = "new-provider-group"
    else:
        data["seed"] = None
    with pytest.raises(ValueError, match="overlaps a prior partition"):
        reserve_benchmark(
            registry(),
            benchmark_id="disguised-group",
            members=(BenchmarkMember.model_validate(data),),
            reference=reference(),
            at=at(2),
        )


def test_split_isolation_and_repeated_historical_membership() -> None:
    first = member("first", 11)
    same_group = snapshot_member("export", snapshot("second", 900.0), 11)
    with pytest.raises(ValueError, match="overlapping partitions"):
        history("leaking", "1", (("train", first), ("test", same_group)))
    with pytest.raises(ValueError, match="duplicate experiment membership"):
        history("duplicate", "1", (("train", first), ("train", first)))
    # Reuse in distinct past datasets is retained, rather than hiding old use.
    current = register_history(registry(), history("old-reuse", "3", (("test", first),)))
    assert len(current.history) == 3
    assert current.history_attestation is None


def test_reservation_requires_explicit_complete_inventory_and_scope() -> None:
    with pytest.raises(ValueError, match="must be explicitly attested"):
        reserve_benchmark(
            BenchmarkRegistry(),
            benchmark_id="unattested",
            members=(member("new", 31),),
            reference=reference(),
            at=at(2),
        )
    with pytest.raises(ValueError, match="source namespaces"):
        reserve_benchmark(
            registry(),
            benchmark_id="unknown-source",
            members=(member("new", 31, namespace="not-in-inventory"),),
            reference=reference(),
            at=at(2),
        )


def test_freeze_precedes_explicit_open_and_provenance_stays_independent() -> None:
    current = reserved()
    with pytest.raises(ValueError, match="only after choices freeze"):
        open_benchmark(current, "fixture-final", at=at(5))
    current = record_selection_started(current, "fixture-final", at=at(3))
    current = frozen()
    assert current.benchmarks[0].reference.kind == "human"
    freeze = current.benchmarks[0].freeze
    assert freeze is not None and freeze.training_reference.kind == "automatic"
    current = open_benchmark(current, "fixture-final", at=at(5))
    with pytest.raises(ValueError, match="previously unopened"):
        open_benchmark(current, "fixture-final", at=at(6))
    with pytest.raises(ValueError, match="cannot rewrite selection"):
        _preserves_history(current, frozen())


@pytest.mark.parametrize("event", ["selection", "freeze", "open"])
def test_equal_or_reversed_event_times_are_rejected(event: str) -> None:
    if event == "selection":
        with pytest.raises(ValueError, match="after benchmark reservation"):
            record_selection_started(reserved(), "fixture-final", at=at(2))
    elif event == "freeze":
        with pytest.raises(ValueError, match="after selection starts"):
            freeze_choices(
                record_selection_started(reserved(), "fixture-final", at=at(3)),
                "fixture-final",
                at=at(3),
                choices_sha256="d" * 64,
                model_artifact_sha256="e" * 64,
                training_dataset_hashes=("1" * 64,),
                training_reference=reference(),
            )
    else:
        with pytest.raises(ValueError, match="only after choices freeze"):
            open_benchmark(frozen(), "fixture-final", at=at(4))


def test_semantics_must_match_while_reference_source_can_differ() -> None:
    data = reference().model_dump()
    data["semantics"]["decoder_version"] = "different-decoder/2.0"
    incompatible = ReferenceProvenance.model_validate(data)
    with pytest.raises(ValueError, match="incompatible semantics"):
        freeze_choices(
            record_selection_started(reserved(), "fixture-final", at=at(3)),
            "fixture-final",
            at=at(4),
            choices_sha256="d" * 64,
            model_artifact_sha256="e" * 64,
            training_dataset_hashes=("1" * 64,),
            training_reference=incompatible,
        )


def test_new_dataset_cannot_overlap_a_reserved_benchmark() -> None:
    current = reserved()
    with pytest.raises(ValueError, match="overlaps a prior partition"):
        register_history(
            current,
            history("late-training", "3", (("train", member("new-name", 31)),)),
        )
    current = register_history(
        current, history("late-training", "3", (("train", member("new-name", 41)),))
    )
    with pytest.raises(ValueError, match="must be explicitly attested"):
        record_selection_started(current, "fixture-final", at=at(3))
    # The original inventory evidence survives later inventory changes.
    assert current.benchmarks[0].inventory_attestation.history_sha256


def test_frozen_choices_are_immutable_and_cannot_be_replaced() -> None:
    current = frozen()
    freeze = current.benchmarks[0].freeze
    assert freeze is not None
    with pytest.raises(ValidationError, match="frozen"):
        freeze.model_artifact_sha256 = "9" * 64
    with pytest.raises(ValidationError, match="frozen"):
        current.benchmarks[0].members[0].group_id = "another-group"
    data = current.model_dump()
    data["benchmarks"][0]["freeze"]["model_artifact_sha256"] = "9" * 64
    modified = BenchmarkRegistry.model_validate(data)
    with pytest.raises(ValueError, match="cannot rewrite selection"):
        _preserves_history(current, modified)
    with pytest.raises(ValueError, match="previously unfrozen"):
        freeze_choices(
            current,
            "fixture-final",
            at=at(6),
            choices_sha256="9" * 64,
            model_artifact_sha256="e" * 64,
            training_dataset_hashes=("1" * 64,),
            training_reference=reference(),
        )


def test_incomplete_or_false_reference_provenance_is_rejected() -> None:
    data = reference().model_dump()
    data["independent"] = True
    with pytest.raises(ValueError, match="automatic references cannot claim"):
        ReferenceProvenance.model_validate(data)
    data = reference(human=True).model_dump()
    data["blind_review"] = False
    with pytest.raises(ValueError, match="blind-review attestation"):
        ReferenceProvenance.model_validate(data)
    data = registry().model_dump()
    data["history_attestation"]["source_namespaces"] = ("candidate",)
    with pytest.raises(ValueError, match="complete recorded history"):
        BenchmarkRegistry.model_validate(data)


def test_model_copy_cannot_bypass_validation_at_persistence_boundary() -> None:
    # Revalidate the same immutable model at the service persistence boundary.
    broken = frozen().model_copy(update={"history": ()})
    with pytest.raises(ValueError, match=r"attestation|history identity"):
        BenchmarkRegistry.model_validate(broken)
