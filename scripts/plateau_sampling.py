"""Enrich a training cohort with observed short sideways intervals at a fixed size."""

from collections import defaultdict
from collections.abc import Callable, Sequence

from regimex.data_catalog import (
    SnapshotGrid,
    SnapshotMember,
    SourceSnapshot,
    create_snapshot_member,
)
from regimex.directions import AnalysisConfig, Direction
from regimex.segmentation import annotate

PROFILE = AnalysisConfig(version="2.2")
SHORT_STEPS = 64


def _seed(member: SnapshotMember) -> int:
    seed = member.split_group.seed
    if seed is None or member.partition != "train":
        raise ValueError("Mining requires native training members only")
    return seed


def _swap_within_strata(
    uniform: Sequence[SnapshotMember],
    pool: Sequence[SnapshotMember],
    short_counts: dict[int, int],
) -> tuple[tuple[SnapshotMember, ...], list[dict[str, int]]]:
    retained = {_seed(member) for member in uniform}
    strata: dict[tuple[int, int], list[SnapshotMember]] = defaultdict(list)
    for member in pool:
        strata[(member.step_count, len(member.instrument_ids))].append(member)

    replacements: dict[int, SnapshotMember] = {}
    audit = []
    for (steps, instruments), stratum in sorted(strata.items()):
        # Ordering by native seed makes the tie break independent of labeling and file order.
        ordered = sorted(stratum, key=_seed)
        original = [member for member in ordered if _seed(member) in retained]
        negatives = [member for member in original if not short_counts[_seed(member)]]
        positives = [
            member
            for member in ordered
            if _seed(member) not in retained and short_counts[_seed(member)]
        ]
        positive_count = len(original) - len(negatives)
        swaps = min(len(positives), max(0, len(original) // 2 - positive_count))
        for outgoing, incoming in zip(negatives[:swaps], positives[:swaps], strict=True):
            replacements[_seed(outgoing)] = incoming
        enriched = [replacements.get(_seed(member), member) for member in original]
        audit.append(
            {
                "steps": steps,
                "instruments": instruments,
                "experiments": len(original),
                "pool_experiments": len(stratum),
                "swaps": swaps,
                "positive_before": positive_count,
                "positive_after": positive_count + swaps,
                "intervals_before": sum(short_counts[_seed(m)] for m in original),
                "intervals_after": sum(short_counts[_seed(m)] for m in enriched),
            }
        )
    return tuple(replacements.get(_seed(m), m) for m in uniform), audit


def enrich_short_sideways(
    uniform: Sequence[SnapshotMember],
    pool: Sequence[SnapshotMember],
    snapshot_reader: Callable[[SnapshotMember], SourceSnapshot],
) -> tuple[tuple[SnapshotMember, ...], dict[str, object]]:
    """Retain the length and instrument mix, using annotations from training members only."""
    members = {_seed(m): m for m in pool}
    if not uniform or len(members) != len(pool) or len({_seed(m) for m in uniform}) != len(uniform):
        raise ValueError("Training cohorts require distinct experiments and a nonempty selection")
    if any(members.get(_seed(m)) != m for m in uniform):
        raise ValueError("The initial cohort must be a subset of the independent training pool")

    short_counts = {}
    for member in pool:
        source = snapshot_reader(member)
        if create_snapshot_member(source, "train") != member:
            raise ValueError("A training snapshot differs from its registered identity")
        labels = annotate(SnapshotGrid.from_export(source.export), PROFILE)
        short_counts[_seed(member)] = sum(
            interval.direction == Direction.SIDEWAYS
            and interval.stop - interval.start < SHORT_STEPS
            for interval in labels.intervals
        )
    selected, strata = _swap_within_strata(uniform, pool, short_counts)
    return selected, {"max_steps_exclusive": SHORT_STEPS, "strata": strata}
