"""Measure sensitivity of automatic direction intervals on the width development corpus."""

import argparse
import json
import sys
from collections import Counter
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass, field, replace
from decimal import Decimal
from pathlib import Path

import numpy as np
from compare_profile_21 import Corpus
from compare_tcn_widths import DATASET_SHA256, POPULATIONS, PROFILE, generate

from regimex.data_catalog import SnapshotGrid, create_snapshot_member
from regimex.directions import DIRECTION_NAMES, DirectionAnnotations
from regimex.evaluation import IOU_THRESHOLD, dense_labels, matched_interval_indices
from regimex.segmentation import annotate

SHORTER_THAN = 64
TAIL = PROFILE.min_duration
MOVEMENT_THRESHOLDS = {"movement_0.45pct": 0.0045, "movement_0.55pct": 0.0055}


@dataclass
class IntervalCounts:
    total: int = 0
    matched: int = 0
    exact: int = 0
    majority_other_direction: int = 0
    perturbed_total: int = 0
    perturbed_unmatched: int = 0


@dataclass
class Agreement:
    intervals: dict[str, dict[str, IntervalCounts]] = field(
        default_factory=lambda: {
            name: {duration: IntervalCounts() for duration in ("short", "long")}
            for name in DIRECTION_NAMES
        }
    )
    observed_steps: int = 0
    same_direction_steps: int = 0
    availability_flips: int = 0

    def compare(
        self,
        original: DirectionAnnotations,
        perturbed: DirectionAnnotations,
        grid: SnapshotGrid,
        score_stop: int,
    ) -> None:
        for instrument in grid.instruments:
            expected = tuple(
                i for i in original.intervals if i.instrument_id == instrument.instrument_id
            )
            actual = tuple(
                i for i in perturbed.intervals if i.instrument_id == instrument.instrument_id
            )
            before = dense_labels(expected, score_stop)
            after = dense_labels(actual, score_stop)
            observed = before >= 0
            self.observed_steps += int(observed.sum())
            self.same_direction_steps += int(np.count_nonzero(observed & (before == after)))
            self.availability_flips += int(np.count_nonzero(observed != (after >= 0)))
            for label, name in enumerate(DIRECTION_NAMES):
                all_expected = [i for i in expected if i.direction == name]
                all_actual = [i for i in actual if i.direction == name]
                reference = [i for i in all_expected if i.stop <= score_stop]
                candidate = [i for i in all_actual if i.stop <= score_stop]
                # Count complete intervals on each side; a counterpart can span the guard.
                matches = matched_interval_indices(reference, all_actual)
                for index, interval in enumerate(reference):
                    duration = "short" if interval.stop - interval.start < SHORTER_THAN else "long"
                    counts = self.intervals[name][duration]
                    counts.total += 1
                    covered = after[interval.start : interval.stop]
                    other = int(np.count_nonzero((covered >= 0) & (covered != label)))
                    counts.majority_other_direction += 2 * other > len(covered)
                    if index in matches:
                        found = all_actual[matches[index]]
                        counts.matched += 1
                        counts.exact += (interval.start, interval.stop) == (found.start, found.stop)
                reverse_matches = matched_interval_indices(candidate, all_expected)
                for index, interval in enumerate(candidate):
                    duration = "short" if interval.stop - interval.start < SHORTER_THAN else "long"
                    counts = self.intervals[name][duration]
                    counts.perturbed_total += 1
                    counts.perturbed_unmatched += index not in reverse_matches


def quantize(grid: SnapshotGrid) -> tuple[SnapshotGrid, int, int]:
    """Coarsen published closes, preserving repeated prices and the complete time grid."""
    places = {i.instrument_id: i.units.price_decimal_places for i in grid.instruments}
    rows = []
    known = changed = 0
    for row in grid.observations:
        if row.last_price is not None:
            known += 1
            precision = places[row.instrument_id]
            ticks = Decimal(str(row.last_price)).scaleb(precision)
            even_ticks = 2 * round(ticks / 2)
            if even_ticks <= 0:
                raise ValueError("Rounding to the coarser price grid would produce zero")
            if even_ticks != ticks:
                changed += 1
                price = float(Decimal(even_ticks).scaleb(-precision))
                row = row.model_copy(update={"last_price": price})
        rows.append(row)
    return replace(grid, observations=tuple(rows)), changed, known


def audit(corpus: Corpus) -> dict:
    experiments = Counter()
    short_sideways = Counter()
    agreements = {
        name: Agreement() for name in ("two_tick_prices", "remove_tail", *MOVEMENT_THRESHOLDS)
    }
    known = changed = instruments = short_sideways_experiments = 0
    for member in corpus.dataset.members:
        if member.partition not in {"train", "validation"}:
            continue
        snapshot = corpus.snapshot(member)
        if create_snapshot_member(snapshot, member.partition) != member:
            raise ValueError("Snapshot differs from its reserved identity")
        grid = SnapshotGrid.from_export(snapshot.export)
        original = annotate(grid, PROFILE)
        experiments[member.partition] += 1
        instruments += len(grid.instruments)
        short_count = sum(
            i.direction == "sideways" and i.stop - i.start < SHORTER_THAN
            for i in original.intervals
        )
        short_sideways_experiments += short_count > 0
        short_sideways[member.partition] += short_count
        coarser, moved, available = quantize(grid)
        changed += moved
        known += available
        agreements["two_tick_prices"].compare(
            original, annotate(coarser, PROFILE), grid, grid.step_count
        )

        cut = grid.step_count - TAIL
        prefix = replace(
            grid,
            observations=tuple(row for row in grid.observations if row.step < cut),
            step_count=cut,
        )
        agreements["remove_tail"].compare(original, annotate(prefix, PROFILE), grid, cut - TAIL)
        for scenario, threshold in MOVEMENT_THRESHOLDS.items():
            config = PROFILE.model_copy(update={"minimum_relative_change": threshold})
            agreements[scenario].compare(original, annotate(grid, config), grid, grid.step_count)

    return {
        "dataset_sha256": corpus.dataset.content_sha256,
        "profile": PROFILE.model_dump(mode="json"),
        "experiments": dict(experiments),
        "instrument_sequences": instruments,
        "experiments_with_short_sideways": short_sideways_experiments,
        "short_sideways_intervals": dict(short_sideways),
        "rounded_known_closes": changed,
        "known_closes": known,
        "shorter_than_steps": SHORTER_THAN,
        "minimum_match_iou": IOU_THRESHOLD,
        "tail_removed_steps": TAIL,
        "guard_before_cut_steps": TAIL,
        "movement_thresholds": MOVEMENT_THRESHOLDS,
        "scenarios": {name: asdict(counts) for name, counts in agreements.items()},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path, help="Directory for the locked width comparison data.")
    parser.add_argument(
        "--generate", action="store_true", help="Create the corpus in a new directory."
    )
    args = parser.parse_args()
    if args.generate:
        args.corpus.mkdir(parents=True)
        with redirect_stdout(sys.stderr):
            corpus = generate(args.corpus, POPULATIONS, "tcn-width-comparison", DATASET_SHA256)
    else:
        corpus = Corpus(args.corpus, expected_digest=DATASET_SHA256)
    print(json.dumps(audit(corpus), indent=2))
