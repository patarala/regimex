"""Measure pause refinements and their rounding stability on the mixed training cohort."""

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median

import numpy as np
from check_label_stability import SHORTER_THAN, quantize
from compare_profile_21 import Corpus
from compare_tcn_widths import PROFILE
from refine_trend_pauses import annotate_pauses

from regimex.data_catalog import SnapshotGrid
from regimex.directions import DIRECTION_NAMES, DirectionInterval
from regimex.evaluation import IOU_THRESHOLD, dense_labels, matched_interval_indices
from regimex.segmentation import annotate
from regimex.storage import digest, json_bytes, write_new_file

MIXED_192 = "4ff45ca06aa65f3a5fddebb4045ecbb69617f286d4702d5ec67106e42c62e8f7"


def inserted_short_pauses(original: list[DirectionInterval], revised: list[DirectionInterval]):
    matched = matched_interval_indices(revised, original)
    return [
        interval
        for index, interval in enumerate(revised)
        if interval.stop - interval.start < SHORTER_THAN and index not in matched
    ]


@dataclass
class Counts:
    experiments: int = 0
    sequences: int = 0
    changed_experiments: int = 0
    changed_sequences: int = 0
    baseline_short_sideways: int = 0
    refined_short_sideways: int = 0
    inserted_short: int = 0
    experiments_with_insertions: int = 0
    inserted_short_matched_after_rounding: int = 0
    inserted_short_exact_after_rounding: int = 0
    inserted_short_still_new_after_rounding: int = 0
    inserted_short_with_constant_close: int = 0
    inserted_short_with_zero_trades: int = 0
    rounded_known_closes: int = 0
    known_closes: int = 0


def unavailable(intervals):
    return [(i.start, i.stop, i.status) for i in intervals if i.direction is None]


def audit(corpus: Corpus):
    counts = Counts()
    statuses = Counter()
    insertions_by_length: Counter[int] = Counter()
    inserted_trade_activity: list[float] = []
    transitions = np.zeros((len(DIRECTION_NAMES), len(DIRECTION_NAMES)), dtype=np.int64)
    for member in corpus.dataset.members:
        if member.partition != "train":
            continue
        snapshot = corpus.snapshot(member)
        grid = SnapshotGrid.from_export(snapshot.export)
        coarser, changed, known = quantize(grid)
        counts.rounded_known_closes += changed
        counts.known_closes += known
        before, after = annotate(grid, PROFILE), annotate_pauses(grid, PROFILE)
        rounded_before, rounded_after = (
            annotate(coarser, PROFILE),
            annotate_pauses(coarser, PROFILE),
        )
        before.model_copy(update={"intervals": after.intervals}).validate_for(grid)
        rounded_before.model_copy(update={"intervals": rounded_after.intervals}).validate_for(
            coarser
        )
        counts.experiments += 1
        changed_experiment = inserted_experiment = False
        for instrument in grid.instruments:
            counts.sequences += 1
            series = [
                tuple(
                    i for i in annotation.intervals if i.instrument_id == instrument.instrument_id
                )
                for annotation in (before, after, rounded_before, rounded_after)
            ]
            gaps = unavailable(series[0])
            if any(unavailable(intervals) != gaps for intervals in series[1:]):
                raise ValueError("Unavailable interval boundaries or statuses changed")
            for interval in series[0]:
                statuses[interval.status] += interval.stop - interval.start
            left, right = (dense_labels(intervals, grid.step_count) for intervals in series[:2])
            observed = left >= 0
            np.add.at(transitions, (left[observed], right[observed]), 1)
            if not np.array_equal(left, right):
                counts.changed_sequences += 1
                changed_experiment = True

            flat, refined, coarse_flat, coarse_refined = (
                [i for i in intervals if i.direction == "sideways"] for intervals in series
            )
            counts.baseline_short_sideways += sum(i.stop - i.start < SHORTER_THAN for i in flat)
            counts.refined_short_sideways += sum(i.stop - i.start < SHORTER_THAN for i in refined)
            inserted = inserted_short_pauses(flat, refined)
            inserted_experiment |= bool(inserted)
            counts.inserted_short += len(inserted)
            insertions_by_length[grid.step_count] += len(inserted)
            if inserted:
                rows = [r for r in grid.observations if r.instrument_id == instrument.instrument_id]
                for pause in inserted:
                    closes = {
                        r.last_price
                        for r in rows[pause.start : pause.stop]
                        if r.last_price is not None
                    }
                    counts.inserted_short_with_constant_close += len(closes) == 1
                    counts.inserted_short_with_zero_trades += pause.statistics.trade_count == 0
                    if pause.statistics.trade_active_fraction is not None:
                        inserted_trade_activity.append(pause.statistics.trade_active_fraction)
            stable = matched_interval_indices(inserted, coarse_refined)
            counts.inserted_short_matched_after_rounding += len(stable)
            counts.inserted_short_exact_after_rounding += sum(
                (inserted[i].start, inserted[i].stop)
                == (coarse_refined[j].start, coarse_refined[j].stop)
                for i, j in stable.items()
            )
            coarse_inserted = inserted_short_pauses(coarse_flat, coarse_refined)
            counts.inserted_short_still_new_after_rounding += len(
                matched_interval_indices(inserted, coarse_inserted)
            )
        counts.changed_experiments += changed_experiment
        counts.experiments_with_insertions += inserted_experiment
    return {
        "dataset_sha256": corpus.dataset.content_sha256,
        "detector_sha256": digest(Path(__file__).with_name("refine_trend_pauses.py")),
        "analysis_profile": PROFILE.model_dump(mode="json"),
        "partition": "train",
        "minimum_match_iou": IOU_THRESHOLD,
        "shorter_than_steps": SHORTER_THAN,
        **asdict(counts),
        "short_insertions_by_experiment_length": dict(sorted(insertions_by_length.items())),
        "median_trade_active_fraction_inserted": median(inserted_trade_activity)
        if inserted_trade_activity
        else None,
        "steps_by_status": dict(statuses),
        "direction_order": DIRECTION_NAMES,
        "step_transitions_original_to_refined": transitions.tolist(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "corpus", type=Path, help="Mixed 192-experiment development manifest directory"
    )
    parser.add_argument("output", type=Path, help="New file for the summary")
    args = parser.parse_args()
    result = audit(Corpus(args.corpus, expected_digest=MIXED_192))
    write_new_file(args.output, json_bytes(result))
    print(json.dumps(result, indent=2))
