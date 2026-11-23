"""Measure automatic reference agreement with the known directions after the gap trial closes."""

import argparse
from pathlib import Path

import check_adaptive_gap_transfer as replay
from check_direction_context_transfer import INSTRUMENT

from regimex.data_catalog import SnapshotGrid
from regimex.evaluation import dense_labels
from regimex.segmentation import annotate
from regimex.storage import digest, json_bytes, write_new_file


def measure(directory):
    scores = directory / "results.json"
    plan = directory / "plan.json"
    if replay.read(scores)["plan_sha256"] != digest(plan) or json_bytes(
        replay.read(plan)["protocol"]
    ) != json_bytes(replay.protocol()):
        raise ValueError("Label comparison requires the closed original gap benchmark")
    totals = replay.Totals()
    for index, seed in enumerate(replay.GROUPS):
        for full, masked, gap in replay.pairs(seed, index):
            decoded = []
            for view in (full, masked):
                grid = SnapshotGrid("path", replay.STEPS, (INSTRUMENT,), view.rows)
                intervals = annotate(grid, replay.PROFILE).intervals
                totals.add_view(view, replay.references(view), intervals)
                decoded.append(dense_labels(intervals, replay.STEPS))
            for band, count in replay.changed_steps(*decoded, gap).items():
                totals.changes[band] += count
                totals.changes["pooled"] += count
    return {
        "closed_results_sha256": digest(scores),
        "script_sha256": digest(Path(__file__)),
        "profile": replay.PROFILE.version,
        "interpretation": "Post-hoc comparison of automatic labels with the scheduled drift.",
        "results": totals.summary(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Completed original gap transfer comparison")
    args = parser.parse_args()
    write_new_file(args.directory / "label-agreement.json", json_bytes(measure(args.directory)))
