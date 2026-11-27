"""Repeat the gap and short-pause benchmark after changing the forest training population."""

import argparse
from pathlib import Path

import check_adaptive_gap_transfer as replay

from regimex.data_catalog import DatasetManifest
from regimex.directions import DIRECTION_NAMES
from regimex.evaluation import dense_labels
from regimex.segmentation import decode
from regimex.storage import digest, json_bytes, write_new_file

GROUPS = tuple(range(3_831_944_100, 3_831_944_132))


def check_seeds(previous, prior, native):
    if not set(previous).issubset(prior):
        raise ValueError("Prior exposure inventory omits a previously scored transfer group")
    if set(GROUPS).intersection(prior | native):
        raise ValueError("New transfer groups intersect historical, training or evaluation seeds")


def design(comparison, previous):
    from compare_forest_training_mix import checked_reservation

    receipt = checked_reservation(comparison)
    previous_plan = replay.read(previous / "plan.json")
    if json_bytes(previous_plan["protocol"]) != json_bytes(replay.protocol()):
        raise ValueError("Original gap generator, scoring rules or thresholds have changed")
    original_plan_sha = digest(previous / "plan.json")
    if replay.read(previous / "results.json")["plan_sha256"] != original_plan_sha:
        raise ValueError("Earlier transfer benchmark must have a result bound to its plan")
    prior = set(replay.read(comparison / "prior.json")["excluded_seeds"])
    combined = DatasetManifest.for_modeling((comparison / "dataset.json").read_bytes())
    training = {m.split_group.seed for m in combined.members if m.partition == "train"}
    native = set()
    for name, pool in receipt["pools"].items():
        dataset = DatasetManifest.for_modeling((comparison / name / "dataset.json").read_bytes())
        if dataset.content_sha256 != pool["dataset_sha256"]:
            raise ValueError("Native population differs from the reservation")
        native.update(member.split_group.seed for member in dataset.members)
    check_seeds(previous_plan["protocol"]["groups"], prior, native | training)
    return {
        "script_sha256": digest(Path(__file__)),
        "native_reservation_sha256": digest(comparison / "reservation.json"),
        "previous_plan_sha256": original_plan_sha,
        "previous_result_sha256": digest(previous / "results.json"),
        "protocol": {**previous_plan["protocol"], "groups": list(GROUPS)},
        "native_distinct_seeds": len(native),
        "training_distinct_seeds": len(training),
        "prior_seeds": len(prior),
    }


def reserve(output, comparison, previous):
    if (comparison / "freeze.json").exists():
        raise ValueError("Reserve the fresh transfer groups before selecting a model")
    write_new_file(output / "plan.json", json_bytes(design(comparison, previous)))


def evaluate_group(seed, index, predict, totals):
    cases = list(replay.pairs(seed, index))
    views = [view for full, masked, _ in cases for view in (full, masked)]
    expected = [replay.references(view) for view in views]
    scores = predict(replay.inputs_for(views))
    for method, counters in totals.items():
        probabilities = scores[method].reshape(len(views), replay.STEPS, len(DIRECTION_NAMES))
        for i, (full, masked, gap) in enumerate(cases):
            decoded = []
            for j, view in enumerate((full, masked), start=2 * i):
                intervals = decode(probabilities[j], view.rows, replay.PROFILE)
                counters.add_view(view, expected[j], intervals)
                decoded.append(dense_labels(intervals, replay.STEPS))
            for band, count in replay.changed_steps(*decoded, gap).items():
                counters.changes[band] += count
                counters.changes["pooled"] += count


def score(output, comparison, previous, bundle):
    from compare_forest_training_mix import CONTROL, load_predictors, read_freeze

    plan = output / "plan.json"
    if replay.read(plan) != design(comparison, previous):
        raise ValueError("Registered transfer population or recipe has changed")
    frozen = read_freeze(comparison)
    winner = frozen["winner"]
    if winner == CONTROL:
        raise ValueError("The release won validation; the transfer groups remain reserved")
    opening = {
        "plan_sha256": digest(plan),
        "freeze_sha256": digest(comparison / "freeze.json"),
        "winner": winner,
    }
    totals = {key: replay.Totals() for key in (CONTROL, winner)}
    predict = load_predictors(comparison, tuple(totals), bundle)
    write_new_file(output / "opened.json", json_bytes(opening))
    for index, seed in enumerate(GROUPS):
        evaluate_group(seed, index, predict, totals)
    results = {key: value.summary() for key, value in totals.items()}
    checks = replay.gate(results[CONTROL], results[winner])
    if (
        frozen != read_freeze(comparison)
        or digest(comparison / "freeze.json") != opening["freeze_sha256"]
        or digest(plan) != opening["plan_sha256"]
        or replay.read(plan) != design(comparison, previous)
    ):
        raise ValueError("Frozen model or transfer population changed during evaluation")
    result = {
        **opening,
        "groups": len(GROUPS),
        "paired_paths": len(GROUPS) * len(replay.SHAPES) * len(replay.MARKETS),
        "results": results,
        "checks": checks,
        "passed": checks["pooled"] and checks["stability"] and all(checks["per_class"].values()),
    }
    write_new_file(output / "results.json", json_bytes(result))
    print(f"{winner}: transfer checks passed={result['passed']}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_subparsers(dest="phase", required=True)
    for phase in ("reserve", "score"):
        command = phases.add_parser(phase)
        command.add_argument("output", type=Path)
        command.add_argument("comparison", type=Path)
        command.add_argument(
            "previous", type=Path, help="Scored adaptive interval transfer directory"
        )
        if phase == "score":
            command.add_argument("--control", type=Path, required=True)
    args = parser.parse_args()
    if args.phase == "reserve":
        reserve(args.output, args.comparison, args.previous)
    else:
        score(args.output, args.comparison, args.previous, args.control)
