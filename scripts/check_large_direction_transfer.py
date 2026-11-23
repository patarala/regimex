"""Compare two frozen bundles on paired complete and incomplete price paths."""

import argparse
from pathlib import Path

import check_adaptive_gap_transfer as replay
from check_forest_training_transfer import evaluate_group
from compare_adaptive_intervals import BUNDLE, BUNDLE_SHA, CONTROL
from compare_profile_21 import RUNTIME
from evaluate_large_direction import DATA, OUTPUT
from prepare_large_direction import TRANSFER_SEEDS

from regimex.backends.adaptive import AdaptiveBackend
from regimex.data_catalog import DatasetManifest
from regimex.storage import digest, json_bytes, write_new_file
from regimex.terminal_training import register_frozen_selection

GROUPS = tuple(TRANSFER_SEEDS)
TRANSFER = Path("var/verification/large-direction-transfer-v1")


def design():
    return {
        "protocol": {**replay.protocol(), "groups": list(GROUPS)},
        "scoring_sha256": digest(Path(__file__).with_name("check_forest_training_transfer.py")),
        "control_bundle_sha256": BUNDLE_SHA,
        "selection_rule": "Compare only the highest-scoring new model on native validation.",
    }


def reserve(output, comparison):
    if (comparison / "selection").exists():
        raise ValueError("Reserve transfer groups before validation selection")
    write_new_file(output / "plan.json", json_bytes(design()))


def score(output, comparison, data, control):
    plan = output / "plan.json"
    if plan.read_bytes() != json_bytes(design()):
        raise ValueError("Transfer generator or scoring rules differ from the reservation")
    dataset = DatasetManifest.model_validate_json((data / "dataset.json").read_bytes())
    if set(GROUPS).intersection(member.split_group.seed for member in dataset.members):
        raise ValueError("Transfer groups overlap the native population")
    prior = data / "prior.json"
    if digest(prior) != replay.read(data / "reservation.json")["prior_sha256"] or not set(
        GROUPS
    ).issubset(replay.read(prior)["excluded_seeds"]):
        raise ValueError("Transfer groups were not reserved in the native exposure inventory")
    freeze = comparison / "selection" / "freeze.json"
    frozen = replay.read(freeze)
    if frozen["control_bundle_sha256"] != BUNDLE_SHA:
        raise ValueError("The selected comparison uses a different release")
    name = frozen["challenger"]
    backend = AdaptiveBackend()
    models = {
        CONTROL: backend.load_bundle(control, manifest_sha256=BUNDLE_SHA, runtime=RUNTIME),
        name: backend.load_bundle(
            freeze.parent / "bundle", manifest_sha256=frozen["bundle_sha256"], runtime=RUNTIME
        ),
    }
    opening = {
        "freeze_sha256": digest(freeze),
        "plan_sha256": digest(plan),
        "dataset_sha256": dataset.content_sha256,
        "challenger": name,
        "bundle_sha256": frozen["bundle_sha256"],
    }
    opened = output / "opened.json"
    if opened.exists():
        if replay.read(opened) != opening or (output / "results.json").exists():
            raise ValueError("Transfer scoring is complete or belongs to a different selection")
    else:
        write_new_file(opened, json_bytes(opening))

    def predict(inputs):
        return {key: model.predict(inputs, runtime=RUNTIME) for key, model in models.items()}

    totals = {key: replay.Totals() for key in models}
    for index, seed in enumerate(GROUPS):
        evaluate_group(seed, index, predict, totals)
    results = {key: counters.summary() for key, counters in totals.items()}
    checks = replay.gate(results[CONTROL], results[name])
    if digest(freeze) != opening["freeze_sha256"] or digest(plan) != opening["plan_sha256"]:
        raise ValueError("Selection or transfer plan changed during scoring")
    result = {
        **opening,
        "groups": len(GROUPS),
        "paired_paths": len(GROUPS) * len(replay.SHAPES) * len(replay.MARKETS),
        "results": results,
        "checks": checks,
        "passed": checks["pooled"] and checks["stability"] and all(checks["per_class"].values()),
    }
    write_new_file(output / "results.json", json_bytes(result))
    register_frozen_selection(comparison, controlled_final=output / "results.json")
    print(f"{name} passed transfer checks: {result['passed']}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("reserve", "score"))
    parser.add_argument("--output", type=Path, default=TRANSFER)
    parser.add_argument("--comparison", type=Path, default=OUTPUT)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--control", type=Path, default=BUNDLE)
    args = parser.parse_args()
    if args.command == "reserve":
        reserve(args.output, args.comparison)
    else:
        score(args.output, args.comparison, args.data, args.control)
