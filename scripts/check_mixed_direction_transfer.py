"""Evaluate mixed-supervision forests against drift specified before generating prices."""

import argparse
from pathlib import Path

import check_adaptive_gap_transfer as replay
from check_forest_training_transfer import evaluate_group
from compare_adaptive_intervals import BUNDLE, BUNDLE_SHA, CONTROL
from compare_profile_21 import RUNTIME

from regimex.backends.adaptive import AdaptiveBackend
from regimex.data_catalog import DatasetManifest
from regimex.storage import digest, json_bytes, write_new_file

GROUPS = {
    "validation": range(3_961_177_100, 3_961_177_132),
    "test": range(3_961_177_200, 3_961_177_264),
}
NATIVE_MODEL = "multiscale-balanced"
NATIVE_SHA = "e60f2a3f4de68436a396e904b1781492df94de5531f1c9b5bf7d55989e6d5495"
MIXED = ("mixed-10", "mixed-25", "mixed-45")
DATA = Path("var/verification/direction-mixed-v2")
COMPARISON = Path("var/verification/direction-mixed-evaluation-v2")
TRANSFER = Path("var/verification/direction-mixed-transfer-v2")
NATIVE_BUNDLE = Path("var/release/direction-large-v1/bundle")


def protocol():
    return {
        "protocol": {**replay.protocol(), "groups": {k: list(v) for k, v in GROUPS.items()}},
        "scoring_sha256": digest(Path(__file__).with_name("check_forest_training_transfer.py")),
        "control_bundle_sha256": BUNDLE_SHA,
        "native_only_bundle_sha256": NATIVE_SHA,
        "validation_models": [CONTROL, NATIVE_MODEL, *MIXED],
        "final_models": "Release and frozen selection from joint validation only.",
    }


def score(phase, output, data, comparison, control, native_bundle):
    plan = output / "plan.json"
    selection = comparison / "selection"
    if plan.read_bytes() != json_bytes(protocol()):
        raise ValueError("Drift benchmark differs from its registered design")
    dataset = DatasetManifest.model_validate_json((data / "dataset.json").read_bytes())
    seeds = set(GROUPS["validation"]) | set(GROUPS["test"])
    if seeds.intersection(m.split_group.seed for m in dataset.members):
        raise ValueError("Controlled and native evaluation populations overlap")
    prior = data / "prior.json"
    if digest(prior) != replay.read(data / "reservation.json")[
        "prior_sha256"
    ] or not seeds.issubset(replay.read(prior)["excluded_seeds"]):
        raise ValueError("Controlled benchmark groups are missing from the native reservation")
    opening = {"plan_sha256": digest(plan), "dataset_sha256": dataset.content_sha256}
    if phase == "validation":
        from evaluate_mixed_direction import load_validation_models

        if selection.exists():
            raise ValueError("Joint validation has already selected a model")
        models = load_validation_models(data, control, native_bundle)
        if set(models) != {CONTROL, NATIVE_MODEL, *MIXED}:
            raise ValueError("Unexpected validation candidates")
        weights = {name: data / "fitted" / f"{name}.npz" for name in MIXED}
        weights.update(
            {CONTROL: control / "weights.npz", NATIVE_MODEL: native_bundle / "weights.npz"}
        )
        if digest(native_bundle / "manifest.json") != NATIVE_SHA:
            raise ValueError("Unexpected native-only bundle")
    else:
        freeze = selection / "freeze.json"
        frozen = replay.read(freeze)
        name = frozen["challenger"]
        if name not in (NATIVE_MODEL, *MIXED) or frozen["control_bundle_sha256"] != BUNDLE_SHA:
            raise ValueError("Unexpected final comparison")
        backend = AdaptiveBackend()
        models = {
            CONTROL: backend.load_bundle(control, manifest_sha256=BUNDLE_SHA, runtime=RUNTIME),
            name: backend.load_bundle(
                selection / "bundle", manifest_sha256=frozen["bundle_sha256"], runtime=RUNTIME
            ),
        }
        weights = {CONTROL: control / "weights.npz", name: selection / "bundle" / "weights.npz"}
        opening.update(freeze_sha256=digest(freeze), challenger=name)
    opening["weights_sha256"] = {k: digest(path) for k, path in weights.items()}
    opened = output / f"{phase}-opened.json"
    destination = output / f"{phase}.json"
    if opened.exists():
        if replay.read(opened) != opening or destination.exists():
            raise ValueError("Benchmark already scored or bound to different weights")
    else:
        write_new_file(opened, json_bytes(opening))

    def predict(inputs):
        return {key: model.predict(inputs, runtime=RUNTIME) for key, model in models.items()}

    totals = {name: replay.Totals() for name in models}
    groups = GROUPS[phase]
    for index, seed in enumerate(groups):
        evaluate_group(seed, index, predict, totals)
        if (index + 1) % 16 == 0:
            print(f"Scored {index + 1}/{len(groups)} {phase} groups", flush=True)
    results = {name: counters.summary() for name, counters in totals.items()}
    checks = {k: replay.gate(results[CONTROL], v) for k, v in results.items() if k != CONTROL}
    if opening["weights_sha256"] != {k: digest(path) for k, path in weights.items()}:
        raise ValueError("Model weights changed during scoring")
    if digest(plan) != opening["plan_sha256"] or (
        phase == "test" and digest(selection / "freeze.json") != opening["freeze_sha256"]
    ):
        raise ValueError("Registered comparison changed during scoring")
    write_new_file(
        destination,
        json_bytes(
            {
                **opening,
                "groups": len(groups),
                "paired_paths": len(groups) * len(replay.SHAPES) * len(replay.MARKETS),
                "results": results,
                "checks": checks,
            }
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("reserve", "validation", "test"))
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--comparison", type=Path, default=COMPARISON)
    parser.add_argument("--output", type=Path, default=TRANSFER)
    parser.add_argument("--control", type=Path, default=BUNDLE)
    parser.add_argument("--native-bundle", type=Path, default=NATIVE_BUNDLE)
    args = parser.parse_args()
    if args.phase == "reserve":
        if (args.comparison / "selection").exists():
            raise ValueError("Register both populations before model selection")
        write_new_file(args.output / "plan.json", json_bytes(protocol()))
    else:
        score(args.phase, args.output, args.data, args.comparison, args.control, args.native_bundle)
