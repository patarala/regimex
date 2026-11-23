"""Reserve and score independent drift paths for fixed probability blends."""

import argparse
from pathlib import Path

import check_adaptive_gap_transfer as replay
from check_mixed_direction_transfer import NATIVE_BUNDLE, NATIVE_MODEL, NATIVE_SHA, measure
from compare_adaptive_intervals import BUNDLE, BUNDLE_SHA, CONTROL
from compare_profile_21 import RUNTIME

from regimex.backends.adaptive import AdaptiveBackend
from regimex.data_catalog import DatasetManifest
from regimex.storage import digest, json_bytes, write_new_file
from regimex.terminal_training import register_frozen_selection

GROUPS = {
    "validation": range(3_961_177_300, 3_961_177_332),
    "test": range(3_961_177_400, 3_961_177_464),
}
FRACTIONS = {"blend-125": 0.125, "blend-250": 0.25, "blend-375": 0.375, "blend-500": 0.5}
PARENTS = {CONTROL: BUNDLE_SHA, NATIVE_MODEL: NATIVE_SHA}


def design():
    return {
        "protocol": {**replay.protocol(), "groups": {k: list(v) for k, v in GROUPS.items()}},
        "scoring_sha256": digest(Path(__file__).with_name("check_forest_training_transfer.py")),
        "fractions": FRACTIONS,
        "parents": PARENTS,
        "final_models": "Release and the selected blend, loaded from their frozen bundles.",
    }


def checked_population(data):
    dataset = DatasetManifest.model_validate_json((data / "dataset.json").read_bytes())
    receipt = data / "reservation.json"
    registration = replay.read(receipt)
    prior = data / "prior.json"
    groups = set(GROUPS["validation"]) | set(GROUPS["test"])
    if (
        registration["dataset_sha256"] != dataset.content_sha256
        or digest(prior) != registration["prior_sha256"]
        or not groups.issubset(replay.read(prior)["excluded_seeds"])
        or groups.intersection(m.split_group.seed for m in dataset.members)
    ):
        raise ValueError("The drift groups and native data do not match their reservations")
    return dataset.content_sha256, digest(receipt)


def bind_validation(output, data):
    """Link the independent drift evaluation to the native cohort checked for overlap."""
    dataset_sha, receipt_sha = checked_population(data)
    results = output / "validation.json"
    plan_sha = digest(output / "plan.json")
    if replay.read(results)["plan_sha256"] != plan_sha:
        raise ValueError("The drift validation result belongs to another population")
    evidence = {
        "dataset_sha256": dataset_sha,
        "reservation_sha256": receipt_sha,
        "plan_sha256": plan_sha,
        "validation_sha256": digest(results),
    }
    binding = output / "population-binding.json"
    if binding.exists():
        if replay.read(binding) != evidence:
            raise ValueError("Registered native population binding has changed")
    else:
        write_new_file(binding, json_bytes(evidence))


def score(phase, output, data, comparison, control, native_bundle):
    plan = output / "plan.json"
    if plan.read_bytes() != json_bytes(design()):
        raise ValueError("Drift benchmark differs from the registered design")
    checked_population(data)
    binding = {"plan_sha256": digest(plan), "parents": PARENTS, "fractions": FRACTIONS}
    selection = comparison / "selection"
    if phase == "validation":
        from evaluate_ensemble_direction import validation_probabilities

        if selection.exists():
            raise ValueError("Model selection is already closed")
        predict = validation_probabilities(control, native_bundle)
        names = (*PARENTS, *FRACTIONS)
    else:
        freeze = selection / "freeze.json"
        frozen = replay.read(freeze)
        name = frozen["challenger"]
        if name not in FRACTIONS or frozen["control_bundle_sha256"] != BUNDLE_SHA:
            raise ValueError("The frozen comparison does not match the registered candidates")
        backend = AdaptiveBackend()
        models = {
            CONTROL: backend.load_bundle(control, manifest_sha256=BUNDLE_SHA, runtime=RUNTIME),
            name: backend.load_bundle(
                selection / "bundle", manifest_sha256=frozen["bundle_sha256"], runtime=RUNTIME
            ),
        }
        names = tuple(models)
        binding.update(freeze_sha256=digest(freeze), challenger=name)

        def predict(inputs):
            return {key: model.predict(inputs, runtime=RUNTIME) for key, model in models.items()}

    opened = output / f"{phase}-opened.json"
    result = output / f"{phase}.json"
    if opened.exists():
        if replay.read(opened) != binding or result.exists():
            raise ValueError("Drift benchmark completed or belongs to a different freeze")
    else:
        write_new_file(opened, json_bytes(binding))
    groups = GROUPS[phase]
    results = measure(groups, names, predict)
    if digest(plan) != binding["plan_sha256"] or (
        phase == "test" and digest(selection / "freeze.json") != binding["freeze_sha256"]
    ):
        raise ValueError("Registered comparison changed during scoring")
    checks = {
        key: replay.gate(results[CONTROL], val) for key, val in results.items() if key != CONTROL
    }
    write_new_file(
        result,
        json_bytes(
            {
                **binding,
                "groups": len(groups),
                "paired_paths": len(groups) * len(replay.SHAPES) * len(replay.MARKETS),
                "results": results,
                "checks": checks,
            }
        ),
    )

    if phase == "test":
        register_frozen_selection(comparison, controlled_final=result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("reserve", "validation", "bind", "test"))
    parser.add_argument("--data", type=Path, default=Path("var/verification/direction-ensemble-v3"))
    parser.add_argument(
        "--comparison", type=Path, default=Path("var/verification/direction-ensemble-evaluation-v3")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("var/verification/direction-ensemble-transfer-v3")
    )
    parser.add_argument("--control", type=Path, default=BUNDLE)
    parser.add_argument("--native-bundle", type=Path, default=NATIVE_BUNDLE)
    args = parser.parse_args()
    if args.phase == "reserve":
        if (args.comparison / "selection").exists():
            raise ValueError("Register transfer groups before selection")
        write_new_file(args.output / "plan.json", json_bytes(design()))
    elif args.phase == "bind":
        bind_validation(args.output, args.data)
    else:
        score(args.phase, args.output, args.data, args.comparison, args.control, args.native_bundle)
        if args.phase == "validation":
            bind_validation(args.output, args.data)
