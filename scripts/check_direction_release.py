"""Compare the frozen direction winner with the published tcn-0.2.0 on the closed native cohort."""

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
from compare_direction_context import (
    PROFILE,
    RUNTIME,
    checked_reservation,
    experiment_counts,
    open_pool,
    read,
    read_freeze,
    summary,
)
from confirm_tcn_widths import audit_population
from confirm_training_populations import contrast

from regimex.backends.tcn import TCNBackend
from regimex.data_catalog import DatasetManifest
from regimex.storage import digest, json_bytes, write_new_file

PUBLICATION = Path("var/model-finalization")
MANIFEST = "511222586d3d3410509a414d9a39d78a01a9d4a1c09edfc4987f401a6081f844"
EXPOSURE = "4b17e3bbed0d919d4c3908f164eda7d79eaf23d8d6dffe31dcaf9db81bb4d543"
MAX_CLASS_LOSS = 0.03


def protocol(comparison, publication):
    receipt = checked_reservation(comparison)
    data = publication / "data"
    bundle = publication / "publication/native-model/bundle"
    model_file = bundle / "manifest.json"
    exposure = data / "exposure-index.json"
    if digest(model_file) != MANIFEST or digest(exposure) != EXPOSURE:
        raise ValueError("Published model or its recorded exposure inventory changed")
    manifest = read(model_file)
    training = DatasetManifest.model_validate_json(
        (data / "training-512/dataset.json").read_bytes()
    )
    if training.content_sha256 != manifest["training"]["dataset_sha256"]:
        raise ValueError("Training membership differs from the published bundle")
    disjoint = {}
    for pool in ("validation", "holdout"):
        cohort = DatasetManifest.model_validate_json(
            (comparison / pool / "dataset.json").read_bytes()
        )
        if cohort.content_sha256 != receipt["pools"][pool]["dataset_sha256"]:
            raise ValueError("Direction cohort differs from the native reservation")
        disjoint[pool] = audit_population(training, cohort, exposure)
    return {
        "script_sha256": digest(Path(__file__)),
        "native_reservation_sha256": digest(comparison / "reservation.json"),
        "control": manifest["model_version_id"],
        "manifest_sha256": MANIFEST,
        "training_dataset_sha256": training.content_sha256,
        "disjoint_from_release_training_and_previous_exposures": disjoint,
        "reference": PROFILE.model_dump(mode="json"),
        "primary": "Frozen winner and published model with their saved decoders; IoU 0.5.",
        "diagnostic": "Both sets of probabilities decoded with the common reference profile.",
        "bootstrap": read(comparison / "plan.json")["bootstrap"],
        "gate": {
            "paired_interval_f1_lower_bound_above": 0.0,
            "maximum_class_f1_loss": MAX_CLASS_LOSS,
        },
    }


def reserve(comparison, publication):
    if (comparison / "opened.json").exists() or (comparison / "results.json").exists():
        raise ValueError("Reserve the published-model comparison before opening the native holdout")
    plan = protocol(comparison, publication)
    # Verify the installed loader and all release members before opening either final score.
    TCNBackend().load_bundle(
        publication / "publication/native-model/bundle", manifest_sha256=MANIFEST, runtime=RUNTIME
    )
    write_new_file(comparison / "published/plan.json", json_bytes(plan))


def winner_counts(primary, winner, control):
    ids, lengths, _ = control
    saved = primary["experiment_counts"][winner]
    if set(saved) != set(ids):
        raise ValueError("Cannot pair different experiment identities")
    rows = np.array([saved[i] for i in ids])
    if rows.shape != (len(ids), 20) or rows.dtype.kind != "i" or np.any(rows < 0):
        raise ValueError("Invalid closed winner counts")
    if any(primary["scores"][winner][key] != value for key, value in summary(rows).items()):
        raise ValueError("Winner counts disagree with its closed score")
    return ids, lengths, rows


def score(comparison, publication):
    folder = comparison / "published"
    plan = read(folder / "plan.json")
    if plan != protocol(comparison, publication):
        raise ValueError("Published-model comparison differs from its reservation")
    freeze = read_freeze(comparison)
    winner = freeze["winner"]
    bindings = {
        name: digest(comparison / f"{name}.json") for name in ("freeze", "opened", "results")
    }
    primary = read(comparison / "results.json")
    if (
        read(comparison / "opened.json")["freeze_sha256"] != bindings["freeze"]
        or primary["freeze_sha256"] != bindings["freeze"]
        or primary["winner"] != winner
        or primary["scores"][winner]["decoder_profile"] != PROFILE.version
    ):
        raise ValueError("Official native results differ from the frozen choice")
    bundle = publication / "publication/native-model/bundle"
    model = TCNBackend().load_bundle(bundle, manifest_sha256=MANIFEST, runtime=RUNTIME)
    evaluation = open_pool(comparison, checked_reservation(comparison), "holdout")
    probabilities = model.predict(evaluation.inputs, runtime=RUNTIME)
    common = experiment_counts(evaluation, probabilities)
    native = experiment_counts(replace(evaluation, analysis=model.analysis), probabilities)
    selected = winner_counts(primary, winner, native)
    native_score = summary(native[2])
    selected_score = summary(selected[2])
    paired = contrast(native, selected)
    checks = {
        "interval_gain": paired["interval_macro_f1"]["interval_95pct"][0] > 0.0,
        "per_class_f1": all(
            selected_score["interval_per_class"][name]["f1"] >= metrics["f1"] - MAX_CLASS_LOSS
            for name, metrics in native_score["interval_per_class"].items()
        ),
    }
    if any(digest(comparison / f"{name}.json") != value for name, value in bindings.items()):
        raise ValueError("Native evidence changed during replay")
    write_new_file(
        folder / "results.json",
        json_bytes(
            {
                "plan_sha256": digest(folder / "plan.json"),
                "primary_sha256": bindings,
                "winner": winner,
                "experiments": len(selected[0]),
                "winner_score": selected_score,
                "release_native": {"profile": model.analysis.version, **native_score},
                "release_common": {"profile": PROFILE.version, **summary(common[2])},
                "paired_vs_release_native": paired,
                "paired_vs_release_common": contrast(common, selected),
                "checks": checks,
                "qualified_against_published_release": all(checks.values()),
            }
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["reserve", "score"])
    parser.add_argument(
        "comparison", type=Path, help="Output directory of compare_direction_context.py"
    )
    parser.add_argument("--publication", type=Path, default=PUBLICATION)
    args = parser.parse_args()
    (reserve if args.phase == "reserve" else score)(args.comparison, args.publication)


if __name__ == "__main__":
    main()
