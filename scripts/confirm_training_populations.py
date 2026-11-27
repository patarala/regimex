"""Reserve a length-balanced cohort, then evaluate frozen training-population candidates."""

import argparse
from collections import Counter
from pathlib import Path
from statistics import mean

import numpy as np
from compare_profile_21 import RUNTIME, Corpus, record, sha256
from compare_tcn_slopes import ExposureIndex
from compare_tcn_widths import DATASET_SHA256, POLICY, PROFILE, generate
from compare_temporal_architectures import (
    DRAW_SEED,
    DRAWS,
    LENGTHS,
    evaluator,
    experiment_counts,
    paired,
    read,
    scientific_digests,
    summary,
)
from compare_training_populations import ARMS, per_seed_history, policy_for
from confirm_tcn_widths import audit_population
from pydantic import TypeAdapter

from regimex.admin_contracts import GenerationSpec, RawGenerationPlan
from regimex.backends.tcn import TCNBackend
from regimex.data_catalog import DatasetManifest, _load_partition, content_hash
from regimex.model_registry import TCNParameters
from regimex.storage import json_bytes, output_directory, write_new_file

PER_LENGTH = 64
HOLDOUT_SHA256 = "8743f7b0c9196a9400ff94cbb40d94d5e99329b87382438cf77f7844fa5bb441"
POPULATIONS = tuple(
    GenerationSpec(
        dataset_id=f"training-populations-final-{steps}",
        plan=RawGenerationPlan(
            seed=3489701557 + steps, step_count=steps, experiment_count=PER_LENGTH
        ),
    )
    for steps in LENGTHS
)
# Within each pair, the first arm is the reference for the paired difference.
CONTRASTS = {
    "length_mix_at_96": ("short-96", "mixed-96"),
    "length_mix_at_192": ("short-192", "mixed-192"),
    "population_size_at_short": ("short-96", "short-192"),
    "population_size_at_mixed": ("mixed-96", "mixed-192"),
    "update_budget_at_mixed_192": ("mixed-192", "mixed-192-double"),
    "exploratory_short_sideways_enrichment": ("mixed-96", "mixed-96-enriched"),
}


def protocol():
    return {
        "candidates": list(ARMS),
        "populations": [spec.model_dump(mode="json") for spec in POPULATIONS],
        "partitions": "All generated experiments belong to the holdout.",
        "profile": PROFILE.model_dump(mode="json"),
        "runtime": RUNTIME.model_dump(mode="json"),
        "selection": "Highest mean of seed-specific best validation scores; earliest arm on a tie.",
        "primary": ["pooled decoded interval macro-F1 (IoU >= 0.5)", "same F1 at 1024 steps"],
        "secondary": ["point macro-F1", "sideways recall below 64 steps", "sideways precision"],
        "strata": "32 one-instrument and 32 two-instrument experiments at each registered length.",
        "contrasts": CONTRASTS,
        "selected_recipe": "Compare the validation winner with short-96 if they differ.",
        "bootstrap": {
            "unit": "Whole experiments sampled with replacement within each sequence length.",
            "draws": DRAWS,
            "seed": DRAW_SEED,
            "interval": "Pointwise percentile 95%; conditional on frozen models.",
        },
    }


def environment():
    result = scientific_digests()
    for name in (
        "confirm_training_populations",
        "compare_training_populations",
        "plateau_sampling",
    ):
        path = Path(__file__).with_name(f"{name}.py")
        result["scripts"][path.name] = sha256(path)
    return result


def manifest(path: Path):
    return DatasetManifest.for_modeling(path.read_bytes())


def holdout_folder(output: Path) -> Path:
    return output / "confirmation" / "holdout"


def reserve(output: Path, original: Path, exposure: Path) -> None:
    confirmation = output / "confirmation"
    development = output / "development"
    if (output / "fitted").exists() or (confirmation / "reservation.json").exists():
        raise ValueError("A reservation must precede fitting and can be made only once")
    history = TypeAdapter(ExposureIndex).validate_json(exposure.read_bytes(), strict=True)
    if not history["excluded_seeds"] or not history["identity_keys_sha256"]:
        raise ValueError("Reservation requires an inventory of previous experiment exposures")

    old = Corpus(original, expected_digest=DATASET_SHA256).dataset
    extra = manifest(development / "augmentation" / "dataset.json")
    augmentation_audit = audit_population(old, extra, exposure)
    development_plan = read(development / "plan.json")
    if (
        set(development_plan["candidates"]) != set(ARMS)
        or tuple(development_plan["order"]) != ARMS
        or development_plan["development_sha256"] != old.content_sha256
        or development_plan["augmentation_sha256"] != extra.content_sha256
        or development_plan["exposure_audit"] != augmentation_audit
    ):
        raise ValueError("Development and confirmation must use the same registered inputs")

    holdout = holdout_folder(output)
    if not holdout.exists():
        with output_directory(holdout) as stage:
            generate(stage, POPULATIONS, "training-populations-final", HOLDOUT_SHA256)
    fresh = Corpus(holdout, expected_digest=HOLDOUT_SHA256).dataset
    expected = {(size, n): PER_LENGTH // 2 for size in LENGTHS for n in (1, 2)}
    if Counter((m.step_count, len(m.instrument_ids)) for m in fresh.members) != expected:
        raise ValueError("Unexpected holdout length or instrument counts")
    audit_population(old, fresh, exposure)
    audit = audit_population(extra, fresh, exposure)
    write_new_file(
        confirmation / "reservation.json",
        json_bytes(
            {
                "protocol": protocol(),
                "environment": environment(),
                "development_plan_sha256": sha256(development / "plan.json"),
                "original_dataset_sha256": old.content_sha256,
                "augmentation_sha256": extra.content_sha256,
                "locations_sha256": sha256(holdout / "locations.json"),
                "augmentation_audit": augmentation_audit,
                **audit,
            }
        ),
    )
    print(f"Reserved {len(fresh.members)} independent experiments: {fresh.content_sha256}")


def checked_reservation(output: Path):
    receipt = read(output / "confirmation" / "reservation.json")
    holdout = holdout_folder(output)
    development = output / "development"
    if (
        json_bytes(receipt["protocol"]) != json_bytes(protocol())
        or receipt["environment"] != environment()
        or receipt["dataset_sha256"] != HOLDOUT_SHA256
        or receipt["locations_sha256"] != sha256(holdout / "locations.json")
        or receipt["development_plan_sha256"] != sha256(development / "plan.json")
        or receipt["augmentation_sha256"]
        != manifest(development / "augmentation" / "dataset.json").content_sha256
    ):
        raise ValueError("The environment, dataset or protocol differs from the reservation")
    return receipt


def frozen_models(output: Path, reservation: dict):
    folder = output / "fitted"
    freeze = read(folder / "freeze.json")
    candidates = freeze["candidates"]
    if (
        freeze["reservation_sha256"] != sha256(output / "confirmation" / "reservation.json")
        or freeze["development_plan_sha256"] != reservation["development_plan_sha256"]
        or set(candidates) != set(ARMS)
    ):
        raise ValueError("The frozen comparison must bind every candidate to the reservation")
    plan = read(output / "development" / "plan.json")
    models = {}
    for name in ARMS:
        entry = candidates[name]
        if entry["bundle_path"] != f"{name}/bundle":
            raise ValueError(f"Unexpected bundle location for {name}")
        bundle = folder / entry["bundle_path"]
        model = TCNBackend().load_bundle(
            bundle, manifest_sha256=entry["bundle_sha256"], runtime=RUNTIME
        )
        metadata = read(bundle / "manifest.json")
        training = metadata["training"]
        scores = per_seed_history(bundle / "history.json", policy_for(name))
        best = max(scores, key=lambda row: row["interval_macro_f1"])
        if (
            metadata["model_version_id"] != name
            or entry["dataset_sha256"] != plan["candidates"][name]["dataset_sha256"]
            or training["dataset_sha256"] != entry["dataset_sha256"]
            or training["seeds"] != list(POLICY.seeds)
            or training["selected_seed"] != best["seed"]
            or training["validation_segment_macro_f1"] != best["interval_macro_f1"]
            or entry["mean_interval_macro_f1"] != mean(row["interval_macro_f1"] for row in scores)
            or entry["per_seed"] != scores
            or model.analysis != PROFILE
            or model.configuration.parameters != TCNParameters(channels=48)
        ):
            raise ValueError(f"The frozen {name} does not match its registered training recipe")
        models[name] = model
    winner = max(ARMS, key=lambda name: candidates[name]["mean_interval_macro_f1"])
    if winner != freeze["validation_choice"]:
        raise ValueError("The frozen choice disagrees with the selection rule")
    return models, winner


def select_length(group, length):
    ids, lengths, counts = group
    mask = lengths == length
    return [k for k, keep in zip(ids, mask, strict=True) if keep], lengths[mask], counts[mask]


def contrast(control, candidate):
    intervals = paired(control, candidate)
    a, b = summary(control[2]), summary(candidate[2])
    for key, entry in intervals.items():
        entry["difference"] = None if a[key] is None or b[key] is None else b[key] - a[key]
    return intervals


def score(output: Path) -> None:
    reservation = checked_reservation(output)
    models, winner = frozen_models(output, reservation)
    holdout = Corpus(holdout_folder(output), expected_digest=reservation["dataset_sha256"])
    samples = []
    digests = {}
    for partition in ("train", "validation", "test"):
        references, labels = holdout.references(partition, profile=PROFILE)
        digests[partition] = references.content_sha256
        samples.extend(
            _load_partition(
                holdout.dataset,
                references,
                snapshot_reader=holdout.snapshot,
                reference_reader=lambda _, m, annotations=labels: annotations[m.source_namespace],
            )
        )
    evaluation = evaluator(tuple(samples))
    expected = {
        content_hash([m.source_namespace, m.snapshot_id]): m.step_count
        for m in holdout.dataset.members
    }
    groups = {}
    results = {}
    for name, model in models.items():
        probabilities = model.predict(evaluation.inputs, runtime=RUNTIME)
        group = experiment_counts(evaluation, probabilities)
        ids, lengths, counts = group
        if dict(zip(ids, lengths.tolist(), strict=True)) != expected:
            raise ValueError(f"Experiment membership differs for {name}")
        groups[name] = group
        results[name] = {
            **summary(counts),
            "short_sideways_experiments": int(np.count_nonzero(counts[:, 10])),
            "by_length": {size: summary(counts[lengths == size]) for size in LENGTHS},
            "experiment_counts": dict(zip(ids, counts.tolist(), strict=True)),
        }
        print(name, "interval macro-F1", round(results[name]["interval_macro_f1"], 4), flush=True)
    comparisons = {}
    contrasts = dict(CONTRASTS)
    if winner != "short-96":
        contrasts["validation_choice_vs_original"] = ("short-96", winner)
    for name, (control, candidate) in contrasts.items():
        left, right = groups[control], groups[candidate]
        comparisons[name] = {
            "candidate": candidate,
            "control": control,
            "pooled": contrast(left, right),
            "1024": contrast(select_length(left, 1024), select_length(right, 1024)),
        }
    record(
        output / "confirmation" / "results.json",
        {
            "reservation_sha256": sha256(output / "confirmation" / "reservation.json"),
            "freeze_sha256": sha256(output / "fitted" / "freeze.json"),
            "dataset_sha256": holdout.dataset.content_sha256,
            "reference_digests_by_partition": digests,
            "validation_choice": winner,
            "experiments": len(samples),
            "instrument_sequences": len(evaluation.inputs.info),
            "results": results,
            "paired_comparisons": comparisons,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_subparsers(dest="phase", required=True)
    registration = phases.add_parser("reserve")
    registration.add_argument("output", type=Path, help="Root containing development/plan.json")
    registration.add_argument("--original", required=True, type=Path, help="TCN width corpus")
    registration.add_argument("--exposure-index", required=True, type=Path)
    phases.add_parser("score").add_argument("output", type=Path)
    args = parser.parse_args()
    if args.phase == "reserve":
        reserve(args.output, args.original, args.exposure_index)
    else:
        score(args.output)
