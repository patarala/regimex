"""Confirm a change in TCN training targets against original and refined automatic references."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from build_exposure_index import build
from compare_profile_21 import RUNTIME, Corpus, record, sha256, write
from compare_tcn_slopes import metrics
from compare_tcn_widths import PROFILE, generate
from compare_temporal_architectures import DRAW_SEED, DRAWS, LENGTHS, read, scientific_digests
from confirm_tcn_widths import audit_population
from confirm_training_populations import manifest

from regimex.admin_contracts import GenerationSpec, RawGenerationPlan
from regimex.application.analysis_computation import snapshot_inputs
from regimex.data_catalog import SnapshotGrid, content_hash
from regimex.evaluation import _segment_counts, matched_interval_indices
from regimex.segmentation import annotate, decode
from regimex.storage import output_directory

PER_LENGTH = 40
POPULATIONS = tuple(
    GenerationSpec(
        dataset_id=f"pause-targets-final-{length}",
        plan=RawGenerationPlan(
            seed=4039162000 + length, experiment_count=PER_LENGTH, step_count=length
        ),
    )
    for length in LENGTHS
)


def protocol():
    return {
        "populations": [s.model_dump(mode="json") for s in POPULATIONS],
        "members": "All members of each generated manifest are final data.",
        "predictors": ["original", "refined"],
        "primary": (
            "Fraction of new refined reference sideways intervals "
            "under 64 steps matched by the model."
        ),
        "new": (
            "Unmatched by original sideways intervals in "
            "one-to-one IoU >= 0.5 matching per instrument."
        ),
        "prediction_matches": (
            "Match all refined sideways references to "
            "predictions once before counting new references."
        ),
        "secondary": [
            "pooled interval macro-F1",
            "short sideways recall",
            "all-sideways precision",
            "false sideways intervals",
        ],
        "reference_policies": ["original", "refined"],
        "paired_differences": [
            "refined minus original predictor on each policy's interval macro-F1",
            "new short sideways recall",
        ],
        "resampling": {
            "unit": "whole experiments stratified by length",
            "draws": DRAWS,
            "seed": DRAW_SEED,
            "interval": "paired pointwise percentile 95%",
        },
        "profile": PROFILE.model_dump(mode="json"),
        "inference": RUNTIME.model_dump(mode="json"),
    }


def prepare(output, panels):
    with output_directory(output) as stage:
        receipt = read(panels / "reservation.json")
        manifests = [panels / name / "dataset.json" for name in ("validation", "holdout")]
        for name, path in zip(("validation", "holdout"), manifests, strict=True):
            if manifest(path).content_sha256 != receipt["pools"][name]["dataset_sha256"]:
                raise ValueError("Historical selection panel differs from its reservation")
        prior = panels / "prior.json"
        if sha256(prior) != receipt["pools"]["holdout"]["external_inventory_sha256"]:
            raise ValueError("Historical inventory differs from its selection reservation")
        write(stage / "plan.json", protocol())
        index = stage / "prior.json"
        write(index, build(manifests, [prior]))
        holdout = stage / "holdout"
        holdout.mkdir()
        fresh = generate(holdout, POPULATIONS, "pause-targets-final")
        counts = Counter((m.step_count, len(m.instrument_ids)) for m in fresh.dataset.members)
        if counts != {(n, k): PER_LENGTH // 2 for n in LENGTHS for k in (1, 2)}:
            raise ValueError("Confirmation length and instrument strata differ from the plan")
        audit = audit_population(manifest(manifests[-1]), fresh.dataset, index)
        write(
            stage / "population.json",
            {
                **audit,
                "plan_sha256": sha256(stage / "plan.json"),
                "locations_sha256": sha256(holdout / "locations.json"),
                "selection_reservation_sha256": sha256(panels / "reservation.json"),
            },
        )


def reserve(output, populations):
    from fit_pause_targets import training_protocol

    if (output / "fitted").exists() or (output / "reservation.json").exists():
        raise ValueError("Reservation must precede training")
    population = read(output / "population.json")
    if population["plan_sha256"] != sha256(output / "plan.json") or population[
        "plan_sha256"
    ] != content_hash(protocol()):
        raise ValueError("Prepared confirmation differs from the protocol")
    record(
        output / "reservation.json",
        {
            **population,
            "training_protocol": training_protocol(),
            "evaluator_sha256": sha256(Path(__file__)),
            "training_script_sha256": sha256(Path(__file__).with_name("fit_pause_targets.py")),
            "environment": scientific_digests(),
            "control_sha256": read(populations / "fitted" / "freeze.json")["candidates"][
                "mixed-192"
            ]["bundle_sha256"],
        },
    )


def policy_counts(reference, prediction):
    flat = [i for i in reference if i.direction == "sideways"]
    matches = matched_interval_indices(flat, [i for i in prediction if i.direction == "sideways"])
    short = {k for k, interval in enumerate(flat) if interval.stop - interval.start < 64}
    counts = _segment_counts(reference, prediction).ravel().tolist()
    return [*counts, len(short.intersection(matches)), len(short)], flat, short, matches


def sequence_counts(original, refined, predicted):
    original_counts, original_flat, _, _ = policy_counts(original, predicted)
    refined_counts, refined_flat, short, matches = policy_counts(refined, predicted)
    # All refined references compete in the same assignment. A predicted interval
    # already paired with an existing plateau cannot also claim a new plateau.
    inserted = short.difference(matched_interval_indices(refined_flat, original_flat))
    return np.array(
        [*original_counts, *refined_counts, len(inserted.intersection(matches)), len(inserted)],
        dtype=np.int64,
    )


def evaluate(corpus, models):
    from refine_trend_pauses import annotate_pauses

    records = []
    for member in corpus.dataset.members:
        source = corpus.snapshot(member)
        grid = SnapshotGrid.from_export(source.export)
        original, refined = annotate(grid, PROFILE), annotate_pauses(grid, PROFILE)
        inputs = snapshot_inputs(source.export)
        counts = {}
        for name, model in models.items():
            probabilities = model.predict(inputs, runtime=RUNTIME)
            sequences = probabilities.reshape(len(inputs.info), grid.step_count, 3)
            group = np.zeros(24, dtype=np.int64)
            for instrument, rows, scores in zip(
                grid.instruments, inputs.observations, sequences, strict=True
            ):
                references = [
                    tuple(i for i in a.intervals if i.instrument_id == instrument.instrument_id)
                    for a in (original, refined)
                ]
                group += sequence_counts(*references, decode(scores, rows, PROFILE))
            counts[name] = group.tolist()
        records.append(
            {
                "experiment": content_hash([member.source_namespace, member.snapshot_id]),
                "steps": member.step_count,
                "counts": counts,
            }
        )
    return records


def summaries(totals):
    return {
        "original_references": metrics(totals[:11]),
        "refined_references": metrics(totals[11:22]),
        "new_pause_matched": int(totals[22]),
        "new_pause_references": int(totals[23]),
        "new_pause_recall": float(totals[22] / totals[23]) if totals[23] else None,
    }


def differences(left, right):
    deltas = {
        key: right[key]["interval_macro_f1"] - left[key]["interval_macro_f1"]
        for key in ("original_references", "refined_references")
    }
    deltas["new_pause_recall"] = (
        None
        if left["new_pause_recall"] is None
        else right["new_pause_recall"] - left["new_pause_recall"]
    )
    return deltas


def score_groups(records):
    arrays = [
        np.array([r["counts"][name] for r in records], dtype=np.int64)
        for name in ("original", "refined")
    ]
    left, right = arrays
    if not np.array_equal(left[:, 23], right[:, 23]):
        raise ValueError("Models received different new-pause references")
    a, b = (summaries(x.sum(axis=0)) for x in arrays)
    deltas = differences(a, b)
    lengths = np.array([r["steps"] for r in records])
    groups = [np.flatnonzero(lengths == length) for length in LENGTHS]
    rng = np.random.default_rng(DRAW_SEED)
    draws = np.concatenate(
        [rng.choice(group, size=(DRAWS, len(group))) for group in groups], axis=1
    )
    boot = {key: [] for key in deltas}
    for sample in draws:
        for key, value in differences(
            summaries(left[sample].sum(axis=0)), summaries(right[sample].sum(axis=0))
        ).items():
            if value is not None:
                boot[key].append(value)
    return {
        "original": a,
        "refined": b,
        "experiments_with_new_pauses": int(np.count_nonzero(left[:, 23])),
        "paired_differences": {
            key: {
                "difference": value,
                "interval_95pct": np.quantile(boot[key], [0.025, 0.975]).tolist()
                if boot[key]
                else None,
                "defined_draws": len(boot[key]),
            }
            for key, value in deltas.items()
        },
    }


def score(output, populations):
    from fit_pause_targets import load_predictors, training_protocol

    receipt = read(output / "reservation.json")
    if (
        receipt["evaluator_sha256"] != sha256(Path(__file__))
        or receipt["training_script_sha256"]
        != sha256(Path(__file__).with_name("fit_pause_targets.py"))
        or receipt["plan_sha256"] != content_hash(protocol())
        or receipt["plan_sha256"] != sha256(output / "plan.json")
        or receipt["training_protocol"] != training_protocol()
        or receipt["environment"] != scientific_digests()
        or receipt["locations_sha256"] != sha256(output / "holdout" / "locations.json")
        or receipt["external_inventory_sha256"] != sha256(output / "prior.json")
    ):
        raise ValueError("The reserved environment, training protocol or cohort changed")
    freeze = read(output / "fitted" / "freeze.json")
    if freeze["reservation_sha256"] != sha256(output / "reservation.json"):
        raise ValueError("The two models must be frozen against this reservation")
    models = load_predictors(output, populations)
    if set(models) != {"original", "refined"}:
        raise ValueError("Both registered models are required before opening references")
    binding = {
        "freeze_sha256": sha256(output / "fitted" / "freeze.json"),
        "reservation_sha256": sha256(output / "reservation.json"),
    }
    record(output / "opened.json", binding)
    corpus = Corpus(output / "holdout", expected_digest=receipt["dataset_sha256"])
    records = evaluate(corpus, models)
    summary = score_groups(records)
    record(output / "results.json", {**binding, **summary, "experiments": records})
    print(summary, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_subparsers(dest="phase", required=True)
    command = phases.add_parser("prepare")
    command.add_argument("output", type=Path)
    command.add_argument("--panels", required=True, type=Path)
    for phase in ("reserve", "score"):
        command = phases.add_parser(phase)
        command.add_argument("output", type=Path)
        command.add_argument("--populations", required=True, type=Path)
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args.output, args.panels)
    elif args.phase == "reserve":
        reserve(args.output, args.populations)
    else:
        score(args.output, args.populations)
