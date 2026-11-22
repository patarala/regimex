"""Confirm a local-slope TCN against the existing refined-target TCN on new exchange runs."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from build_exposure_index import build
from compare_profile_21 import RUNTIME, Corpus, record, sha256, write
from compare_tcn_widths import PROFILE, generate
from compare_temporal_architectures import LENGTHS, read, scientific_digests
from confirm_pause_targets import protocol as target_protocol
from confirm_pause_targets import score_groups, sequence_counts
from confirm_tcn_widths import audit_population
from confirm_training_populations import manifest
from refine_model_intervals import refine_decoded
from refine_trend_pauses import annotate_pauses

from regimex.admin_contracts import GenerationSpec, RawGenerationPlan
from regimex.application.analysis_computation import snapshot_inputs
from regimex.data_catalog import SnapshotGrid, content_hash, create_snapshot_member
from regimex.segmentation import annotate, decode
from regimex.storage import output_directory

PER_LENGTH = 48
POPULATIONS = tuple(
    GenerationSpec(
        dataset_id=f"local-pause-features-final-{length}",
        plan=RawGenerationPlan(
            seed=1807360000 + length, experiment_count=PER_LENGTH, step_count=length
        ),
    )
    for length in LENGTHS
)


def protocol():
    return {
        **target_protocol(),
        "populations": [s.model_dump(mode="json") for s in POPULATIONS],
        "predictors": ["control", "candidate", "hybrid"],
        "control": "Previously frozen TCN, refined training targets, original feature inputs.",
        "candidate": "Same training data and targets with center-versus-flank slope inputs.",
        "hybrid": (
            "Control sequence decoded once, then a price-only pause check inside "
            "its predicted trends."
        ),
        "contrasts": {"primary": ["control", "candidate"], "secondary": ["control", "hybrid"]},
        "paired_differences": [
            "comparison minus control on each reference policy's interval macro-F1",
            "comparison minus control on newly inserted short sideways recall",
        ],
    }


def prepare(output: Path, previous: Path):
    receipt = read(previous / "reservation.json")
    last_dataset = previous / "holdout" / "dataset.json"
    prior = previous / "prior.json"
    if (
        manifest(last_dataset).content_sha256 != receipt["dataset_sha256"]
        or sha256(prior) != receipt["external_inventory_sha256"]
    ):
        raise ValueError("Prior exposures differ from the previous experiment reservation")
    frozen = read(previous / "fitted" / "freeze.json")
    if frozen["reservation_sha256"] != sha256(previous / "reservation.json"):
        raise ValueError("The previous freeze must identify its original reservation")
    with output_directory(output) as stage:
        write(stage / "plan.json", protocol())
        index = stage / "prior.json"
        write(index, build([last_dataset], [prior]))
        holdout = stage / "holdout"
        holdout.mkdir()
        corpus = generate(holdout, POPULATIONS, "local-pause-features-final")
        groups = Counter((m.step_count, len(m.instrument_ids)) for m in corpus.dataset.members)
        if groups != {(size, n): PER_LENGTH // 2 for size in LENGTHS for n in (1, 2)}:
            raise ValueError("The new population differs from the registered size and length mix")
        audit = audit_population(manifest(last_dataset), corpus.dataset, index)
        write(
            stage / "population.json",
            {
                **audit,
                "plan_sha256": sha256(stage / "plan.json"),
                "locations_sha256": sha256(holdout / "locations.json"),
                "previous_freeze_sha256": sha256(previous / "fitted" / "freeze.json"),
                "control_weights_sha256": frozen["weights_sha256"],
            },
        )


def evaluator_digests():
    files = (
        Path(__file__),
        Path(__file__).with_name("confirm_pause_targets.py"),
        Path(__file__).with_name("build_exposure_index.py"),
        Path(__file__).with_name("refine_model_intervals.py"),
        Path(__file__).with_name("refine_trend_pauses.py"),
    )
    return {**scientific_digests(), "comparison_scripts": {p.name: sha256(p) for p in files}}


def reserve(output: Path, populations: Path, previous: Path):
    from fit_local_pause_features import training_protocol
    from fit_pause_targets import load_predictors as previous_predictors

    if (output / "fitted").exists() or (output / "reservation.json").exists():
        raise ValueError("Reserve the new evaluation before fitting")
    population = read(output / "population.json")
    if (
        population["plan_sha256"] != content_hash(protocol())
        or population["plan_sha256"] != sha256(output / "plan.json")
        or population["previous_freeze_sha256"] != sha256(previous / "fitted" / "freeze.json")
        or population["control_weights_sha256"] != sha256(previous / "fitted" / "weights.npz")
        or population["external_inventory_sha256"] != sha256(output / "prior.json")
        or population["locations_sha256"] != sha256(output / "holdout" / "locations.json")
        or population["dataset_sha256"]
        != manifest(output / "holdout" / "dataset.json").content_sha256
    ):
        raise ValueError("The prepared cohort, exposure inventory or control changed")
    previous_predictors(previous, populations)
    record(
        output / "reservation.json",
        {
            **population,
            "training_protocol": training_protocol(),
            "environment": evaluator_digests(),
        },
    )


def evaluate(corpus, models):
    records = []
    for member in corpus.dataset.members:
        source = corpus.snapshot(member)
        if create_snapshot_member(source, member.partition) != member:
            raise ValueError("The opened snapshot differs from its reserved identity")
        grid = SnapshotGrid.from_export(source.export)
        inputs = snapshot_inputs(source.export)
        original = annotate(grid, PROFILE)
        refined = annotate_pauses(grid, PROFILE)
        probabilities = {
            name: model.predict(inputs, runtime=RUNTIME).reshape(
                len(inputs.info), grid.step_count, 3
            )
            for name, model in models.items()
        }
        counts = {arm: np.zeros(24, dtype=np.int64) for arm in ("control", "candidate", "hybrid")}
        for index, rows in enumerate(inputs.observations):
            references = [
                tuple(i for i in labels.intervals if i.instrument_id == rows[0].instrument_id)
                for labels in (original, refined)
            ]
            predictions = {
                arm: decode(steps[index], rows, PROFILE) for arm, steps in probabilities.items()
            }
            predictions["hybrid"] = refine_decoded(rows, predictions["control"], PROFILE)
            for arm, intervals in predictions.items():
                counts[arm] += sequence_counts(*references, intervals)
        records.append(
            {
                "experiment": content_hash([member.source_namespace, member.snapshot_id]),
                "steps": member.step_count,
                "counts": {arm: totals.tolist() for arm, totals in counts.items()},
            }
        )
    return records


def summarize(records):
    # The previous bootstrap takes two model arms named after their target policies.
    result = {"paired_differences": {}}
    for arm in ("candidate", "hybrid"):
        adapted = [
            {
                "steps": row["steps"],
                "counts": {"original": row["counts"]["control"], "refined": row["counts"][arm]},
            }
            for row in records
        ]
        comparison = score_groups(adapted)
        result["control"] = comparison["original"]
        result[arm] = comparison["refined"]
        result["experiments_with_new_pauses"] = comparison["experiments_with_new_pauses"]
        result["paired_differences"][arm] = comparison["paired_differences"]
    return result


def score(output: Path, populations: Path, previous: Path):
    from fit_local_pause_features import load_predictors, training_protocol

    receipt = read(output / "reservation.json")
    if (
        receipt["plan_sha256"] != content_hash(protocol())
        or receipt["plan_sha256"] != sha256(output / "plan.json")
        or receipt["previous_freeze_sha256"] != sha256(previous / "fitted" / "freeze.json")
        or receipt["control_weights_sha256"] != sha256(previous / "fitted" / "weights.npz")
        or receipt["locations_sha256"] != sha256(output / "holdout" / "locations.json")
        or receipt["external_inventory_sha256"] != sha256(output / "prior.json")
        or receipt["environment"] != evaluator_digests()
        or receipt["training_protocol"] != training_protocol()
    ):
        raise ValueError(
            "The registered population, control, feature extractor or scoring code changed"
        )
    binding = {
        "reservation_sha256": sha256(output / "reservation.json"),
        "freeze_sha256": sha256(output / "fitted" / "freeze.json"),
    }
    if (
        read(output / "fitted" / "freeze.json")["reservation_sha256"]
        != binding["reservation_sha256"]
    ):
        raise ValueError("The fitted candidate must be frozen against this reservation")
    models = load_predictors(output, populations, previous)
    if set(models) != {"control", "candidate"}:
        raise ValueError("Both registered models are required before opening references")
    record(output / "opened.json", binding)
    corpus = Corpus(output / "holdout", expected_digest=receipt["dataset_sha256"])
    records = evaluate(corpus, models)
    summary = summarize(records)
    record(output / "results.json", {**binding, **summary, "experiments": records})
    print(summary, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_subparsers(dest="phase", required=True)
    for name in ("prepare", "reserve", "score"):
        command = phases.add_parser(name)
        command.add_argument("output", type=Path)
        command.add_argument("--previous", type=Path, required=True)
        if name != "prepare":
            command.add_argument("--populations", type=Path, required=True)
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args.output, args.previous)
    elif args.phase == "reserve":
        reserve(args.output, args.populations, args.previous)
    else:
        score(args.output, args.populations, args.previous)
