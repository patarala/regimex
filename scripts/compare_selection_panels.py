"""Select previously fitted TCN bundles on size-matched panels and confirm the choices."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from build_exposure_index import build
from compare_profile_21 import RUNTIME, Corpus, record, sha256, write
from compare_tcn_widths import PROFILE, generate
from compare_temporal_architectures import (
    DRAW_SEED,
    DRAWS,
    LENGTHS,
    evaluator,
    experiment_counts,
    read,
    summary,
)
from compare_training_populations import ARMS
from confirm_tcn_widths import audit_population
from confirm_training_populations import (
    checked_reservation,
    contrast,
    environment,
    frozen_models,
    manifest,
    select_length,
)

from regimex.admin_contracts import GenerationSpec, RawGenerationPlan
from regimex.data_catalog import _load_partition, content_hash
from regimex.storage import output_directory

ENSEMBLE = ("short-96", "short-192", "mixed-192")
POOLS = {"validation": (2407131000, (24, 24, 16)), "holdout": (3652471000, (48, 48, 48))}
PANELS = {"short": (24, 24, 0), "balanced": (16, 16, 16)}
COMPARISONS = (
    ("short", "balanced"),
    ("historical", "short"),
    ("historical", "balanced"),
    ("historical", "ensemble"),
)


def specs(pool):
    seed, sizes = POOLS[pool]
    return tuple(
        GenerationSpec(
            dataset_id=f"selection-{pool}-{length}",
            plan=RawGenerationPlan(seed=seed + length, experiment_count=size, step_count=length),
        )
        for length, size in zip(LENGTHS, sizes, strict=True)
    )


def protocol():
    return {
        "pools": {pool: [s.model_dump(mode="json") for s in specs(pool)] for pool in POOLS},
        "panel_sizes": PANELS,
        "membership": "First native seeds in each length and instrument-count stratum.",
        "selection": "Largest pooled decoded interval macro-F1, IoU >= 0.50; first arm on ties.",
        "arms": ARMS,
        "ensemble": ENSEMBLE,
        "ensemble_rule": "Unweighted mean of per-step class probabilities, decoded once.",
        "historical": "short-96",
        "contrasts_control_candidate": COMPARISONS,
        "scores": [
            "pooled interval macro-F1",
            "interval macro-F1 at 1,024 steps",
            "point macro-F1",
            "short sideways recall",
            "sideways precision (all lengths)",
        ],
        "bootstrap": {"unit": "whole experiment within length", "draws": DRAWS, "seed": DRAW_SEED},
        "profile": PROFILE.model_dump(mode="json"),
        "runtime": RUNTIME.model_dump(mode="json"),
    }


def software():
    result = environment()
    for name in ("build_exposure_index.py", Path(__file__).name):
        result["scripts"][name] = sha256(Path(__file__).with_name(name))
    return result


def experiment_id(member):
    return content_hash([member.source_namespace, member.snapshot_id])


def panel_ids(members, sizes):
    ids = []
    for length, size in zip(LENGTHS, sizes, strict=True):
        for instruments in (1, 2):
            stratum = [
                m
                for m in members
                if m.step_count == length and len(m.instrument_ids) == instruments
            ]
            if len(stratum) < size // 2 or size % 2:
                raise ValueError("Insufficient experiments for a size-matched validation panel")
            chosen = sorted(stratum, key=lambda m: m.split_group.seed)[: size // 2]
            ids.extend(experiment_id(m) for m in chosen)
    return sorted(ids)


def reserve(output, source, exposure):
    previous = checked_reservation(source)
    _, historical = frozen_models(source, previous)
    if historical != "short-96" or sha256(exposure) != previous["external_inventory_sha256"]:
        raise ValueError("Expected the prior comparison and its original exposure index")
    previous_datasets = [
        source / "development" / "short-96" / "dataset.json",
        source / "development" / "augmentation" / "dataset.json",
        source / "confirmation" / "holdout" / "dataset.json",
    ]
    previous_holdout = manifest(previous_datasets[-1])
    if [manifest(path).content_sha256 for path in previous_datasets] != [
        previous["original_dataset_sha256"],
        previous["augmentation_sha256"],
        previous["dataset_sha256"],
    ]:
        raise ValueError("Exposure source differs from the original training and confirmation data")
    with output_directory(output) as stage:
        write(stage / "plan.json", protocol())
        inventory = stage / "prior.json"
        write(inventory, build(previous_datasets, [exposure]))
        pools = {}
        for name in POOLS:
            folder = stage / name
            folder.mkdir()
            corpus = generate(folder, specs(name), f"selection-{name}")
            expected = {
                (s.plan.step_count, n): s.plan.experiment_count // 2
                for s in specs(name)
                for n in (1, 2)
            }
            counts = Counter((m.step_count, len(m.instrument_ids)) for m in corpus.dataset.members)
            if counts != expected:
                raise ValueError("Generated length and instrument strata differ from registration")
            audit = audit_population(previous_holdout, corpus.dataset, inventory)
            pools[name] = {**audit, "locations_sha256": sha256(folder / "locations.json")}
        validation = manifest(stage / "validation" / "dataset.json")
        holdout = manifest(stage / "holdout" / "dataset.json")
        audit_population(validation, holdout, inventory)
        panels = {name: panel_ids(validation.members, size) for name, size in PANELS.items()}
        write(
            stage / "reservation.json",
            {
                "plan_sha256": sha256(stage / "plan.json"),
                "previous_freeze_sha256": sha256(source / "fitted" / "freeze.json"),
                "environment": software(),
                "pools": pools,
                "panels": panels,
            },
        )


def checked(output, source):
    receipt = read(output / "reservation.json")
    if not (
        receipt["plan_sha256"] == content_hash(protocol()) == sha256(output / "plan.json")
        and receipt["previous_freeze_sha256"] == sha256(source / "fitted" / "freeze.json")
        and receipt["environment"] == software()
    ):
        raise ValueError("Protocol, frozen candidates or evaluator differs from the reservation")
    models, _ = frozen_models(source, checked_reservation(source))
    return receipt, models


def open_corpus(output, receipt, pool):
    path = output / pool
    info = receipt["pools"][pool]
    if sha256(path / "locations.json") != info["locations_sha256"]:
        raise ValueError("Corpus locations changed after reservation")
    return Corpus(path, expected_digest=info["dataset_sha256"])


def predictions(corpus, models, names):
    samples = []
    for partition in ("train", "validation", "test"):
        refs, labels = corpus.references(partition, profile=PROFILE)
        samples.extend(
            _load_partition(
                corpus.dataset,
                refs,
                snapshot_reader=corpus.snapshot,
                reference_reader=lambda _, m, labels=labels: labels[m.source_namespace],
            )
        )
    evaluation = evaluator(tuple(samples))
    probabilities = {
        name: models[name].predict(evaluation.inputs, runtime=RUNTIME)
        for name in dict.fromkeys((*names, *ENSEMBLE))
    }
    probabilities["ensemble"] = np.mean([probabilities[k] for k in ENSEMBLE], axis=0)
    groups = {
        name: experiment_counts(evaluation, probabilities[name]) for name in (*names, "ensemble")
    }
    expected = {experiment_id(m): m.step_count for m in corpus.dataset.members}
    for ids, lengths, _ in groups.values():
        if dict(zip(ids, lengths.tolist(), strict=True)) != expected:
            raise ValueError("Scored examples differ from the reserved experiment identities")
    return groups


def ranking(scores):
    return sorted(ARMS, key=lambda name: scores[name]["interval_macro_f1"], reverse=True)


def select(output, source):
    receipt, models = checked(output, source)
    corpus = open_corpus(output, receipt, "validation")
    groups = predictions(corpus, models, ARMS)
    panels = {}
    for name, size in PANELS.items():
        ids = set(panel_ids(corpus.dataset.members, size))
        if ids != set(receipt["panels"][name]):
            raise ValueError("Panel composition changed after reservation")
        scores = {}
        for arm, (experiment_ids, _, counts) in groups.items():
            scores[arm] = summary(counts[[i in ids for i in experiment_ids]])
        ordered = ranking(scores)
        panels[name] = {"rank": ordered, "choice": ordered[0], "scores": scores}
    record(
        output / "selection.json",
        {"reservation_sha256": sha256(output / "reservation.json"), "panels": panels},
    )


def choices(output):
    selection = read(output / "selection.json")
    if selection["reservation_sha256"] != sha256(output / "reservation.json"):
        raise ValueError("Frozen panel choices refer to a different reservation")
    selected = {"historical": "short-96", "ensemble": "ensemble"}
    for name in PANELS:
        panel = selection["panels"][name]
        ordered = ranking(panel["scores"])
        if ordered != panel["rank"] or ordered[0] != panel["choice"]:
            raise ValueError("Selected bundle differs from the recorded validation ranking")
        selected[name] = ordered[0]
    return selected


def score(output, source):
    selected = choices(output)
    receipt, models = checked(output, source)
    record(output / "opened.json", {"selection_sha256": sha256(output / "selection.json")})
    corpus = open_corpus(output, receipt, "holdout")
    names = tuple(dict.fromkeys(selected[name] for name in ("historical", *PANELS)))
    groups = predictions(corpus, models, names)
    results = {}
    for name, (ids, lengths, counts) in groups.items():
        results[name] = {
            **summary(counts),
            "by_length": {n: summary(counts[lengths == n]) for n in LENGTHS},
            "experiment_counts": dict(zip(ids, counts.tolist(), strict=True)),
        }
    comparisons = {}
    for control, candidate in COMPARISONS:
        left, right = groups[selected[control]], groups[selected[candidate]]
        comparisons[f"{candidate}_vs_{control}"] = {
            "pooled": contrast(left, right),
            "1024": contrast(select_length(left, 1024), select_length(right, 1024)),
        }
    record(
        output / "results.json",
        {
            "selection_sha256": sha256(output / "selection.json"),
            "choices": selected,
            "results": results,
            "comparisons": comparisons,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)
    for phase in ("reserve", "select", "score"):
        phase_parser = subparsers.add_parser(phase)
        phase_parser.add_argument("output", type=Path)
        phase_parser.add_argument(
            "--source", type=Path, required=True, help="Previous training-population comparison."
        )
        if phase == "reserve":
            phase_parser.add_argument("--exposure-index", type=Path, required=True)
    args = parser.parse_args()
    if args.phase == "reserve":
        reserve(args.output, args.source, args.exposure_index)
    elif args.phase == "select":
        select(args.output, args.source)
    else:
        score(args.output, args.source)
