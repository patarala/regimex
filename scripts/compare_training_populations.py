"""Compare native TCN training populations on a fixed, previously reserved validation set."""

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from statistics import mean

import numpy as np
from compare_profile_21 import RUNTIME, Corpus, record, sha256, software, write
from compare_tcn_widths import DATASET_SHA256, POLICY, PROFILE, PROFILE_ID, training_data
from confirm_tcn_widths import audit_population
from plateau_sampling import enrich_short_sideways

from regimex.application.training import TrainingRequest, train_candidate
from regimex.backends.tcn_update_records import UpdateHistoryRecord
from regimex.data_catalog import (
    DatasetManifest,
    Partition,
    SnapshotMember,
    SourceSnapshot,
    content_hash,
    create_snapshot_member,
    seal_dataset,
)
from regimex.market_export import sample_snapshot
from regimex.market_identity import market_seed_group
from regimex.model_registry import ArchitectureRegistry, FixedUpdatePolicy, TrainingRuntime
from regimex.storage import json_bytes, output_directory
from regimex.terminal_training import register_completed_bundle

ARMS = ("short-96", "mixed-96", "short-192", "mixed-192", "mixed-192-double", "mixed-96-enriched")
EXTRA = {256: 48, 512: 48, 1024: 64}
SEED_BASE = 3291167000


def read(path: Path):
    return json.loads(path.read_bytes())


def extra_experiments():
    for length, count in EXTRA.items():
        # A native manifest needs validation and test members. These two extra seeds
        # satisfy that contract without taking an experiment out of the training pool.
        for index in range(count + (2 if length == 256 else 0)):
            partition: Partition = "train"
            if index >= count:
                partition = ("validation", "test")[index - count]
            seed = int(np.random.SeedSequence([SEED_BASE + length, index]).generate_state(1)[0])
            yield length, 1 + index % 2, seed, partition


def save_corpus(folder: Path, dataset: DatasetManifest, locations: dict[str, str]) -> None:
    write(folder / "dataset.json", dataset.model_dump(mode="json"))
    write(folder / "locations.json", locations)
    write(folder / "reservation.json", {"dataset_sha256": dataset.content_sha256})


def generate_augmentation(folder: Path) -> None:
    with output_directory(folder) as stage:
        (stage / "output").mkdir()
        members, locations = [], {}
        for length, instruments, seed, partition in extra_experiments():
            exp = sample_snapshot(seed=seed, steps=length, instruments=instruments)
            ns = content_hash(["training-populations", seed])
            source = SourceSnapshot(
                source_namespace=ns, split_group=market_seed_group(seed), export=exp
            )
            members.append(create_snapshot_member(source, partition))
            locations[ns] = f"output/{seed}.json"
            write(stage / locations[ns], exp.model_dump(mode="json"))
            if len(members) % 16 == 0:
                print(f"Generated {len(members)} additional experiments", flush=True)
        save_corpus(
            stage, seal_dataset("training-populations-augmentation", tuple(members)), locations
        )


def augmentation(folder: Path) -> Corpus:
    if not (folder / "dataset.json").exists():
        generate_augmentation(folder)
    extra = Corpus(folder, expected_digest=read(folder / "reservation.json")["dataset_sha256"])
    actual = [
        (m.step_count, len(m.instrument_ids), m.split_group.seed, m.partition)
        for m in extra.dataset.members
    ]
    if actual != list(extra_experiments()):
        raise ValueError("Additional experiments differ from the declared generation plan")
    return extra


def stratified_prefix(members, totals):
    chosen = set()
    for (length, instruments), count in totals.items():
        group = sorted(
            (m for m in members if m.step_count == length and len(m.instrument_ids) == instruments),
            key=lambda m: m.split_group.seed,
        )
        if not 0 <= count <= len(group):
            raise ValueError("Insufficient experiments in a declared training stratum")
        chosen.update(m.source_namespace for m in group[:count])
    return tuple(m for m in members if m.source_namespace in chosen)


def uniform_cohorts(original: DatasetManifest, extra: DatasetManifest):
    original_train = tuple(m for m in original.members if m.partition == "train")
    extra_train = tuple(m for m in extra.members if m.partition == "train")
    short_extra = tuple(m for m in extra_train if m.step_count < 1024)
    long = tuple(m for m in extra_train if m.step_count == 1024)
    strata = {(length, instruments): 16 for length in (256, 512) for instruments in (1, 2)}
    original_counts = Counter((m.step_count, len(m.instrument_ids)) for m in original_train)
    missing = {key: 32 - original_counts[key] for key in strata}
    return {
        "short-96": original_train,
        "mixed-96": stratified_prefix(original_train, strata) + long[:32],
        "short-192": original_train + short_extra,
        "mixed-192": original_train + stratified_prefix(short_extra, missing) + long,
    }


def population_summary(members):
    counts = Counter((m.step_count, len(m.instrument_ids)) for m in members)
    return {
        "experiments": sum(counts.values()),
        "sequences": sum(instruments * n for (_, instruments), n in counts.items()),
        "rows": sum(length * instruments * n for (length, instruments), n in counts.items()),
        "strata": [
            {"steps": length, "instruments": instruments, "experiments": n}
            for (length, instruments), n in sorted(counts.items())
        ],
    }


def scientific_environment():
    folder = Path(__file__).parent
    names = (
        "compare_training_populations.py",
        "plateau_sampling.py",
        "compare_tcn_widths.py",
        "confirm_tcn_widths.py",
    )
    return {**software(), "scripts": {name: sha256(folder / name) for name in names}}


def policy_for(name: str) -> FixedUpdatePolicy:
    if name == "mixed-192-double":
        return FixedUpdatePolicy.model_validate(
            {**POLICY.model_dump(), "updates": 192, "validation_updates": (64, 128, 192)}
        )
    return POLICY


def prepare(output: Path, development: Path, exposure: Path) -> None:

    base = Corpus(development, expected_digest=DATASET_SHA256)
    directory = output / "development"
    extra = augmentation(directory / "augmentation")
    audit = audit_population(base.dataset, extra.dataset, exposure)
    cohorts = uniform_cohorts(base.dataset, extra.dataset)
    locations = {
        ns: corpus.output / path
        for corpus in (base, extra)
        for ns, path in corpus.locations.items()
    }

    def snapshot(member: SnapshotMember):
        corpus = base if member.source_namespace in base.locations else extra
        return corpus.snapshot(member)

    cohorts["mixed-96-enriched"], enrichment = enrich_short_sideways(
        cohorts["mixed-96"], cohorts["mixed-192"], snapshot
    )
    fixed = tuple(m for m in base.dataset.members if m.partition != "train")
    validation = tuple(m for m in fixed if m.partition == "validation")
    candidates = {}
    for name in ARMS:
        population = "mixed-192" if name == "mixed-192-double" else name
        train = cohorts[population]
        dataset = (
            base.dataset
            if name == "short-96"
            else seal_dataset(f"population-{population}", train + fixed)
        )
        folder = directory / population
        folder.mkdir(parents=True, exist_ok=True)
        record(folder / "dataset.json", dataset.model_dump(mode="json"))
        record(
            folder / "locations.json",
            {
                m.source_namespace: os.path.relpath(locations[m.source_namespace], folder)
                for m in dataset.members
            },
        )
        record(folder / "reservation.json", {"dataset_sha256": dataset.content_sha256})
        candidates[name] = {
            "dataset_path": population,
            "dataset_sha256": dataset.content_sha256,
            "train": population_summary(train),
            "policy": policy_for(name).model_dump(mode="json"),
        }
    reference_set, _ = base.references("validation", profile=PROFILE)
    record(
        directory / "plan.json",
        {
            "development_sha256": DATASET_SHA256,
            "augmentation_sha256": extra.dataset.content_sha256,
            "exposure_audit": audit,
            "architecture": "tcn",
            "parameters": {"channels": 48},
            "profile": PROFILE.model_dump(mode="json"),
            "validation": population_summary(validation),
            "validation_labels_sha256": content_hash(
                [member.model_dump(mode="json") for member in reference_set.members]
            ),
            "enrichment": enrichment,
            "order": ARMS,
            "selection": (
                "Largest unrounded mean seed-best validation interval macro-F1; first arm on ties."
            ),
            "checkpoint": (
                "Highest validation interval macro-F1 across seeds and updates; "
                "first checkpoint on ties."
            ),
            "candidates": candidates,
            "training_runtime": TrainingRuntime(threads=2).model_dump(mode="json"),
            "verification_runtime": RUNTIME.model_dump(mode="json"),
            "environment": scientific_environment(),
        },
    )
    print(f"Prepared {directory / 'plan.json'}", flush=True)


def per_seed_history(path: Path, policy: FixedUpdatePolicy):
    history = UpdateHistoryRecord.model_validate_json(path.read_bytes())
    result = []
    for seed in policy.seeds:
        entries = [entry for entry in history.updates if entry.seed == seed]
        scores = [
            (entry.update, entry.validation_segment_macro_f1)
            for entry in entries
            if entry.validation_segment_macro_f1 is not None
        ]
        if (
            len(entries) != policy.updates
            or tuple(step for step, _ in scores) != policy.validation_updates
        ):
            raise ValueError("Recorded optimizer updates differ from the registered policy")
        step, score = max(scores, key=lambda item: item[1])
        result.append(
            {
                "seed": seed,
                "update": step,
                "interval_macro_f1": score,
                "optimizer_updates": len(entries),
                "trained_sequences": sum(row.trained_sequences for row in entries),
                "real_rows": sum(row.real_rows for row in entries),
                "padding_rows": sum(row.padding_rows for row in entries),
                "eligible_rows": sum(row.eligible_rows for row in entries),
            }
        )
    return result


def fit_candidate(directory, name, candidate, plan, registry, destination):
    corpus = Corpus(
        directory / candidate["dataset_path"], expected_digest=candidate["dataset_sha256"]
    )
    policy = FixedUpdatePolicy.model_validate_json(json_bytes(candidate["policy"]))
    references, _ = corpus.references("validation", profile=PROFILE)
    if (
        content_hash([m.model_dump(mode="json") for m in references.members])
        != plan["validation_labels_sha256"]
    ):
        raise ValueError("Validation membership, ordering or annotations have changed")
    data = training_data(corpus)
    request = TrainingRequest(
        run_id=name,
        model_version_id=name,
        architecture_id="tcn",
        parameters=plan["parameters"],
        policy=policy,
        training_runtime=TrainingRuntime.model_validate(plan["training_runtime"]),
        verification_runtime=RUNTIME,
        analysis_profile_id=PROFILE_ID,
        build_id="source-" + plan["environment"]["source_sha256"],
    )
    with output_directory(destination) as stage:
        bundle = stage / "bundle"
        fitting = train_candidate(
            data, request, bundle, registry=registry, worker_id="population-comparison"
        )
        stats = fitting.fit_statistics
        if (
            stats.train_rows != candidate["train"]["rows"]
            or stats.validation_rows != plan["validation"]["rows"]
        ):
            raise ValueError("Native trainer row counts differ from the population plan")
        per_seed = per_seed_history(bundle / "history.json", policy)
        summary = {
            "bundle_path": f"{name}/bundle",
            "bundle_sha256": fitting.manifest_sha256,
            "dataset_sha256": fitting.dataset_sha256,
            "training_labels_sha256": data.labels.content_sha256,
            "validation_reference_sha256": fitting.validation_reference_sha256,
            "validation_labels_sha256": plan["validation_labels_sha256"],
            "selected_checkpoint": read(bundle / "validation.json"),
            "per_seed": per_seed,
            "mean_interval_macro_f1": mean(item["interval_macro_f1"] for item in per_seed),
            "fit_statistics": stats.model_dump(mode="json"),
        }
        write(stage / "summary.json", summary)
    register_completed_bundle(destination / "bundle", fitting.manifest_sha256)
    return summary


def fit(output: Path) -> None:

    from regimex.backends.tcn import TCNBackend

    directory, fitted = output / "development", output / "fitted"
    plan_path = directory / "plan.json"
    reservation_path = output / "confirmation" / "reservation.json"
    plan, reservation = read(plan_path), read(reservation_path)
    binding = {
        "development_plan_sha256": sha256(plan_path),
        "reservation_sha256": sha256(reservation_path),
    }
    if (
        reservation["development_plan_sha256"] != binding["development_plan_sha256"]
        or reservation["augmentation_sha256"] != plan["augmentation_sha256"]
        or plan["environment"] != scientific_environment()
        or tuple(plan["order"]) != ARMS
    ):
        raise ValueError(
            "The training plan, confirmation reservation or implementation has changed"
        )
    fitted.mkdir(exist_ok=True)
    record(fitted / "binding.json", binding)
    results = {}
    registry = ArchitectureRegistry.discover()
    for name in ARMS:
        destination = fitted / name
        candidate = plan["candidates"][name]
        if destination.exists():
            summary = read(destination / "summary.json")
            manifest = read(destination / "bundle" / "manifest.json")
            TCNBackend().load_bundle(
                destination / "bundle", manifest_sha256=summary["bundle_sha256"], runtime=RUNTIME
            )
            if (
                summary["dataset_sha256"] != candidate["dataset_sha256"]
                or manifest["training"]["dataset_sha256"] != candidate["dataset_sha256"]
                or manifest["model_version_id"] != name
                or summary["per_seed"]
                != per_seed_history(destination / "bundle" / "history.json", policy_for(name))
                or summary["mean_interval_macro_f1"]
                != mean(item["interval_macro_f1"] for item in summary["per_seed"])
            ):
                raise ValueError(
                    "A previously finished candidate differs from the registered comparison"
                )
        else:
            print(f"Fitting {name}", flush=True)
            summary = fit_candidate(directory, name, candidate, plan, registry, destination)
        results[name] = summary
        print(f"{name} validation {summary['mean_interval_macro_f1']:.4f}", flush=True)
    if (
        sha256(plan_path) != binding["development_plan_sha256"]
        or sha256(reservation_path) != binding["reservation_sha256"]
    ):
        raise ValueError("A registered file changed during fitting")
    choice = max(ARMS, key=lambda name: results[name]["mean_interval_macro_f1"])
    record(fitted / "freeze.json", {**binding, "validation_choice": choice, "candidates": results})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="phase", required=True)
    preparation = commands.add_parser(
        "prepare", help="Register training cohorts and validation before reservation."
    )
    preparation.add_argument("output", type=Path)
    preparation.add_argument(
        "--development", type=Path, required=True, help="Original width-comparison corpus."
    )
    preparation.add_argument("--exposure-index", type=Path, required=True)
    fitting = commands.add_parser(
        "fit", help="Fit and freeze all registered candidates after reservation."
    )
    fitting.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args.output, args.development, args.exposure_index)
    else:
        fit(args.output)


if __name__ == "__main__":
    main()
