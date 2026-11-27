"""Compare centered price features with additional past and future slopes on short plateaus."""

import argparse
import platform
from collections.abc import Iterable, Iterator
from importlib.metadata import version
from pathlib import Path
from typing import TypedDict

import numpy as np
from compare_profile_21 import LENGTHS, PROFILE, RUNTIME, Corpus, record, request, sha256
from pydantic import JsonValue, TypeAdapter

from regimex.admin_contracts import GenerationSpec, RawGenerationPlan
from regimex.application.admin_computation import _generation
from regimex.application.analysis_computation import snapshot_inputs
from regimex.application.training import TrainingRequest, train_candidate
from regimex.backends.tcn import TCNBackend, TCNPredictor
from regimex.benchmark_contracts import dataset_identity_keys
from regimex.data_catalog import (
    DatasetManifest,
    SnapshotGrid,
    SnapshotMember,
    content_hash,
    load_training_data,
    seal_dataset,
)
from regimex.evaluation import _metrics, _segment_counts, matched_interval_indices
from regimex.exports import ExportV2, load_export_v2
from regimex.features import IntArray
from regimex.model_bundles import ModelBundle
from regimex.model_registry import ArchitectureRegistry, TCNParameters
from regimex.operations import Sha256
from regimex.segmentation import annotate, decode
from regimex.storage import json_bytes
from regimex.terminal_training import register_completed_bundle

MODES = ("centered", "with-side-slopes")
ROOT = Path(__file__).resolve().parents[1]
HOLDOUT_SEED = 2865017021
COUNT_PER_LENGTH = 32
HOLDOUT_DIGEST = "5bdc936d5116f7c72c7b641a7d888c95223b94796c4f89a4f397df847079ce6d"


class ExposureIndex(TypedDict):
    excluded_seeds: list[int]
    identity_keys_sha256: list[Sha256]


class SnapshotLocation(TypedDict):
    path: str


class SelectionPlan(TypedDict):
    requests: dict[str, TrainingRequest]


class SelectionFreeze(TypedDict):
    training_plan_sha256: Sha256
    bundles: dict[str, Sha256]


class ComparisonPlan(TypedDict):
    corpus: Sha256
    centered_manifest: Sha256
    training: dict[str, JsonValue]
    holdout: dict[str, int | list[int]]
    selection: str
    endpoints: list[str]
    source: dict[str, Sha256]
    lockfile: Sha256
    environment: dict[str, str]
    python: str


class Reservation(TypedDict):
    dataset: Sha256
    locations_sha256: Sha256
    checked_keys: int
    checked_seeds: int
    exposure_index_sha256: Sha256 | None


class Freeze(TypedDict):
    manifests: dict[str, Sha256]
    holdout_reservation_sha256: Sha256
    plan_sha256: Sha256
    validation_sha256: Sha256


class PairedDifference(TypedDict):
    draws_with_support: int
    interval_95pct: list[float] | None


type Metrics = dict[str, float | int | None]


class Comparison(TypedDict):
    pooled: dict[str, Metrics]
    experiments: int
    per_experiment: dict[str, list[list[int]]]
    paired_bootstrap: dict[str, PairedDifference]


def read[T](path: Path, adapter: TypeAdapter[T]) -> T:
    return adapter.validate_json(path.read_bytes(), strict=True)


def seeds_of(members: Iterable[SnapshotMember]) -> set[int]:
    result = set[int]()
    for member in members:
        if member.split_group.seed is None:
            raise ValueError("The synthetic cohort must identify each generator seed")
        result.add(member.split_group.seed)
    return result


def reserve(output: Path, history_path: Path | None, corpus: Corpus) -> None:
    """Register every generated export before looking at any reference intervals."""
    folder = output / "holdout"
    folder.mkdir(parents=True)
    history: dict[str, JsonValue] = {"excluded_seeds": [], "identity_keys_sha256": []}
    if history_path:
        history = read(history_path, TypeAdapter(dict[str, JsonValue]))
    exposure = TypeAdapter(ExposureIndex).validate_python(history, strict=True)
    members: list[SnapshotMember] = []
    locations: dict[str, str] = {}
    for length in LENGTHS:
        dest = folder / str(length)
        dest.mkdir()
        generated = _generation(
            GenerationSpec(
                dataset_id=f"tcn-slopes-{length}",
                plan=RawGenerationPlan(
                    seed=HOLDOUT_SEED + length,
                    experiment_count=COUNT_PER_LENGTH,
                    step_count=length,
                ),
            ),
            dest,
            "tcn-slopes",
        )
        manifest = DatasetManifest.for_modeling(json_bytes(generated["manifest"]))
        snapshots = TypeAdapter(list[SnapshotLocation]).validate_python(
            generated["snapshots"], strict=True
        )
        for member, snapshot in zip(manifest.members, snapshots, strict=True):
            members.append(member)
            locations[member.source_namespace] = f"{length}/{snapshot['path']}"
    # Training uses the profile-2.1 corpus. Check its full membership even if the
    # supplied historical exposure inventory predates that comparison.
    prior_members = corpus.dataset.members
    prior_seeds = set(exposure["excluded_seeds"]) | seeds_of(prior_members)
    prior_keys = set(exposure["identity_keys_sha256"]) | {
        content_hash(k) for k in dataset_identity_keys(prior_members)
    }
    seeds = seeds_of(members)
    keys = {content_hash(k) for k in dataset_identity_keys(tuple(members))}
    if len(seeds) != len(members) or seeds & prior_seeds or keys & prior_keys:
        raise ValueError("The reserved population overlaps monitored exposures or repeats a seed")
    dataset = seal_dataset("tcn-slope-comparison", tuple(members))
    if dataset.content_sha256 != HOLDOUT_DIGEST:
        raise ValueError("Generated population differs from the preregistered complete cohort")
    record(folder / "dataset.json", dataset.model_dump(mode="json"))
    record(folder / "locations.json", locations)
    record(
        folder / "reservation.json",
        {
            "dataset": dataset.content_sha256,
            "locations_sha256": sha256(folder / "locations.json"),
            "exposure_index_sha256": sha256(history_path) if history_path else None,
            "checked_seeds": len(prior_seeds),
            "checked_keys": len(prior_keys),
        },
    )
    if not history_path:
        return
    record(
        folder / "exposure-index.json",
        {
            **history,
            "previous_sha256": sha256(history_path),
            "scope": "Prior exposures plus the complete temporal slope comparison population.",
            "current_dataset_paths": ["dataset.json"],
            "excluded_seeds": sorted(prior_seeds | seeds),
            "identity_keys_sha256": sorted(prior_keys | keys),
        },
    )


def protocol(corpus: Corpus) -> ComparisonPlan:
    scientific_files = (
        "src/regimex/temporal.py",
        "src/regimex/temporal_updates.py",
        "src/regimex/backends/tcn.py",
        "src/regimex/model_registry.py",
        "src/regimex/application/training.py",
        "src/regimex/directions.py",
        "src/regimex/evaluation.py",
        "src/regimex/segmentation.py",
        "src/regimex/features.py",
        "scripts/compare_tcn_slopes.py",
        "scripts/compare_profile_21.py",
    )
    freeze = read(corpus.output / "freeze.json", TypeAdapter(SelectionFreeze))
    if sha256(corpus.output / "training-plan.json") != freeze["training_plan_sha256"]:
        raise ValueError("The training plan of the centered model differs from its freeze")
    original_request = read(corpus.output / "training-plan.json", TypeAdapter(SelectionPlan))[
        "requests"
    ]["tcn"]
    if original_request != request("tcn", original_request.build_id):
        raise ValueError("The reference TCN did not use the comparison's training budget")
    return {
        "corpus": corpus.dataset.content_sha256,
        "centered_manifest": freeze["bundles"]["tcn"],
        "training": original_request.model_dump(mode="json"),
        "holdout": {"seed": HOLDOUT_SEED, "lengths": list(LENGTHS), "per_length": COUNT_PER_LENGTH},
        "selection": "Select each checkpoint by decoded validation interval macro-F1. Freeze both.",
        "endpoints": [
            "Recall of reference sideways intervals shorter than 64 steps",
            "Pooled decoded interval macro-F1",
            "Overall sideways interval precision",
        ],
        "source": {f: sha256(ROOT / f) for f in scientific_files},
        "lockfile": sha256(ROOT / "uv.lock"),
        "environment": {n: version(n) for n in ("numpy", "torch", "pydantic")},
        "python": platform.python_version(),
    }


def fit(output: Path, corpus: Corpus) -> None:
    output.mkdir(parents=True, exist_ok=True)
    plan = protocol(corpus)
    reservation = output / "holdout/reservation.json"
    if read(reservation, TypeAdapter(Reservation))["dataset"] != HOLDOUT_DIGEST:
        raise ValueError("Reserve the registered population before fitting")
    record(output / "plan.json", plan)
    training, labels = corpus.references("train")
    validation, annotations = corpus.references("validation")
    labels.update(annotations)
    data = load_training_data(
        corpus.dataset,
        training,
        validation,
        snapshot_reader=corpus.snapshot,
        reference_reader=lambda _, member: labels[member.source_namespace],
    )
    req = request("tcn", "tcn-slopes-" + content_hash(plan))
    req = req.model_copy(
        update={
            "model_version_id": "tcn-side-slopes",
            "run_id": "tcn-side-slopes",
            "parameters": {**req.parameters, "side_slopes": True},
        }
    )
    fitted = train_candidate(
        data,
        req,
        output / MODES[1],
        registry=ArchitectureRegistry.discover(),
        worker_id="tcn-slope-comparison",
    )
    register_completed_bundle(output / MODES[1], fitted.manifest_sha256)
    manifests = {MODES[0]: plan["centered_manifest"], MODES[1]: fitted.manifest_sha256}
    models = load_models(output, corpus, manifests)
    for name, model in models.items():
        if (
            model.configuration.parameters
            != TCNParameters.model_validate({**req.parameters, "side_slopes": name == MODES[1]})
            or model.configuration.training_policy != req.policy
        ):
            raise ValueError("The compared models differ from the registered protocol")
        path = corpus.output / "tcn" if name == MODES[0] else output / name
        evidence = ModelBundle.model_validate_json((path / "manifest.json").read_bytes()).training
        if (
            evidence.labels.content_sha256 != training.content_sha256
            or evidence.validation_references.content_sha256 != validation.content_sha256
            or evidence.dataset_sha256 != corpus.dataset.content_sha256
        ):
            raise ValueError("Training or validation provenance differs for the two models")
    reports = compare((item.export for item in data.validation), models)
    record(output / "validation.json", reports)
    record(
        output / "freeze.json",
        {
            "manifests": manifests,
            "holdout_reservation_sha256": sha256(reservation),
            "plan_sha256": sha256(output / "plan.json"),
            "validation_sha256": sha256(output / "validation.json"),
        },
    )
    print("Validation:", reports["pooled"], flush=True)


def load_models(output: Path, corpus: Corpus, manifests: dict[str, str]) -> dict[str, TCNPredictor]:
    if manifests.keys() != set(MODES):
        raise ValueError("The freeze must identify exactly the two comparison models")
    models = {
        mode: TCNBackend().load_bundle(
            corpus.output / "tcn" if mode == MODES[0] else output / mode,
            manifest_sha256=manifests[mode],
            runtime=RUNTIME,
        )
        for mode in MODES
    }
    if any(model.analysis != PROFILE for model in models.values()):
        raise ValueError("Model analysis profile differs from the registered protocol")
    return models


def compare(exports: Iterable[ExportV2], models: dict[str, TCNPredictor]) -> Comparison:
    counts: dict[str, list[IntArray]] = {name: [] for name in MODES}
    for export in exports:
        inputs = snapshot_inputs(export)
        annotation = annotate(SnapshotGrid.from_export(export), PROFILE)
        for name, model in models.items():
            probabilities = model.predict(inputs, runtime=RUNTIME)
            offset = 0
            # Counts per experiment keep correlated instruments together in the bootstrap.
            totals = np.zeros(11, dtype=np.int64)
            for info, rows in zip(inputs.info, inputs.observations, strict=True):
                expected = tuple(
                    i for i in annotation.intervals if i.instrument_id == info.instrument_id
                )
                predicted = decode(probabilities[offset : offset + len(rows)], rows, PROFILE)
                offset += len(rows)
                totals[:9] += _segment_counts(expected, predicted).ravel()
                sideways = [i for i in expected if i.direction == "sideways"]
                found = matched_interval_indices(
                    sideways, [i for i in predicted if i.direction == "sideways"]
                )
                short = {idx for idx, item in enumerate(sideways) if item.stop - item.start < 64}
                totals[9:] += (len(short & found.keys()), len(short))
            counts[name].append(totals)
    size = len(counts[MODES[0]])
    if not size:
        raise ValueError("Comparison needs at least one experiment")
    arrays = {name: np.stack(data) for name, data in counts.items()}
    pooled = {mode: metrics(arrays[mode].sum(axis=0)) for mode in MODES}
    rng = np.random.default_rng(1729)
    groups = rng.integers(size, size=(1000, size))
    deltas: dict[str, list[float]] = {
        key: [] for key in ("interval_macro_f1", "short_sideways_recall", "sideways_precision")
    }
    for sample in groups:
        left, right = (metrics(arrays[mode][sample].sum(axis=0)) for mode in MODES)
        for key, draws in deltas.items():
            before, after = left[key], right[key]
            if before is not None and after is not None:
                draws.append(after - before)
    return {
        "pooled": pooled,
        "experiments": size,
        "per_experiment": {name: values.tolist() for name, values in arrays.items()},
        "paired_bootstrap": {
            key: {
                "draws_with_support": len(draws),
                "interval_95pct": np.quantile(draws, [0.025, 0.975]).tolist() if draws else None,
            }
            for key, draws in deltas.items()
        },
    }


def metrics(counts: IntArray) -> Metrics:
    segments = counts[:9].reshape(3, 3)
    matched, extra, _ = segments[0]
    short_matched, short_references = map(int, counts[9:])
    macro_f1 = _metrics(segments)["macro_f1"]
    if not isinstance(macro_f1, float):
        raise ValueError("Expected a numeric interval macro-F1")
    return {
        "interval_macro_f1": macro_f1,
        "short_sideways_matched": short_matched,
        "short_sideways_references": short_references,
        "short_sideways_recall": short_matched / short_references if short_references else None,
        "sideways_precision": float(matched / (matched + extra)) if matched + extra else None,
        "false_sideways_intervals": int(extra),
    }


def score(output: Path, corpus: Corpus) -> None:
    plan = protocol(corpus)
    freeze = read(output / "freeze.json", TypeAdapter(Freeze))
    folder = output / "holdout"
    reservation = read(folder / "reservation.json", TypeAdapter(Reservation))
    holdout = DatasetManifest.for_modeling((folder / "dataset.json").read_bytes())
    locations = read(folder / "locations.json", TypeAdapter(dict[str, str]))
    if (
        json_bytes(plan) != (output / "plan.json").read_bytes()
        or freeze["plan_sha256"] != sha256(output / "plan.json")
        or freeze["validation_sha256"] != sha256(output / "validation.json")
        or freeze["manifests"].get(MODES[0]) != plan["centered_manifest"]
        or freeze["holdout_reservation_sha256"] != sha256(folder / "reservation.json")
        or holdout.content_sha256 != HOLDOUT_DIGEST
        or reservation["dataset"] != holdout.content_sha256
        or reservation["locations_sha256"] != sha256(folder / "locations.json")
    ):
        raise ValueError("Comparison protocol or holdout differs from the recorded evidence")
    models = load_models(output, corpus, freeze["manifests"])
    record(
        folder / "opened.json",
        {
            "freeze_sha256": sha256(output / "freeze.json"),
            "reservation_sha256": sha256(folder / "reservation.json"),
        },
    )

    def exports() -> Iterator[ExportV2]:
        # All members of this separately generated population are held out. Its
        # internal catalog partition names are unused by the evaluation.
        for member in holdout.members:
            export = load_export_v2(folder / locations[member.source_namespace])
            yield member.bind_export(export).export

    result = compare(exports(), models)
    record(folder / "results.json", result)
    print("Held out:", result["pooled"], result["paired_bootstrap"], flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("reserve", "fit", "score"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True, help="Prepared profile-2.1 corpus")
    parser.add_argument("--exposure-index", type=Path, help="Existing exposure inventory")
    args = parser.parse_args()
    corpus = Corpus(args.corpus)
    if args.exposure_index and args.phase != "reserve":
        parser.error("--exposure-index applies only during reservation")
    if args.phase == "reserve":
        reserve(args.output, args.exposure_index, corpus)
    elif args.phase == "fit":
        fit(args.output, corpus)
    else:
        score(args.output, corpus)


if __name__ == "__main__":
    main()
