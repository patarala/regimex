"""Compare four logistic feature summaries before scoring a fresh paired price gap."""

import argparse
import json
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np
from compare_profile_21 import PROFILE, RUNTIME, Corpus, record, request, sha256
from evaluate_missing_data import dense_states, prepare_masked_members

from regimex.admin_contracts import GenerationSpec, RawGenerationPlan
from regimex.application.admin_computation import _generation
from regimex.application.analysis_computation import snapshot_inputs
from regimex.application.training import train_candidate
from regimex.backends.baseline import BaselineBackend
from regimex.benchmark_contracts import dataset_identity_keys
from regimex.data_catalog import (
    DatasetManifest,
    SnapshotGrid,
    SourceSnapshot,
    content_hash,
    create_snapshot_member,
    load_training_data,
)
from regimex.evaluation import _metrics, _segment_counts, scores
from regimex.export_robustness import NO_TRADES, WINDOW, masked_export
from regimex.exports import load_export_v2
from regimex.model_registry import ArchitectureRegistry, SummaryMode
from regimex.segmentation import annotate, decode
from regimex.storage import json_bytes
from regimex.terminal_training import register_completed_bundle

ORIGINAL = SummaryMode.LOCAL_AND_GLOBAL
GENERATION = GenerationSpec(
    dataset_id="baseline-context-evaluation",
    plan=RawGenerationPlan(seed=1947046183, experiment_count=48, step_count=512),
)
SELECTION = (
    "Highest mean pristine and price-gap validation interval macro-F1; first candidate wins ties."
)
GAP = "price_gap"
FINAL_DATASET_SHA256 = "010468ec0cdbe200c41fc7d77ee370039582de9447065588c0694dc945f9950f"


def read(path):
    return json.loads(path.read_bytes())


def identity_keys(members):
    return {content_hash(key) for key in dataset_identity_keys(tuple(members))}


def model_path(mode, corpus, output):
    return corpus / "baseline" if mode == ORIGINAL else output / mode


def compare(exports, model, *, interval_counts=None):
    counts = {name: np.zeros((3, 3), dtype=np.int64) for name in ("pristine", GAP)}
    expected, pristine, masked, distant = [], [], [], []
    for export, gap in exports:
        labels = annotate(SnapshotGrid.from_export(export), PROFILE)
        experiment = {}
        for name, source, states in (("pristine", export, pristine), (GAP, gap, masked)):
            inputs = snapshot_inputs(source)
            probabilities = model.predict(inputs, runtime=RUNTIME)
            experiment[name] = np.zeros((3, 3), dtype=np.int64)
            offset = 0
            for info, rows in zip(inputs.info, inputs.observations, strict=True):
                reference = tuple(
                    i for i in labels.intervals if i.instrument_id == info.instrument_id
                )
                predicted = decode(probabilities[offset : offset + len(rows)], rows, PROFILE)
                offset += len(rows)
                experiment[name] += _segment_counts(reference, predicted)
                states.append(dense_states(predicted, len(rows)))
                if name == "pristine":
                    expected.append(dense_states(reference, len(rows)))
                    steps = np.arange(len(rows))
                    # One further step allows for returns involving the missing price.
                    distant.append((steps < WINDOW[0] - 65) | (steps >= WINDOW[1] + 65))
            counts[name] += experiment[name]
        if interval_counts is not None:
            interval_counts.append(experiment)
    reference, original, incomplete = map(np.concatenate, (expected, pristine, masked))
    common = (reference < NO_TRADES) & (original < NO_TRADES) & (incomplete < NO_TRADES)
    far = common & np.concatenate(distant)
    changed = original != incomplete
    return {
        "experiments": len(exports),
        "instrument_series": len(expected),
        "classifiable_pristine": int((original < NO_TRADES).sum()),
        "classifiable_gap": int((incomplete < NO_TRADES).sum()),
        "pristine_interval_f1": _metrics(counts["pristine"])["macro_f1"],
        "gap_interval_f1": _metrics(counts[GAP])["macro_f1"],
        "common_steps": int(common.sum()),
        "changed_directions": int((changed & common).sum()),
        "distant_steps": int(far.sum()),
        "changed_distant": int((changed & far).sum()),
        "pristine_common_point_f1": scores(reference[common], original[common])["macro_f1"],
        "gap_common_point_f1": scores(reference[common], incomplete[common])["macro_f1"],
    }


def paired_intervals(original, candidate):
    rng = np.random.default_rng(1729)
    groups = rng.integers(0, len(original), size=(1000, len(original)))
    intervals = {}
    for scenario in ("pristine", GAP):
        left, right = (
            np.stack([experiment[scenario] for experiment in side])
            for side in (original, candidate)
        )
        deltas = [
            _metrics(right[sample].sum(axis=0))["macro_f1"]
            - _metrics(left[sample].sum(axis=0))["macro_f1"]
            for sample in groups
        ]
        intervals[scenario] = np.quantile(deltas, [0.025, 0.975]).tolist()
    return {"seed": 1729, "draws": len(groups), "interval_f1_delta_95pct": intervals}


def selection(reports):
    return max(
        SummaryMode,
        key=lambda mode: (
            (reports[mode]["pristine_interval_f1"] + reports[mode]["gap_interval_f1"]) / 2
        ),
    )


def protocol(corpus_path, corpus):
    root = Path(__file__).resolve().parents[1]
    paths = [
        root / "src/regimex" / name
        for name in (
            "baseline.py",
            "backends/baseline.py",
            "features.py",
            "segmentation.py",
            "model_registry.py",
            "export_robustness.py",
            "evaluation.py",
            "application/training.py",
        )
    ]
    paths += list(
        root / "scripts" / name
        for name in (
            "compare_profile_21.py",
            "evaluate_missing_data.py",
            "compare_baseline_context.py",
        )
    )
    return {
        "data_sha256": corpus.dataset.content_sha256,
        "original_manifest": read(corpus_path / "freeze.json")["bundles"]["baseline"],
        "generation": GENERATION.model_dump(mode="json"),
        "final_dataset_sha256": FINAL_DATASET_SHA256,
        "modes": list(SummaryMode),
        "selection": SELECTION,
        "comparison": {
            "scenario": GAP,
            "window": list(WINDOW),
            "context_sizes": [32, 128],
            "regularization": 1.0,
        },
        "profile": PROFILE.model_dump(mode="json"),
        "runtime": RUNTIME.model_dump(mode="json"),
        "source_sha256": content_hash({str(p.relative_to(root)): sha256(p) for p in paths}),
        "lockfile_sha256": sha256(root / "uv.lock"),
        "python": platform.python_version(),
        "packages": {
            name: version(name) for name in ("numpy", "scipy", "scikit-learn", "pydantic")
        },
    }


def fit(corpus_path, output):
    corpus = Corpus(corpus_path)
    output.mkdir(parents=True)
    plan = protocol(corpus_path, corpus)
    record(output / "plan.json", plan)
    train, annotations = corpus.references("train")
    val, val_annotations = corpus.references("validation")
    annotations.update(val_annotations)
    data = load_training_data(
        corpus.dataset,
        train,
        val,
        snapshot_reader=corpus.snapshot,
        reference_reader=lambda _, m: annotations[m.source_namespace],
    )
    pairs, variants = [], []
    members = {m.source_namespace: m for m in corpus.dataset.members}
    for item in data.validation:
        member = members[item.source_namespace]
        gap = masked_export(item.export, GAP)
        pairs.append((item.export, gap))
        source = SourceSnapshot(
            source_namespace=member.source_namespace, split_group=member.split_group, export=gap
        )
        variants.append(create_snapshot_member(source, member.partition))
    record(output / "validation-keys.json", sorted(identity_keys(variants)))

    reports, manifests = {}, {}
    for mode in SummaryMode:
        print(f"Validation: {mode}", flush=True)
        if mode == ORIGINAL:
            manifest = plan["original_manifest"]
        else:
            req = request("baseline", "baseline-context-" + plan["source_sha256"])
            req = req.model_copy(
                update={
                    "model_version_id": "baseline-" + mode,
                    "run_id": "baseline-" + mode,
                    "parameters": {
                        "regularization": [1.0],
                        "context_sizes": [32, 128],
                        "summaries": mode.value,
                    },
                }
            )
            fitted = train_candidate(
                data,
                req,
                output / mode,
                registry=ArchitectureRegistry.discover(),
                worker_id="baseline-comparison",
            )
            manifest = fitted.manifest_sha256
            register_completed_bundle(output / mode, manifest)
        model = BaselineBackend().load_bundle(
            model_path(mode, corpus_path, output), manifest_sha256=manifest, runtime=RUNTIME
        )
        path = model_path(mode, corpus_path, output)
        if (
            model.analysis != PROFILE
            or model.model.summaries != mode
            or model.model.context_sizes != (32, 128)
            or read(path / "validation.json")["regularization"] != 1.0
            or read(path / "manifest.json")["training"]["validation_references"]["content_sha256"]
            != val.content_sha256
            or read(path / "manifest.json")["training"]["labels"]["content_sha256"]
            != train.content_sha256
        ):
            raise ValueError("Restored comparison model differs from its profile or summary mode")
        reports[mode] = compare(pairs, model)
        manifests[mode] = manifest
        record(output / f"{mode}-validation.json", reports[mode])
        print(reports[mode], flush=True)
    selected = selection(reports)
    record(
        output / "freeze.json",
        {
            "selected": selected,
            "bundles": manifests,
            "plan_sha256": sha256(output / "plan.json"),
            "validation_keys_sha256": sha256(output / "validation-keys.json"),
            "validation_sha256": {m: sha256(output / f"{m}-validation.json") for m in SummaryMode},
        },
    )


def score(corpus, output, history):
    freeze = read(output / "freeze.json")
    plan = read(output / "plan.json")
    reports = {m: read(output / f"{m}-validation.json") for m in SummaryMode}
    if (
        freeze["plan_sha256"] != sha256(output / "plan.json")
        or freeze["validation_keys_sha256"] != sha256(output / "validation-keys.json")
        or any(
            sha256(output / f"{m}-validation.json") != freeze["validation_sha256"][m]
            for m in SummaryMode
        )
        or plan != protocol(corpus, Corpus(corpus))
        or freeze["bundles"][ORIGINAL] != plan["original_manifest"]
        or freeze["selected"] != selection(reports)
    ):
        raise ValueError("The protocol or validation evidence changed after selection")
    selected = SummaryMode(freeze["selected"])
    models = {
        mode: BaselineBackend().load_bundle(
            model_path(mode, corpus, output),
            manifest_sha256=freeze["bundles"][mode],
            runtime=RUNTIME,
        )
        for mode in dict.fromkeys((ORIGINAL, selected))
    }
    folder = output / "final"
    folder.mkdir()
    generated = _generation(GENERATION, folder, "baseline-context")
    dataset = DatasetManifest.model_validate_json(json_bytes(generated["manifest"]))
    paths = [folder / item["path"] for item in generated["snapshots"]]
    variants = prepare_masked_members(paths, dataset.members, scenarios=(GAP,))
    keys = identity_keys((*dataset.members, *variants))
    seeds = {item.split_group.seed for item in dataset.members}
    # A replay with the published dataset digest can run without the private exposure inventory.
    if dataset.content_sha256 != FINAL_DATASET_SHA256:
        raise ValueError("Final population differs from its recorded digest")
    val_keys = set(read(output / "validation-keys.json"))
    if keys & val_keys:
        raise ValueError("Final exports overlap validation")
    receipt = {
        "dataset": dataset.content_sha256,
        "new_seeds": sorted(seeds),
        "new_keys": sorted(keys),
    }
    if history:
        index = read(history)
        if keys.intersection(index["identity_keys_sha256"]) or seeds.intersection(
            index["excluded_seeds"]
        ):
            raise ValueError("Final exports overlap prior exposures")
        receipt["history_sha256"] = sha256(history)
        record(
            folder / "exposure-index.json",
            {
                **index,
                "previous_sha256": sha256(history),
                "scope": "Prior exposures, validation gaps and the frozen baseline evaluation.",
                "current_dataset_paths": ["dataset.json"],
                "excluded_seeds": sorted(set(index["excluded_seeds"]) | seeds),
                "identity_keys_sha256": sorted(
                    set(index["identity_keys_sha256"]) | val_keys | keys
                ),
            },
        )
    record(folder / "dataset.json", generated["manifest"])
    record(folder / "reservation.json", receipt)
    record(
        folder / "opened.json",
        {
            "freeze_sha256": sha256(output / "freeze.json"),
            "reservation_sha256": sha256(folder / "reservation.json"),
        },
    )
    pairs = [(load_export_v2(p), load_export_v2(p.parent / GAP / p.name)) for p in paths]
    counts = {mode: [] for mode in models}
    result = {
        mode: compare(pairs, model, interval_counts=counts[mode]) for mode, model in models.items()
    }
    result["paired"] = paired_intervals(counts[ORIGINAL], counts[selected])
    record(folder / "results.json", result)
    print(result, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("fit", "score"))
    parser.add_argument("--corpus", type=Path, required=True, help="Profile-2.1 comparison folder")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--exposure-index",
        type=Path,
        help="Exclude prior experiments when opening a fresh evaluation",
    )
    args = parser.parse_args()
    if args.phase == "fit":
        if args.exposure_index:
            parser.error("--exposure-index is only valid during scoring")
        fit(args.corpus, args.output)
    else:
        score(args.corpus, args.output, args.exposure_index)


if __name__ == "__main__":
    main()
