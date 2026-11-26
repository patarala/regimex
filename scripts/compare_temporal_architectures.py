"""Compare temporal classifiers with one decoder, a fixed training budget and a fresh holdout."""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from statistics import mean

import numpy as np
import torch
from compare_profile_21 import RUNTIME, Corpus, sha256, software, write
from compare_tcn_slopes import metrics
from compare_tcn_widths import DATASET_SHA256, POLICY, PROFILE, generate, training_data
from confirm_tcn_widths import audit_population
from temporal_candidates import CANDIDATES

from regimex.admin_contracts import GenerationSpec, RawGenerationPlan
from regimex.application.training import SharedSelectionEvaluator, _references, observation_inputs
from regimex.backends.baseline import BaselineBackend
from regimex.backends.tcn import (
    FEATURE_VERSION,
    LEGACY_FEATURE_VERSION,
    fit_preprocessing,
    prepare_observations,
)
from regimex.data_catalog import _load_partition, content_hash
from regimex.evaluation import (
    _metrics,
    _point_counts,
    _segment_counts,
    dense_labels,
    matched_interval_indices,
    point_macro_f1_from_counts,
)
from regimex.model_registry import TrainingRuntime
from regimex.storage import output_directory
from regimex.temporal import TemporalPreprocessor, predict_tcn
from regimex.temporal_updates import fit_temporal_updates

ROOT = Path(__file__).resolve().parents[1]
LENGTHS = (256, 512, 1024)
PER_LENGTH = 40
DRAW_SEED, DRAWS = 1729, 2000


def read(path):
    return json.loads(path.read_bytes())


def plan():
    return {
        "development": DATASET_SHA256,
        "profile": PROFILE.model_dump(mode="json"),
        "candidates": [asdict(candidate) for candidate in CANDIDATES],
        "policy": POLICY.model_dump(mode="json"),
        "dropout": 0.15,
        "smoothing": 0.05,
        "test": {"plan_seed": 745184311, "lengths": LENGTHS, "per_length": PER_LENGTH},
        "primary": "Pooled decoded interval macro-F1 at IoU 0.5.",
        "selection": (
            "Best validation checkpoint per seed, then highest score across seeds; earliest tie. "
            "Rank configurations by mean of seed bests."
        ),
        "secondary": [
            "point macro-F1",
            "sideways recall below 64 steps",
            "sideways precision",
            "interval macro-F1 by sequence length",
        ],
        "paired_comparisons": {
            "control": "tcn-32",
            "draw_unit": "experiment",
            "draws": DRAWS,
            "seed": DRAW_SEED,
        },
    }


def reserve(output, development, exposure):
    protocol = plan()
    populations = tuple(
        GenerationSpec(
            dataset_id=f"temporal-confirmation-{steps}",
            plan=RawGenerationPlan(
                seed=protocol["test"]["plan_seed"] + steps,
                experiment_count=PER_LENGTH,
                step_count=steps,
            ),
        )
        for steps in LENGTHS
    )
    with output_directory(output) as directory:
        write(directory / "plan.json", protocol)
        (directory / "holdout").mkdir()
        holdout = generate(directory / "holdout", populations, "temporal-confirmation")
        audit = audit_population(development.dataset, holdout.dataset, exposure)
        write(
            directory / "reservation.json",
            {
                "plan": sha256(directory / "plan.json"),
                "environment": scientific_digests(),
                **audit,
            },
        )


def checked_plan(output):
    reservation = read(output / "reservation.json")
    if not reservation["plan"] == sha256(output / "plan.json") == content_hash(plan()):
        raise ValueError("The registered comparison plan changed")
    if reservation["environment"] != scientific_digests():
        raise ValueError("Source code or runtime differs from the reservation")
    return reservation


def evaluator(samples):
    return SharedSelectionEvaluator(
        observation_inputs(samples), _references(samples), PROFILE, "automatic"
    )


def experiment_counts(evaluation, probabilities):
    _, decoded = evaluation._decode(probabilities)
    groups = {}
    lengths = {}
    for info, reference, prediction in zip(
        evaluation.inputs.info, evaluation.references, decoded, strict=True
    ):
        group = groups.setdefault(info.experiment_id, np.zeros(20, dtype=np.int64))
        lengths[info.experiment_id] = info.step_count
        group[:9] += _segment_counts(reference, prediction).ravel()
        flat = [i for i in reference if i.direction == "sideways"]
        matches = matched_interval_indices(
            flat, [i for i in prediction if i.direction == "sideways"]
        )
        short = {idx for idx, interval in enumerate(flat) if interval.stop - interval.start < 64}
        group[9:11] += (len(short.intersection(matches)), len(short))
        point, _, _ = _point_counts(
            dense_labels(reference, info.step_count), dense_labels(prediction, info.step_count)
        )
        group[11:] += point.ravel()
    ids = sorted(groups)
    return ids, np.array([lengths[k] for k in ids]), np.stack([groups[k] for k in ids])


def summary(counts):
    pooled = counts.sum(axis=0)
    point_f1 = point_macro_f1_from_counts(pooled[11:].reshape(3, 3))
    return {
        **metrics(pooled[:11]),
        "point_macro_f1": point_f1 if np.isfinite(point_f1) else None,
        "interval_per_class": _metrics(pooled[:9].reshape(3, 3))["per_class"],
    }


def paired(control, candidate):
    # Sample whole experiments within each length to keep the registered mixture fixed.
    ids, lengths, left = control
    other_ids, other_lengths, right = candidate
    if ids != other_ids or not np.array_equal(lengths, other_lengths):
        raise ValueError("Paired comparison needs identical experiment membership")
    rng = np.random.default_rng(DRAW_SEED)
    groups = [np.flatnonzero(lengths == size) for size in LENGTHS]
    draws = np.concatenate([rng.choice(g, size=(DRAWS, len(g))) for g in groups], axis=1)
    keys = ("interval_macro_f1", "short_sideways_recall", "sideways_precision", "point_macro_f1")
    differences = {k: [] for k in keys}
    for sample in draws:
        a, b = summary(left[sample]), summary(right[sample])
        for key in keys:
            if a[key] is not None and b[key] is not None:
                differences[key].append(b[key] - a[key])
    return {
        key: {
            "defined_draws": len(values),
            "interval_95pct": np.quantile(values, [0.025, 0.975]).tolist()
            if len(values) == DRAWS
            else None,
        }
        for key, values in differences.items()
    }


def scientific_digests():
    scripts = (
        "compare_temporal_architectures.py",
        "temporal_candidates.py",
        "compare_tcn_slopes.py",
        "compare_tcn_widths.py",
        "confirm_tcn_widths.py",
        "compare_profile_21.py",
    )
    return {**software(), "scripts": {name: sha256(ROOT / "scripts" / name) for name in scripts}}


def baseline(development, digest):
    folder = development.output / "baseline"
    model = BaselineBackend().load_bundle(folder, manifest_sha256=digest, runtime=RUNTIME)
    manifest = read(folder / "manifest.json")
    if manifest["training"]["dataset_sha256"] != DATASET_SHA256 or model.analysis != PROFILE:
        raise ValueError("The baseline must come from the same profile and development population")
    return model


def fit(output, development):
    reservation = checked_plan(output)
    data = training_data(development)
    validation = evaluator(data.validation)
    inputs = observation_inputs(data.train)
    refs = _references(data.train)
    targets = np.concatenate(
        [dense_labels(ref, info.step_count) for ref, info in zip(refs, inputs.info, strict=True)]
    )
    pre, training = fit_preprocessing(inputs)
    valid = prepare_observations(validation.inputs, pre)
    runtime = TrainingRuntime(threads=2)
    with output_directory(output / "fitted") as dest:
        np.savez(dest / "preprocessing.npz", fill=pre.fill, mean=pre.mean, scale=pre.scale)
        evidence = {
            "plan": reservation["plan"],
            "holdout": reservation["dataset_sha256"],
            "preprocessing": sha256(dest / "preprocessing.npz"),
            "feature_version": FEATURE_VERSION,
            "reservation": sha256(output / "reservation.json"),
            "candidates": {},
        }
        digest = read(development.output / "baseline-validation.json")["bundle_sha256"]
        linear = baseline(development, digest)
        evidence["baseline"] = {
            "manifest": digest,
            "validation": summary(
                experiment_counts(validation, linear.predict(validation.inputs, runtime=RUNTIME))[2]
            ),
        }
        for candidate in CANDIDATES:
            started = time.monotonic()
            model, history, seed = fit_temporal_updates(
                training,
                targets,
                valid,
                build=candidate.build,
                interval_weight_power=candidate.interval_power,
                smoothing=0.05,
                policy=POLICY,
                runtime=runtime,
                scorer=validation,
            )
            fit_seconds = time.monotonic() - started
            bests = []
            for repeat in POLICY.seeds:
                entries = [
                    row
                    for row in history
                    if row["seed"] == repeat and "validation_segment_macro_f1" in row
                ]
                best = max(entries, key=lambda entry: entry["validation_segment_macro_f1"])
                bests.append(
                    {
                        "seed": repeat,
                        "update": best["update"],
                        "score": best["validation_segment_macro_f1"],
                    }
                )
            path = dest / (candidate.name + ".npz")
            np.savez(
                path, **{key: tensor.detach().numpy() for key, tensor in model.state_dict().items()}
            )
            values = summary(experiment_counts(validation, predict_tcn(model, valid))[2])
            if not np.isclose(
                values["interval_macro_f1"], max(row["score"] for row in bests), atol=1e-12, rtol=0
            ):
                raise ValueError("Recomputed validation score differs from the selected checkpoint")
            row = {
                "weights": sha256(path),
                "seed": seed,
                "parameters": sum(p.numel() for p in model.parameters()),
                "fit_seconds": fit_seconds,
                "per_seed": bests,
                "mean_score": mean(row["score"] for row in bests),
                "validation": values,
            }
            evidence["candidates"][candidate.name] = row
            print(
                candidate.name,
                "validation",
                round(row["mean_score"], 4),
                "in",
                round(fit_seconds, 1),
                "s",
                flush=True,
            )
        evidence["validation_choice"] = max(
            evidence["candidates"], key=lambda name: evidence["candidates"][name]["mean_score"]
        )
        write(dest / "freeze.json", evidence)


def score(output, development):
    reservation = checked_plan(output)
    dest = output / "fitted"
    freeze = read(dest / "freeze.json")
    if (
        freeze["plan"] != reservation["plan"]
        or freeze["holdout"] != reservation["dataset_sha256"]
        or freeze["reservation"] != sha256(output / "reservation.json")
        or sha256(dest / "preprocessing.npz") != freeze["preprocessing"]
    ):
        raise ValueError("Frozen comparison inputs changed")
    feature_version = freeze.get("feature_version", LEGACY_FEATURE_VERSION)
    if feature_version not in (LEGACY_FEATURE_VERSION, FEATURE_VERSION):
        raise ValueError("Unrecognized TCN feature version in the frozen comparison")
    with np.load(dest / "preprocessing.npz", allow_pickle=False) as arrays:
        pre = TemporalPreprocessor(
            arrays["fill"],
            arrays["mean"],
            arrays["scale"],
            legacy_regression=feature_version == LEGACY_FEATURE_VERSION,
        )
    models = {}
    for candidate in CANDIDATES:
        path = dest / (candidate.name + ".npz")
        if sha256(path) != freeze["candidates"][candidate.name]["weights"]:
            raise ValueError(f"Frozen weights changed for {candidate.name}")
        net = candidate.build(len(pre.mean))
        with np.load(path, allow_pickle=False) as arrays:
            net.load_state_dict({k: torch.from_numpy(arrays[k]) for k in arrays.files}, strict=True)
        models[candidate.name] = net
    linear = baseline(development, freeze["baseline"]["manifest"])
    holdout = Corpus(output / "holdout", expected_digest=reservation["dataset_sha256"])
    samples = []
    for partition in ("train", "validation", "test"):
        references, labels = holdout.references(partition, profile=PROFILE)
        samples.extend(
            _load_partition(
                holdout.dataset,
                references,
                snapshot_reader=holdout.snapshot,
                reference_reader=lambda _, m, annotations=labels: annotations[m.source_namespace],
            )
        )
    evaluation = evaluator(tuple(samples))
    prepared = prepare_observations(evaluation.inputs, pre)
    torch.set_num_threads(RUNTIME.threads)
    torch.use_deterministic_algorithms(True)
    counts = {
        "baseline": experiment_counts(
            evaluation, linear.predict(evaluation.inputs, runtime=RUNTIME)
        )
    }
    for name, model in models.items():
        counts[name] = experiment_counts(evaluation, predict_tcn(model, prepared, batch_size=8))
    results = {}
    for name, (ids, lengths, rows) in counts.items():
        if len(ids) != PER_LENGTH * len(LENGTHS) or any(
            (lengths == size).sum() != PER_LENGTH for size in LENGTHS
        ):
            raise ValueError("Unexpected held-out cohort membership")
        results[name] = {
            **summary(rows),
            "by_length": {size: summary(rows[lengths == size]) for size in LENGTHS},
            "experiment_counts": dict(zip(ids, rows.tolist(), strict=True)),
        }
        if name != "tcn-32":
            results[name]["paired_vs_control"] = paired(counts["tcn-32"], counts[name])
    write(
        output / "results.json",
        {
            "freeze": sha256(dest / "freeze.json"),
            "experiments": len(samples),
            "instrument_sequences": len(evaluation.inputs.info),
            "results": results,
        },
    )
    for name, row in results.items():
        print(
            name,
            "interval",
            round(row["interval_macro_f1"], 4),
            "short sideways",
            row["short_sideways_matched"],
            "/",
            row["short_sideways_references"],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("reserve", "fit", "score"))
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--development",
        required=True,
        type=Path,
        help="Completed profile 2.2 width comparison directory",
    )
    parser.add_argument(
        "--exposure-index",
        type=Path,
        help="Additional prior seeds and identity digests, used at reservation",
    )
    args = parser.parse_args()
    development = Corpus(args.development, expected_digest=DATASET_SHA256)
    if args.phase == "reserve":
        reserve(args.output, development, args.exposure_index)
    elif args.phase == "fit":
        fit(args.output, development)
    else:
        score(args.output, development)
