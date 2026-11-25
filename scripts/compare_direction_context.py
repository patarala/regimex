"""Train observation-only direction classifiers and compare one frozen choice with saved TCNs."""

import argparse
from collections import Counter
from dataclasses import replace
from importlib import import_module
from pathlib import Path

import numpy as np
from check_pause_overlay_on_tcn import BUNDLES
from compare_profile_21 import RUNTIME, Corpus, sha256, software, write
from compare_selection_panels import experiment_id
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
from confirm_tcn_widths import audit_population
from fit_pause_targets import DATASET

from regimex.admin_contracts import GenerationSpec, RawGenerationPlan
from regimex.backends.tcn import TCNBackend
from regimex.data_catalog import _load_partition, content_hash
from regimex.evaluation import dense_labels
from regimex.storage import json_bytes, output_directory, write_new_file

TRAINING = Path("var/verification/training-populations")
DEFAULT_BUNDLES = {
    "mixed-192": TRAINING / "fitted/mixed-192/bundle",
    "tcn-0.1.0": Path("release/models/tcn"),
}
MODULES = {"context": "direction_context_model", "adaptive": "adaptive_direction_model"}
CONTROL = "mixed-192"
RELEASE = "tcn-0.1.0"
POOLS = {"validation": (2_628_751_000, 24), "holdout": (3_793_641_000, 48)}
CANDIDATES = (CONTROL, *MODULES, *(f"{name}+tcn" for name in MODULES))


def specs(pool):
    seed, count = POOLS[pool]
    return tuple(
        GenerationSpec(
            dataset_id=f"direction-context-{pool}-{steps}",
            plan=RawGenerationPlan(seed=seed + steps, experiment_count=count, step_count=steps),
        )
        for steps in LENGTHS
    )


def protocol():
    return {
        "cohorts": {pool: [s.model_dump(mode="json") for s in specs(pool)] for pool in POOLS},
        "training_dataset": DATASET,
        "bundle_sha256": BUNDLES,
        "reference": PROFILE.model_dump(mode="json"),
        "runtime": RUNTIME.model_dump(mode="json"),
        "candidates": list(CANDIDATES),
        "blend": "Equal arithmetic mean of candidate and mixed-192 probabilities before decoding.",
        "selection": "Highest pooled validation interval macro-F1 at IoU 0.5, first on ties.",
        "release_comparison": (
            "Score the release with its saved decoder as well as the common 2.2 decoder."
        ),
        "bootstrap": {"draws": DRAWS, "seed": DRAW_SEED, "unit": "whole experiment within length"},
        "qualification": (
            "The new candidate must win validation. The paired 95% lower bound on final "
            "interval macro-F1 gain must exceed zero against both mixed-192 and the "
            "native release. "
            "Allow at most 0.02 loss of point macro-F1 and 0.03 loss in any direction's "
            "interval F1 or any length's interval macro-F1 against mixed-192."
        ),
    }


def reserve(output, training, prior):
    development = Corpus(training, expected_digest=DATASET)
    with output_directory(output) as stage:
        write(stage / "plan.json", protocol())
        pools = {}
        previous = development.dataset
        for pool, (_, per_length) in POOLS.items():
            folder = stage / pool
            folder.mkdir()
            cohort = generate(folder, specs(pool), f"direction-context-{pool}")
            expected = {(s, n): per_length // 2 for s in LENGTHS for n in (1, 2)}
            if (
                Counter((m.step_count, len(m.instrument_ids)) for m in cohort.dataset.members)
                != expected
            ):
                raise ValueError("Unexpected membership of the generated cohort")
            audit = audit_population(development.dataset, cohort.dataset, prior)
            if pool == "holdout":
                audit_population(previous, cohort.dataset, None)
            previous = cohort.dataset
            pools[pool] = {
                **audit,
                "locations_sha256": sha256(folder / "locations.json"),
            }
        write(
            stage / "reservation.json",
            {
                "plan_sha256": sha256(stage / "plan.json"),
                "training_locations_sha256": sha256(training / "locations.json"),
                "pools": pools,
            },
        )


def checked_reservation(output):
    receipt = read(output / "reservation.json")
    if receipt["plan_sha256"] != content_hash(protocol()) or receipt["plan_sha256"] != sha256(
        output / "plan.json"
    ):
        raise ValueError("Registered comparison settings changed")
    return receipt


def software_fingerprint():
    scripts = (
        Path(__file__).stem,
        "check_pause_overlay_on_tcn",
        "compare_selection_panels",
        "compare_temporal_architectures",
        "confirm_training_populations",
        "compare_tcn_slopes",
        "compare_tcn_widths",
        "confirm_tcn_widths",
        "fit_pause_targets",
        *MODULES.values(),
    )
    return {
        **software(),
        "scripts": {s: sha256(Path(__file__).with_name(s + ".py")) for s in scripts},
    }


def load_evaluation(corpus, partitions):
    samples = []
    for part in partitions:
        refs, labels = corpus.references(part, profile=PROFILE)
        samples.extend(
            _load_partition(
                corpus.dataset,
                refs,
                snapshot_reader=corpus.snapshot,
                reference_reader=lambda _, member, table=labels: table[member.source_namespace],
            )
        )
    evaluation = evaluator(tuple(samples))
    expected = {
        experiment_id(m): m.step_count for m in corpus.dataset.members if m.partition in partitions
    }
    if {i.experiment_id: i.step_count for i in evaluation.inputs.info} != expected:
        raise ValueError("Scored experiment identities differ from reserved membership")
    return evaluation


def open_pool(output, receipt, pool):
    info = receipt["pools"][pool]
    folder = output / pool
    if sha256(folder / "locations.json") != info["locations_sha256"]:
        raise ValueError("Snapshot locations differ from the reservation")
    return load_evaluation(
        Corpus(folder, expected_digest=info["dataset_sha256"]), ("train", "validation", "test")
    )


def matrix(module, inputs):
    blocks = []
    for observations in inputs.observations:
        block = module.features(observations, PROFILE)
        if block.ndim != 2 or len(block) != len(observations):
            raise ValueError("Features must preserve the original sequence length")
        blocks.append(block)
    return np.concatenate(blocks)


def targets(evaluation):
    return np.concatenate(
        [
            dense_labels(ref, info.step_count)
            for ref, info in zip(evaluation.references, evaluation.inputs.info, strict=True)
        ]
    )


def train(output, training):
    receipt = checked_reservation(output)
    if sha256(training / "locations.json") != receipt["training_locations_sha256"]:
        raise ValueError("Training snapshot locations changed")
    cohort = Corpus(training, expected_digest=DATASET)
    evaluation = load_evaluation(cohort, ("train",))
    y = targets(evaluation)
    environment = software_fingerprint()
    with output_directory(output / "fitted") as stage:
        weights = {}
        for name, module_name in MODULES.items():
            module = import_module(module_name)
            x = matrix(module, evaluation.inputs)
            model = module.fit(x, y)
            path = stage / (name + ".npz")
            model.save(path)
            restored = module.DirectionContextModel.load(path)
            # Verify restoration across the full fit population, including sparse rows.
            np.testing.assert_allclose(
                restored.predict(x), model.predict(x), atol=1e-12, rtol=1e-12
            )
            weights[name] = sha256(path)
            print(f"Fitted {name} on {len(y)} rows with {x.shape[1]} features", flush=True)
            del x
        if environment != software_fingerprint() or checked_reservation(output) != receipt:
            raise ValueError("Fitting inputs changed during training")
        write(
            stage / "fit.json",
            {
                "reservation_sha256": sha256(output / "reservation.json"),
                "software": environment,
                "weights": weights,
                "labels_sha256": content_hash(y.tolist()),
                "classifiable_rows": int((y >= 0).sum()),
            },
        )


def check_fit(output):
    fitted = output / "fitted"
    fit = read(fitted / "fit.json")
    if (
        fit["software"] != software_fingerprint()
        or fit["reservation_sha256"] != sha256(output / "reservation.json")
        or fit["weights"] != {name: sha256(fitted / (name + ".npz")) for name in MODULES}
    ):
        raise ValueError("Learned model, preprocessing or comparison inputs changed")
    return fit


def probability_metrics(y, probabilities):
    keep = y >= 0
    classes = np.eye(3)[y[keep]]
    observed = probabilities[keep]
    if not len(observed):
        return {"rows": 0, "brier": None, "log_loss": None}
    return {
        "rows": len(observed),
        "brier": float(np.square(observed - classes).sum(axis=1).mean()),
        "log_loss": float(
            -np.log(np.maximum(observed[np.arange(len(observed)), y[keep]], 1e-15)).mean()
        ),
    }


def load_predictors(output, names, bundles=DEFAULT_BUNDLES):
    """Load the selected weights once; the returned callable accepts observation inputs only."""
    names = tuple(dict.fromkeys(names))
    allowed = {*CANDIDATES, RELEASE, "release-native"}
    if not names or set(names) - allowed:
        raise ValueError("Unknown or empty selection of frozen models")
    needs_control = CONTROL in names or any(name.endswith("+tcn") for name in names)
    needs_release = RELEASE in names or "release-native" in names
    tcns = {
        name: TCNBackend().load_bundle(
            bundles[name], manifest_sha256=BUNDLES[name], runtime=RUNTIME
        )
        for name, needed in ((CONTROL, needs_control), (RELEASE, needs_release))
        if needed
    }
    profiles = {name: model.analysis for name, model in tcns.items()}
    if needs_control and profiles[CONTROL] != PROFILE:
        raise ValueError("Strong control has a different analysis profile")
    learned = {}
    for name, module_name in MODULES.items():
        if name in names or name + "+tcn" in names:
            module = import_module(module_name)
            learned[name] = (
                module,
                module.DirectionContextModel.load(output / "fitted" / (name + ".npz")),
            )

    def predict(inputs):
        probabilities = {
            name: model.predict(inputs, runtime=RUNTIME) for name, model in tcns.items()
        }
        if needs_release:
            probabilities["release-native"] = probabilities[RELEASE]
        for name, (module, model) in learned.items():
            candidate = model.predict(matrix(module, inputs))
            probabilities[name] = candidate
            if name + "+tcn" in names:
                probabilities[name + "+tcn"] = 0.5 * (candidate + probabilities[CONTROL])
        return {name: probabilities[name] for name in names}

    return predict, profiles


def measure(output, evaluation, bundles, names):
    predict, profiles = load_predictors(output, names, bundles)
    probabilities = predict(evaluation.inputs)
    results, counts = {}, {}
    y = targets(evaluation)
    for name in probabilities:
        scorer = (
            replace(evaluation, analysis=profiles[RELEASE])
            if name == "release-native"
            else evaluation
        )
        counts[name] = experiment_counts(scorer, probabilities[name])
        _, lengths, rows = counts[name]
        results[name] = {
            **summary(rows),
            "by_length": {str(size): summary(rows[lengths == size]) for size in LENGTHS},
            "probabilities": probability_metrics(y, probabilities[name]),
            "decoder_profile": scorer.analysis.version,
        }
    return results, counts


def choose(results):
    return max(CANDIDATES, key=lambda name: results[name]["interval_macro_f1"])


def freeze(output, bundles):
    path = output / "freeze.json"
    if path.exists() or (output / "opened.json").exists():
        raise ValueError("Validation settings are already frozen")
    receipt, fit = checked_reservation(output), check_fit(output)
    validation = open_pool(output, receipt, "validation")
    results, _ = measure(output, validation, bundles, (*CANDIDATES, RELEASE, "release-native"))
    if check_fit(output) != fit:
        raise ValueError("Training evidence changed during validation")
    winner = choose(results)
    write_new_file(
        path,
        json_bytes(
            {
                "fit_sha256": sha256(output / "fitted/fit.json"),
                "winner": winner,
                "scores": results,
            }
        ),
    )
    print(f"Selected {winner}", flush=True)


def read_freeze(output):
    check_fit(output)
    freeze = read(output / "freeze.json")
    if freeze["fit_sha256"] != sha256(output / "fitted/fit.json") or freeze["winner"] != choose(
        freeze["scores"]
    ):
        raise ValueError("The selected candidate differs from the validation scores")
    return freeze


def qualification(winner, results, comparisons):
    selected, baseline = results[winner], results[CONTROL]
    return {
        "new_validation_winner": winner != CONTROL,
        "gain_over_both_controls": all(
            comparisons[name]["interval_macro_f1"]["interval_95pct"][0] > 0
            for name in (CONTROL, "release-native")
        ),
        "point_f1": selected["point_macro_f1"] is not None
        and baseline["point_macro_f1"] is not None
        and selected["point_macro_f1"] >= baseline["point_macro_f1"] - 0.02,
        "per_class_interval_f1": all(
            selected["interval_per_class"][k]["f1"] >= v["f1"] - 0.03
            for k, v in baseline["interval_per_class"].items()
        ),
        "per_length_interval_f1": all(
            selected["by_length"][k]["interval_macro_f1"] >= v["interval_macro_f1"] - 0.03
            for k, v in baseline["by_length"].items()
        ),
    }


def score(output, bundles):
    from confirm_training_populations import contrast

    receipt, freeze = checked_reservation(output), read_freeze(output)
    winner = freeze["winner"]
    opening = {"freeze_sha256": sha256(output / "freeze.json")}
    write_new_file(output / "opened.json", json_bytes(opening))
    holdout = open_pool(output, receipt, "holdout")
    results, counts = measure(
        output, holdout, bundles, (CONTROL, RELEASE, "release-native", winner)
    )
    comparisons = {
        name: contrast(counts[name], counts[winner]) for name in (CONTROL, "release-native")
    }
    checks = qualification(winner, results, comparisons)
    if read_freeze(output) != freeze or sha256(output / "freeze.json") != opening["freeze_sha256"]:
        raise ValueError("The frozen choice changed during confirmation")
    write_new_file(
        output / "results.json",
        json_bytes(
            {
                **opening,
                "winner": winner,
                "scores": results,
                "paired_vs": comparisons,
                "checks": checks,
                "qualified": all(checks.values()),
                "experiment_counts": {
                    name: dict(zip(ids, rows.tolist(), strict=True))
                    for name, (ids, _, rows) in counts.items()
                },
            }
        ),
    )
    print(f"Closed comparison: {winner}, {checks}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_subparsers(dest="phase", required=True)
    for name in ("reserve", "train", "freeze", "score"):
        step = phases.add_parser(name)
        step.add_argument("output", type=Path)
        if name in {"reserve", "train"}:
            step.add_argument("--training", type=Path, default=TRAINING / "development/mixed-192")
        if name == "reserve":
            step.add_argument(
                "--prior",
                type=Path,
                required=True,
                help="An exposure index with seed and identity digests.",
            )
        if name in {"freeze", "score"}:
            step.add_argument("--release", type=Path, default=DEFAULT_BUNDLES[RELEASE])
            step.add_argument("--control", type=Path, default=DEFAULT_BUNDLES[CONTROL])
    args = parser.parse_args()
    if args.phase == "reserve":
        reserve(args.output, args.training, args.prior)
    elif args.phase == "train":
        train(args.output, args.training)
    elif args.phase == "freeze":
        freeze(args.output, {CONTROL: args.control, RELEASE: args.release})
    else:
        score(args.output, {CONTROL: args.control, RELEASE: args.release})


if __name__ == "__main__":
    main()
