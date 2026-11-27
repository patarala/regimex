"""Compare two forest training recipes on the same expanded population of market experiments."""

import argparse
from collections import Counter
from itertools import batched
from pathlib import Path

import balanced_forest_training
import compare_adaptive_intervals as previous_trial
import numpy as np
from build_exposure_index import build
from compare_adaptive_intervals import BUNDLE, BUNDLE_SHA, CONTROL, TRAINING, qualification
from compare_direction_context import matrix, probability_metrics, targets
from compare_profile_21 import RUNTIME, Corpus, sha256, write
from compare_selection_panels import experiment_id
from compare_tcn_widths import PROFILE, generate
from compare_temporal_architectures import LENGTHS, evaluator, experiment_counts, read, summary
from confirm_tcn_widths import audit_population
from confirm_training_populations import contrast
from fit_pause_targets import DATASET

from regimex.admin_contracts import GenerationSpec, RawGenerationPlan
from regimex.backends import adaptive_forest
from regimex.backends.adaptive import AdaptiveBackend
from regimex.data_catalog import (
    DatasetManifest,
    LabeledSnapshot,
    SnapshotGrid,
    ValidatedAnnotations,
    content_hash,
    seal_dataset,
)
from regimex.directions import DIRECTION_NAMES
from regimex.evaluation import dense_labels
from regimex.segmentation import annotate
from regimex.storage import json_bytes, output_directory, write_new_file

CANDIDATES = (CONTROL, "larger", "balanced")
POOLS = {
    "extension": (1_743_832_000, 64, "train"),
    "validation": (2_297_145_000, 32, "validation"),
    "holdout": (3_943_261_000, 80, "test"),
}


def specs(pool):
    base, count, _ = POOLS[pool]
    return tuple(
        GenerationSpec(
            dataset_id=f"forest-mix-{pool}-{length}",
            plan=RawGenerationPlan(seed=base + length, experiment_count=count, step_count=length),
        )
        for length in LENGTHS
    )


def protocol():
    return {
        **previous_trial.protocol(),
        "cohorts": {pool: [s.model_dump(mode="json") for s in specs(pool)] for pool in POOLS},
        "candidates": list(CANDIDATES),
        "training": {
            "original": "Only the train partition from mixed-192.",
            "extension": "Every extension member is assigned to train before fitting.",
            "larger": "Original adaptive forest fitted on the expanded population.",
            "balanced": "Same training rows. Within each class, use systematic sampling weighted "
            "by clip(sqrt(128 / run_length), 0.5, 2) on each instrument; pair every sampled "
            "row with its price-direction reflection. Retain original features and forest "
            "settings.",
        },
    }


def register_history(previous, transfer):
    receipt = previous_trial.checked_reservation(previous)
    transfer_plan = read(transfer / "plan.json")
    if (
        transfer_plan["native_reservation_sha256"] != sha256(previous / "reservation.json")
        or transfer_plan["disjoint_from"]["prior_sha256"] != receipt["prior_sha256"]
        or read(transfer / "results.json")["plan_sha256"] != sha256(transfer / "plan.json")
    ):
        raise ValueError("Earlier transfer evidence does not match the native reservation")
    catalogs = []
    for pool, info in receipt["pools"].items():
        path = previous / pool / "dataset.json"
        if DatasetManifest.for_modeling(path.read_bytes()).content_sha256 != info["dataset_sha256"]:
            raise ValueError("Earlier evaluation population changed")
        catalogs.append(path)
    inventory = build(catalogs, [previous / "prior.json"])
    inventory["excluded_seeds"] = sorted(
        set(inventory["excluded_seeds"]).union(transfer_plan["protocol"]["groups"])
    )
    return inventory


def reserve(output, training, previous, transfer):
    original = Corpus(training, expected_digest=DATASET)
    retained = [m for m in original.dataset.members if m.partition == "train"]
    if len(retained) != 192:
        raise ValueError("Expected 192 original training experiments")
    inventory = register_history(previous, transfer)
    with output_directory(output) as stage:
        write(stage / "plan.json", protocol())
        write(stage / "prior.json", inventory)
        catalogs = []
        pools = {}
        assigned = list(retained)
        for pool, (_, count, partition) in POOLS.items():
            folder = stage / pool
            folder.mkdir()
            catalog = generate(folder, specs(pool), f"forest-mix-{pool}").dataset
            mix = Counter((m.step_count, len(m.instrument_ids)) for m in catalog.members)
            if mix != {(n, k): count // 2 for n in LENGTHS for k in (1, 2)}:
                raise ValueError("Length or instrument counts differ from the registered mixture")
            audit = audit_population(original.dataset, catalog, stage / "prior.json")
            for other in catalogs:
                audit_population(other, catalog, None)
            catalogs.append(catalog)
            assigned.extend(m.model_copy(update={"partition": partition}) for m in catalog.members)
            pools[pool] = {**audit, "locations_sha256": sha256(folder / "locations.json")}
        dataset = seal_dataset("forest-training-mix", tuple(assigned))
        write(stage / "dataset.json", dataset.model_dump(mode="json"))
        write(
            stage / "reservation.json",
            {
                "plan_sha256": sha256(stage / "plan.json"),
                "prior_sha256": sha256(stage / "prior.json"),
                "dataset_sha256": dataset.content_sha256,
                "training_locations_sha256": sha256(training / "locations.json"),
                "previous_reservation_sha256": sha256(previous / "reservation.json"),
                "previous_transfer_sha256": sha256(transfer / "plan.json"),
                "pools": pools,
            },
        )


def checked_reservation(output):
    receipt = read(output / "reservation.json")
    dataset = DatasetManifest.for_modeling((output / "dataset.json").read_bytes())
    if (
        receipt["plan_sha256"] != content_hash(protocol())
        or sha256(output / "plan.json") != receipt["plan_sha256"]
        or sha256(output / "prior.json") != receipt["prior_sha256"]
        or dataset.content_sha256 != receipt["dataset_sha256"]
    ):
        raise ValueError("Registered forest training policy, cohorts or exclusions changed")
    return receipt


def fingerprint():
    identity = previous_trial.fingerprint()
    for path in (Path(__file__), Path(balanced_forest_training.__file__)):
        identity["scripts"][path.stem] = sha256(path)
    return identity


def registered_corpus(output, receipt, pool):
    record = receipt["pools"][pool]
    folder = output / pool
    if sha256(folder / "locations.json") != record["locations_sha256"]:
        raise ValueError("Registered snapshot locations changed")
    return Corpus(folder, expected_digest=record["dataset_sha256"])


def batches(corpus, members):
    for chunk in batched(members, 8, strict=False):
        samples = []
        for member in chunk:
            snapshot = corpus.snapshot(member)
            export = snapshot.export
            grid = SnapshotGrid.from_export(export)
            labels = annotate(grid, PROFILE)
            labels.validate_for(grid, export_sha256=member.export_sha256)
            samples.append(
                LabeledSnapshot(
                    source_namespace=member.source_namespace,
                    snapshot_id=member.snapshot_id,
                    export=export,
                    annotations=ValidatedAnnotations(
                        experiment_id=labels.experiment_id,
                        policy=labels.policy,
                        analysis_profile=PROFILE,
                        reference_kind=labels.reference_kind,
                        reviewer=labels.reviewer,
                        intervals=labels.intervals,
                    ),
                )
            )
        yield evaluator(tuple(samples))


def training_arrays(corpora):
    features, labels, durations = [], [], []
    for corpus, members in corpora:
        for batch in batches(corpus, members):
            features.append(matrix(adaptive_forest, batch.inputs))
            for info, reference in zip(batch.inputs.info, batch.references, strict=True):
                y = dense_labels(reference, info.step_count)
                labels.append(y)
                durations.append(balanced_forest_training.interval_weights(y))
    return np.concatenate(features), np.concatenate(labels), np.concatenate(durations)


def train(output, training):
    receipt = checked_reservation(output)
    if sha256(training / "locations.json") != receipt["training_locations_sha256"]:
        raise ValueError("Original training locations changed")
    original = Corpus(training, expected_digest=DATASET)
    retained = tuple(m for m in original.dataset.members if m.partition == "train")
    extension = registered_corpus(output, receipt, "extension")
    identity = fingerprint()
    x, y, w = training_arrays(((original, retained), (extension, extension.dataset.members)))
    with output_directory(output / "fitted") as stage:
        hashes = {}
        for name in CANDIDATES[1:]:
            model = (
                adaptive_forest.fit(x, y)
                if name == "larger"
                else balanced_forest_training.fit(x, y, w)
            )
            path = stage / (name + ".npz")
            model.save(path)
            restored = adaptive_forest.DirectionContextModel.load(path)
            np.testing.assert_allclose(
                model.predict(x), restored.predict(x), rtol=1e-12, atol=1e-12
            )
            hashes[name] = sha256(path)
            print(f"Fitted {name} on {len(y)} original-grid rows", flush=True)
        if fingerprint() != identity or checked_reservation(output) != receipt:
            raise ValueError("Software or registered data changed during fitting")
        write(
            stage / "fit.json",
            {
                "software": identity,
                "reservation_sha256": sha256(output / "reservation.json"),
                "weights": hashes,
                "dataset_sha256": receipt["dataset_sha256"],
                "experiments": {
                    "original": len(retained),
                    "extension": len(extension.dataset.members),
                },
                "classified_by_direction": {
                    label: int((y == idx).sum()) for idx, label in enumerate(DIRECTION_NAMES)
                },
                "rows": len(y),
                "classifiable_rows": int((y >= 0).sum()),
                "labels_sha256": content_hash(y.tolist()),
                "training_weights_sha256": content_hash(w.tolist()),
            },
        )


def check_fit(output):
    checked_reservation(output)
    folder = output / "fitted"
    fit_record = read(folder / "fit.json")
    hashes = {name: sha256(folder / (name + ".npz")) for name in CANDIDATES[1:]}
    if (
        fit_record["software"] != fingerprint()
        or fit_record["weights"] != hashes
        or fit_record["reservation_sha256"] != sha256(output / "reservation.json")
    ):
        raise ValueError("Training evidence no longer matches the registered comparison")
    return fit_record


def load_predictors(output, names, control_bundle=BUNDLE):
    if not names or set(names) - set(CANDIDATES):
        raise ValueError("Unregistered forest training recipe")
    control = None
    if CONTROL in names:
        control = AdaptiveBackend().load_bundle(
            control_bundle, manifest_sha256=BUNDLE_SHA, runtime=RUNTIME
        )
        if control.analysis != PROFILE:
            raise ValueError("Release analysis profile differs from the trial")
    models = {
        name: adaptive_forest.DirectionContextModel.load(output / "fitted" / (name + ".npz"))
        for name in names
        if name != CONTROL
    }

    def predict(inputs):
        results = {CONTROL: control.predict(inputs, runtime=RUNTIME)} if control else {}
        if models:
            x = matrix(adaptive_forest, inputs)
            results.update((name, model.predict(x)) for name, model in models.items())
        return results

    return predict


def measure(output, pool, names, control_bundle):
    receipt = checked_reservation(output)
    corpus = registered_corpus(output, receipt, pool)
    predict = load_predictors(output, names, control_bundle)
    counts = {name: [] for name in names}
    totals = {name: np.zeros(3) for name in names}
    for batch in batches(corpus, corpus.dataset.members):
        y = targets(batch)
        for name, probabilities in predict(batch.inputs).items():
            counts[name].append(experiment_counts(batch, probabilities))
            metrics = probability_metrics(y, probabilities)
            rows = metrics["rows"]
            if rows:
                totals[name] += (rows, rows * metrics["brier"], rows * metrics["log_loss"])
    expected = {experiment_id(m): m.step_count for m in corpus.dataset.members}
    scores, groups = {}, {}
    for name in names:
        ids = [key for keys, _, _ in counts[name] for key in keys]
        lengths = np.concatenate([lengths for _, lengths, _ in counts[name]])
        values = np.concatenate([values for _, _, values in counts[name]])
        if len(ids) != len(expected) or dict(zip(ids, lengths.tolist(), strict=True)) != expected:
            raise ValueError("Scored experiments differ from the reserved population")
        order = np.argsort(ids)
        groups[name] = (sorted(ids), lengths[order], values[order])
        rows, brier, loss = totals[name]
        scores[name] = {
            **summary(values),
            "by_length": {str(n): summary(values[lengths == n]) for n in LENGTHS},
            "probabilities": {
                "rows": int(rows),
                "brier": brier / rows if rows else None,
                "log_loss": loss / rows if rows else None,
            },
        }
    return scores, groups


def choose(scores):
    return max(CANDIDATES, key=lambda name: scores[name]["interval_macro_f1"])


def freeze(output, control_bundle):
    if (output / "freeze.json").exists() or (output / "opened.json").exists():
        raise ValueError("Validation selection has closed")
    fitted = check_fit(output)
    scores, _ = measure(output, "validation", CANDIDATES, control_bundle)
    if fitted != check_fit(output):
        raise ValueError("Training evidence changed during selection")
    winner = choose(scores)
    write_new_file(
        output / "freeze.json",
        json_bytes(
            {"fit_sha256": sha256(output / "fitted/fit.json"), "scores": scores, "winner": winner}
        ),
    )
    print(f"Selected {winner}", flush=True)


def read_freeze(output):
    check_fit(output)
    record = read(output / "freeze.json")
    if record["winner"] != choose(record["scores"]) or record["fit_sha256"] != sha256(
        output / "fitted/fit.json"
    ):
        raise ValueError("Frozen winner differs from the recorded validation evidence")
    return record


def score(output, control_bundle):
    winner = read_freeze(output)["winner"]
    if winner == CONTROL:
        raise ValueError("The release won validation; the holdout remains reserved")
    opening = {"freeze_sha256": sha256(output / "freeze.json")}
    write_new_file(output / "opened.json", json_bytes(opening))
    scores, groups = measure(output, "holdout", (CONTROL, winner), control_bundle)
    paired = contrast(groups[CONTROL], groups[winner])
    checks = qualification(scores[CONTROL], scores[winner], paired)
    read_freeze(output)
    if sha256(output / "freeze.json") != opening["freeze_sha256"]:
        raise ValueError("Frozen selection changed during confirmation")
    write_new_file(
        output / "results.json",
        json_bytes(
            {
                **opening,
                "winner": winner,
                "scores": scores,
                "paired_vs_control": paired,
                "checks": checks,
                "qualified": all(checks.values()),
                "experiment_counts": {
                    name: dict(zip(ids, counts.tolist(), strict=True))
                    for name, (ids, _, counts) in groups.items()
                },
            }
        ),
    )
    print(f"Confirmed {winner}: {checks}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_subparsers(dest="phase", required=True)
    for phase in ("reserve", "train", "freeze", "score"):
        command = phases.add_parser(phase)
        command.add_argument("output", type=Path)
        if phase in {"reserve", "train"}:
            command.add_argument("--training", type=Path, default=TRAINING)
        if phase == "reserve":
            command.add_argument(
                "--previous", type=Path, default=TRAINING.parents[2] / "adaptive-interval-quality"
            )
            command.add_argument(
                "--previous-transfer",
                type=Path,
                default=TRAINING.parents[2] / "adaptive-interval-transfer",
            )
        if phase in {"freeze", "score"}:
            command.add_argument("--control-bundle", type=Path, default=BUNDLE)
    args = parser.parse_args()
    if args.phase == "reserve":
        reserve(args.output, args.training, args.previous, args.previous_transfer)
    elif args.phase == "train":
        train(args.output, args.training)
    elif args.phase == "freeze":
        freeze(args.output, args.control_bundle)
    else:
        score(args.output, args.control_bundle)


if __name__ == "__main__":
    main()
