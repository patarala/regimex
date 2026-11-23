"""Freeze the best newly fitted forest on validation, then compare it with the release."""

import argparse
from collections import Counter
from itertools import batched
from pathlib import Path

import numpy as np
from compare_adaptive_intervals import BUNDLE, BUNDLE_SHA, CONTROL, MARGINS, qualification
from compare_direction_context import probability_metrics, targets
from compare_profile_21 import RUNTIME, Corpus, software
from compare_selection_panels import experiment_id
from compare_tcn_widths import PROFILE
from compare_temporal_architectures import (
    DRAW_SEED,
    DRAWS,
    LENGTHS,
    evaluator,
    experiment_counts,
    read,
    summary,
)
from confirm_training_populations import contrast
from direction_candidate_bundles import package_candidates
from train_large_direction import CANDIDATES, SEED

from regimex.application.analysis_computation import snapshot_inputs
from regimex.backends.adaptive import AdaptiveBackend, fit_result_from_forest
from regimex.backends.adaptive_forest import DirectionContextModel
from regimex.data_catalog import (
    DatasetManifest,
    LabeledSnapshot,
    ReferenceSet,
    SnapshotGrid,
    ValidatedAnnotations,
    analysis_profile_hash,
    content_hash,
    create_reference_member,
    create_snapshot_member,
    validate_reference_membership,
)
from regimex.directions import DirectionAnnotations
from regimex.model_bundles import TrainingEvidence
from regimex.model_contracts import BundleProvenance
from regimex.model_registry import AdaptiveParameters, SingleFitPolicy, TrainingRuntime
from regimex.segmentation import annotate
from regimex.storage import digest, json_bytes, output_directory, write_json, write_new_file
from regimex.terminal_training import register_frozen_selection

DATA = Path("var/verification/direction-large-v1")
OUTPUT = Path("var/verification/direction-large-evaluation-v1")
CHALLENGERS = ("greedy-large", "multiscale-large", "multiscale-balanced")


def protocol():
    return {
        "control": {"name": CONTROL, "bundle_sha256": BUNDLE_SHA},
        "challengers": list(CHALLENGERS),
        "profile": PROFILE.model_dump(mode="json"),
        "inference_runtime": RUNTIME.model_dump(mode="json"),
        "selection": "Rank by pooled validation interval macro-F1 at IoU >= 0.50. "
        "Ties follow registered order, starting with the release.",
        "confirmation": "Compare the best new model with the control once, even if the control "
        "won validation. Only a validation winner can qualify as a replacement.",
        "native_gate": "Require a positive lower confidence limit for the interval macro-F1 "
        "difference and satisfy every regression margin.",
        "maximum_loss": MARGINS,
        "transfer_gate": "Also require the independent controlled gap-transfer check.",
        "bootstrap": {"draws": DRAWS, "seed": DRAW_SEED, "unit": "whole experiment within length"},
        "reference_kind": "automatic",
    }


def evidence_files(data):
    paths = [
        "candidate-plan.json",
        "prior.json",
        "population.json",
        "reservation.json",
        "dataset.json",
    ]
    for part in ("train", "validation", "test"):
        paths.extend(f"{part}/{name}.json" for name in ("reservation", "locations", "dataset"))
        if part != "test":
            paths.append(f"{part}/references.json")
    for name in CHALLENGERS:
        paths.extend(f"fitted/{name}.{suffix}" for suffix in ("json", "npz"))
    return {key: digest(data / key) for key in paths}


def development(data):
    dataset = DatasetManifest.model_validate_json((data / "dataset.json").read_bytes())
    receipt = read(data / "reservation.json")
    if (
        receipt["dataset_sha256"] != dataset.content_sha256
        or receipt["population_sha256"] != digest(data / "population.json")
        or receipt["prior_sha256"] != digest(data / "prior.json")
        or read(data / "population.json")["candidate_plan_sha256"]
        != digest(data / "candidate-plan.json")
        or any(
            receipt["locations_sha256"][part] != digest(data / part / "locations.json")
            for part in ("train", "validation", "test")
        )
    ):
        raise ValueError("Dataset or split locations differ from the population reservation")
    refs = {}
    for name in ("train", "validation"):
        refs[name] = ReferenceSet.model_validate_json(
            (data / name / "references.json").read_bytes()
        )
        validate_reference_membership(dataset, refs[name])
        if (
            refs[name].partition != name
            or refs[name].analysis_profile != PROFILE
            or refs[name].content_sha256 != receipt["references_sha256"][name]
        ):
            raise ValueError(f"Unexpected reference set for {name}")
    return dataset, refs


class PartitionCorpus(Corpus):
    def __init__(self, data, partition, dataset):
        super().__init__(data / partition, expected_digest=dataset.content_sha256)
        self.members = tuple(m for m in self.dataset.members if m.partition == partition)
        if self.locations.keys() != {m.source_namespace for m in self.members}:
            raise ValueError(f"{partition} snapshot locations differ from the registered split")


def samples(corpus, references):
    by_key = {r.source_namespace: r for r in references.members} if references else {}
    for member in corpus.members:
        snapshot = corpus.snapshot(member)
        if create_snapshot_member(snapshot, member.partition) != member:
            raise ValueError("Snapshot content differs from registered input")
        grid = SnapshotGrid.from_export(snapshot.export)
        if references is None:
            labels = annotate(grid, PROFILE)
        else:
            path = corpus.output / "annotations" / f"{member.split_group.seed}.json"
            labels = DirectionAnnotations.model_validate_json(path.read_bytes())
            if create_reference_member(member, labels) != by_key[member.source_namespace]:
                raise ValueError("Validation annotations differ from the registered references")
        if (
            labels.policy_config != PROFILE.model_dump(mode="json")
            or labels.reference_kind != "automatic"
        ):
            raise ValueError("Unexpected direction reference kind or analysis profile")
        labels.validate_for(grid, export_sha256=member.export_sha256)
        yield LabeledSnapshot(
            source_namespace=member.source_namespace,
            snapshot_id=member.snapshot_id,
            export=snapshot.export,
            annotations=ValidatedAnnotations(
                experiment_id=labels.experiment_id,
                policy=labels.policy,
                analysis_profile=PROFILE,
                reference_kind=labels.reference_kind,
                reviewer=labels.reviewer,
                intervals=labels.intervals,
            ),
        )


def measure(corpus, references, models):
    def predict(inputs):
        return {name: model.predict(inputs, runtime=RUNTIME) for name, model in models.items()}

    return measure_predictions(corpus, references, tuple(models), predict)


def measure_predictions(corpus, references, names, predict):
    counts = {name: {} for name in names}
    losses = {name: np.zeros(3) for name in names}
    for index, chunk in enumerate(batched(samples(corpus, references), 8, strict=False), start=1):
        batch = evaluator(chunk)
        labels = targets(batch)
        for name, probabilities in predict(batch.inputs).items():
            ids, _, values = experiment_counts(batch, probabilities)
            if counts[name].keys() & ids:
                raise ValueError("An experiment was scored more than once")
            counts[name].update(zip(ids, values, strict=True))
            p = probability_metrics(labels, probabilities)
            if p["rows"]:
                losses[name] += p["rows"] * np.array([1, p["brier"], p["log_loss"]])
        if index % 8 == 0:
            print(f"Scored {index * 8}/{len(corpus.members)} experiments", flush=True)
    membership = {experiment_id(m): m.step_count for m in corpus.members}
    ids = sorted(membership)
    lengths = np.array([membership[key] for key in ids])
    scores, groups = {}, {}
    for name, table in counts.items():
        if table.keys() != membership.keys():
            raise ValueError("Scored experiments differ from the registered population")
        values = np.stack([table[key] for key in ids])
        groups[name] = (ids, lengths, values)
        rows, brier, log_loss = losses[name]
        scores[name] = {
            **summary(values),
            "by_length": {str(n): summary(values[lengths == n]) for n in LENGTHS},
            "probabilities": {
                "rows": int(rows),
                "brier": brier / rows if rows else None,
                "log_loss": log_loss / rows if rows else None,
            },
        }
    return scores, groups


def load_fit(data, name, dataset, labels):
    receipt = read(data / "fitted" / f"{name}.json")
    weights = data / "fitted" / f"{name}.npz"
    if receipt["weights_sha256"] != digest(weights) or receipt["training_receipt_sha256"] != digest(
        data / "train" / "reservation.json"
    ):
        raise ValueError(f"Changed training receipt or weights for {name}")
    params = AdaptiveParameters.model_validate(receipt["parameters"])
    if (
        params != CANDIDATES[name]
        or receipt["seed"] != SEED
        or receipt["population_receipt_sha256"] != digest(data / "reservation.json")
        or receipt["dataset_sha256"] != dataset.content_sha256
        or receipt["references_sha256"] != labels.content_sha256
    ):
        raise ValueError(f"Unexpected training inputs or recipe for {name}")
    forest = DirectionContextModel.load(weights, trees=params.trees, max_depth=params.max_depth)
    return {
        "model": forest,
        "parameters": params,
        "policy": SingleFitPolicy(optimizer="extra_trees", seeds=(receipt["seed"],)),
        "runtime": TrainingRuntime(threads=receipt["threads"]),
        "analysis": PROFILE,
        "environment": receipt,
    }


def choose(scores, names):
    return max(names, key=lambda name: scores[name]["interval_macro_f1"])


def package(
    stage,
    name,
    spec,
    score,
    dataset,
    refs,
    corpus,
    *,
    model_version_id=None,
    additional_training_sources=(),
    build_id=None,
):
    policy = spec["policy"]
    fit = fit_result_from_forest(**spec, validation_score=score)
    provenance = BundleProvenance(
        model_version_id=model_version_id or f"direction-{name}-v1",
        build_id=build_id or "direction-large-" + digest(Path(__file__)),
        analysis_profile="direction-32/2.2",
        analysis_profile_sha256=analysis_profile_hash(PROFILE),
        environment_sha256=content_hash(spec["environment"]),
        training=TrainingEvidence(
            run_id=model_version_id or f"direction-{name}-v1",
            dataset_sha256=dataset.content_sha256,
            split_manifest_sha256=dataset.split_manifest_sha256,
            labels=refs["train"].identity(),
            validation_references=refs["validation"].identity(),
            seeds=policy.seeds,
            selected_seed=policy.seeds[0],
            selection_metric="decoded_segment_macro_f1/iou-0.50",
            validation_segment_macro_f1=score,
            additional_training_sources=additional_training_sources,
        ),
    )
    backend = AdaptiveBackend()
    folder = stage / "bundle"
    sha = backend.write_bundle(fit, folder, analysis=PROFILE, provenance=provenance)
    restored = backend.load_bundle(folder, manifest_sha256=sha, runtime=RUNTIME)
    verified = {}
    for member in corpus.members:
        stratum = f"{member.step_count}x{len(member.instrument_ids)}"
        if stratum not in verified:
            inputs = snapshot_inputs(corpus.snapshot(member).export)
            expected = fit.predictor.predict(inputs, runtime=RUNTIME)
            np.testing.assert_array_equal(restored.predict(inputs, runtime=RUNTIME), expected)
            verified[stratum] = {"export_sha256": member.export_sha256, "rows": inputs.row_count}
    return sha, verified


def implementation():
    scripts = Path(__file__).parent
    helpers = (
        "compare_adaptive_intervals.py",
        "compare_direction_context.py",
        "compare_profile_21.py",
        "compare_tcn_slopes.py",
        "compare_temporal_architectures.py",
        "confirm_training_populations.py",
        Path(__file__).name,
    )
    return {**software(), "scoring_helpers": {name: digest(scripts / name) for name in helpers}}


def freeze(data, output, control):
    if (output / "opened.json").exists() or (output / "selection").exists():
        raise ValueError("Validation selection is already closed")
    if read(output / "protocol.json") != protocol():
        raise ValueError("Selection protocol differs from registration")
    inputs = evidence_files(data)
    source = implementation()
    dataset, refs = development(data)
    validation = PartitionCorpus(data, "validation", dataset)
    fits = {name: load_fit(data, name, dataset, refs["train"]) for name in CHALLENGERS}
    models = {
        CONTROL: AdaptiveBackend().load_bundle(control, manifest_sha256=BUNDLE_SHA, runtime=RUNTIME)
    }
    for name, spec in fits.items():
        models[name] = fit_result_from_forest(**spec, validation_score=0).predictor
    scores, _ = measure(validation, refs["validation"], models)
    winner = choose(scores, (CONTROL, *CHALLENGERS))
    challenger = choose(scores, CHALLENGERS)
    with output_directory(output / "selection") as stage:
        sha, verified = package(
            stage,
            challenger,
            fits[challenger],
            scores[challenger]["interval_macro_f1"],
            dataset,
            refs,
            validation,
        )
        if inputs != evidence_files(data) or source != implementation():
            raise ValueError("Comparison inputs or implementation changed during selection")
        write_json(
            stage / "freeze.json",
            {
                "protocol_sha256": digest(output / "protocol.json"),
                "inputs": inputs,
                "implementation": source,
                "validation": scores,
                "validation_winner": winner,
                "challenger": challenger,
                "bundle_sha256": sha,
                "control_bundle_sha256": BUNDLE_SHA,
                "round_trip_on_validation": verified,
            },
        )
    package_candidates(data, output, mixed=False)
    print(f"Validation winner: {winner}; packaged {challenger} ({sha})", flush=True)


def score(data, output, control):
    selection = output / "selection"
    freeze_path = selection / "freeze.json"
    frozen = read(freeze_path)
    if (
        read(output / "protocol.json") != protocol()
        or frozen["protocol_sha256"] != digest(output / "protocol.json")
        or frozen["control_bundle_sha256"] != BUNDLE_SHA
        or frozen["inputs"] != evidence_files(data)
        or frozen["implementation"] != implementation()
        or frozen["validation_winner"] != choose(frozen["validation"], (CONTROL, *CHALLENGERS))
        or frozen["challenger"] != choose(frozen["validation"], CHALLENGERS)
    ):
        raise ValueError("Frozen selection or comparison inputs changed")
    score_frozen(
        data,
        output,
        control,
        frozen,
        validation_qualified=frozen["validation_winner"] == frozen["challenger"],
        evidence=evidence_files,
    )


def score_frozen(data, output, control, frozen, *, validation_qualified, evidence):
    selection = output / "selection"
    freeze_path = selection / "freeze.json"
    name = frozen["challenger"]
    models = {
        CONTROL: AdaptiveBackend().load_bundle(
            control, manifest_sha256=BUNDLE_SHA, runtime=RUNTIME
        ),
        name: AdaptiveBackend().load_bundle(
            selection / "bundle", manifest_sha256=frozen["bundle_sha256"], runtime=RUNTIME
        ),
    }
    opening = {"freeze_sha256": digest(freeze_path)}
    opened = output / "opened.json"
    if opened.exists():
        if read(opened) != opening or (output / "results.json").exists():
            raise ValueError("Final evaluation is complete or belongs to another freeze")
    else:
        write_new_file(opened, json_bytes(opening))
    dataset, _ = development(data)
    corpus = PartitionCorpus(data, "test", dataset)
    scores, groups = measure(corpus, None, models)
    paired = contrast(groups[CONTROL], groups[name])
    checks = qualification(scores[CONTROL], scores[name], paired)
    checks["validation_winner"] = validation_qualified
    if digest(freeze_path) != opening["freeze_sha256"] or frozen["inputs"] != evidence(data):
        raise ValueError("Inputs changed during final scoring")
    result = {
        **opening,
        "control": CONTROL,
        "challenger": name,
        "scores": scores,
        "paired_vs_control": paired,
        "checks": checks,
        "native_qualified": all(checks.values()),
        "experiments": len(corpus.members),
        "lengths": dict(Counter(m.step_count for m in corpus.members)),
        "experiment_counts": {
            key: dict(zip(ids, counts.tolist(), strict=True))
            for key, (ids, _, counts) in groups.items()
        },
    }
    write_new_file(output / "results.json", json_bytes(result))
    register_frozen_selection(output, native_final=output / "results.json")
    print(f"{name}: {checks}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("register", "freeze", "score", "package-candidates"))
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--control", type=Path, default=BUNDLE)
    args = parser.parse_args()
    if args.command == "package-candidates":
        package_candidates(args.data, args.output, mixed=False, explicit=True)
    elif args.command == "register":
        write_new_file(args.output / "protocol.json", json_bytes(protocol()))
    elif args.command == "freeze":
        freeze(args.data, args.output, args.control)
    else:
        score(args.data, args.output, args.control)


if __name__ == "__main__":
    main()
