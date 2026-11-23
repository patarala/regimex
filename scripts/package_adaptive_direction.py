"""Package the confirmed adaptive weights and their exact training and selection populations."""

import argparse
from hashlib import sha256
from json import loads
from pathlib import Path
from runpy import run_path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
from compare_direction_context import PROFILE, RUNTIME, checked_reservation, matrix
from compare_profile_21 import Corpus
from compare_temporal_architectures import evaluator
from fit_pause_targets import DATASET

from regimex.application.analysis_computation import snapshot_inputs
from regimex.data_catalog import (
    ReferencePayload,
    _load_partition,
    analysis_profile_hash,
    content_hash,
    seal_dataset,
    seal_reference_set,
    validate_reference_membership,
)
from regimex.evaluation import dense_labels
from regimex.model_bundles import TrainingEvidence
from regimex.model_contracts import BundleProvenance
from regimex.model_registry import AdaptiveParameters, SingleFitPolicy, TrainingRuntime
from regimex.storage import digest, json_bytes, output_directory, write_new_file
from regimex.terminal_training import register_completed_bundle

PUBLISHED_RESULT = "c198dcb7f354f6cb56a67e096b71187528d6b4260f38c9a88facf8f0439d23db"
TRANSFER_RESULT = "d353c430c806ce4f801e82c386b1f30949f2f92b6c768b211b413ea2be96caf8"
PARTS = ("train", "validation", "test")
MODEL_ID = "adaptive-0.1.0"


def bound_json(path, expected):
    data = path.read_bytes()
    if sha256(data).hexdigest() != expected:
        raise ValueError(f"Unexpected evidence digest: {path}")
    return loads(data)


def frozen_implementation(source, expected_digest):
    # Execute the exact reviewed historical implementation even when the current
    # checkout has moved this model into the service backend.
    code = source.read_bytes()
    if sha256(code).hexdigest() != expected_digest:
        raise ValueError("Historical implementation differs from the fitted source")
    with TemporaryDirectory() as directory:
        module = Path(directory) / "adaptive_direction_model.py"
        module.write_bytes(code)
        return SimpleNamespace(**run_path(str(module)))


def evidence(comparison, transfer):
    published = bound_json(comparison / "published/results.json", PUBLISHED_RESULT)
    external = bound_json(transfer / "results.json", TRANSFER_RESULT)
    ids = published["primary_sha256"]
    primary = bound_json(comparison / "results.json", ids["results"])
    freeze = bound_json(comparison / "freeze.json", ids["freeze"])
    fit = bound_json(comparison / "fitted/fit.json", freeze["fit_sha256"])
    receipt = bound_json(comparison / "reservation.json", fit["reservation_sha256"])
    if (
        receipt != checked_reservation(comparison)
        or primary["winner"] != "adaptive"
        or freeze["winner"] != "adaptive"
        or external["native_freeze_sha256"] != ids["freeze"]
        or not (
            primary["qualified"]
            and published["qualified_against_published_release"]
            and external["passed"]
        )
        or digest(comparison / "fitted/adaptive.npz") != fit["weights"]["adaptive"]
    ):
        raise ValueError(
            "Cannot package weights without their confirmed comparison and transfer results"
        )
    return fit, freeze, receipt


def members_by_partition(corpus, parts, target):
    return tuple(
        member.model_copy(update={"partition": target})
        for part in parts
        for member in corpus.dataset.members
        if member.partition == part
    )


def catalog(training, validation, holdout):
    # Selection and holdout used every snapshot across their original storage partitions.
    return seal_dataset(
        "adaptive-0.1.0-populations",
        members_by_partition(training, ("train",), "train")
        + members_by_partition(validation, PARTS, "validation")
        + members_by_partition(holdout, PARTS, "test"),
    )


def references(corpus, parts, target, dataset):
    members, dense, validation_samples = [], [], []
    for part in parts:
        labels, annotations = corpus.references(part, profile=PROFILE)
        members.extend(labels.members)
        if target == "validation":
            validation_samples.extend(
                _load_partition(
                    corpus.dataset,
                    labels,
                    snapshot_reader=corpus.snapshot,
                    reference_reader=lambda _, m, table=annotations: table[m.source_namespace],
                )
            )
        else:
            for member in corpus.dataset.members:
                if member.partition == part:
                    intervals = annotations[member.source_namespace].intervals
                    for instrument in member.instrument_ids:
                        series = tuple(i for i in intervals if i.instrument_id == instrument)
                        dense.append(dense_labels(series, member.step_count))
    merged = seal_reference_set(
        ReferencePayload(
            reference_set_id=f"adaptive-0.1.0-{target}",
            dataset_sha256=dataset.content_sha256,
            partition=target,
            kind=labels.kind,
            policy=labels.policy,
            semantics=labels.semantics,
            analysis_profile=PROFILE,
            members=tuple(members),
        )
    )
    validate_reference_membership(dataset, merged)
    return merged, dense, validation_samples


def package(comparison, transfer, train_path, source_path, destination):
    from regimex.backends import adaptive, adaptive_forest
    from regimex.backends.adaptive import AdaptiveBackend, fit_result_from_forest
    from regimex.backends.adaptive_forest import DirectionContextModel

    fit, freeze, receipt = evidence(comparison, transfer)
    original = frozen_implementation(
        source_path, fit["software"]["scripts"]["adaptive_direction_model"]
    )
    if digest(train_path / "locations.json") != receipt["training_locations_sha256"]:
        raise ValueError("Historical training snapshot locations changed")
    training = Corpus(train_path, expected_digest=DATASET)
    pools = {}
    for name in ("validation", "holdout"):
        folder = comparison / name
        if digest(folder / "locations.json") != receipt["pools"][name]["locations_sha256"]:
            raise ValueError(f"Historical {name} snapshot locations changed")
        pools[name] = Corpus(folder, expected_digest=receipt["pools"][name]["dataset_sha256"])
    dataset = catalog(training, pools["validation"], pools["holdout"])
    labels, dense, _ = references(training, ("train",), "train", dataset)
    targets = np.concatenate(dense)
    if (
        content_hash(targets.tolist()) != fit["labels_sha256"]
        or int(np.sum(targets >= 0)) != fit["classifiable_rows"]
    ):
        raise ValueError("Reconstructed targets differ from the fitted training rows")
    validation, _, samples = references(pools["validation"], PARTS, "validation", dataset)
    held = evaluator(tuple(samples))
    weights = comparison / "fitted/adaptive.npz"
    historical = original.DirectionContextModel.load(weights)
    expected = historical.predict(matrix(original, held.inputs))
    score = held(expected)
    if score != freeze["scores"]["adaptive"]["interval_macro_f1"]:
        raise ValueError(
            "Published selection score cannot be reproduced on its exact validation membership"
        )
    parameters = AdaptiveParameters(trees=40, max_depth=12)
    policy = SingleFitPolicy(optimizer="extra_trees", seeds=(173,))
    replay = {
        "fit_sha256": freeze["fit_sha256"],
        "weights_sha256": fit["weights"]["adaptive"],
        "original_training_dataset_sha256": DATASET,
        "selection_dataset_sha256": pools["validation"].dataset.content_sha256,
        "holdout_dataset_sha256": pools["holdout"].dataset.content_sha256,
        "published_comparison_sha256": PUBLISHED_RESULT,
        "transfer_comparison_sha256": TRANSFER_RESULT,
        "native_source_sha256": {
            module.__name__: digest(Path(module.__file__)) for module in (adaptive, adaptive_forest)
        },
        "training_runtime": (
            "The original forest used one sklearn worker. Import supplies service defaults "
            "for the time and memory limits that the experiment did not record."
        ),
    }
    environment = {**fit["software"], "experiment": replay}
    version = TrainingEvidence(
        run_id="direction-context-adaptive",
        dataset_sha256=dataset.content_sha256,
        split_manifest_sha256=dataset.split_manifest_sha256,
        labels=labels.identity(),
        validation_references=validation.identity(),
        seeds=policy.seeds,
        selected_seed=173,
        selection_metric="decoded_segment_macro_f1/iou-0.50",
        validation_segment_macro_f1=score,
    )
    source = DirectionContextModel.load(
        weights, trees=parameters.trees, max_depth=parameters.max_depth
    )
    result = fit_result_from_forest(
        source,
        parameters=parameters,
        policy=policy,
        analysis=PROFILE,
        runtime=TrainingRuntime(threads=1),
        validation_score=score,
        environment=environment,
    )
    provenance = BundleProvenance(
        model_version_id=MODEL_ID,
        build_id="adaptive-codec-" + content_hash(replay["native_source_sha256"]),
        analysis_profile="direction-32/2.2",
        analysis_profile_sha256=analysis_profile_hash(PROFILE),
        environment_sha256=content_hash(environment),
        training=version,
    )
    backend = AdaptiveBackend()
    with output_directory(destination) as stage:
        bundle = stage / "bundle"
        bundle.mkdir()
        manifest = backend.write_bundle(result, bundle, provenance=provenance, analysis=PROFILE)
        restored = backend.load_bundle(bundle, manifest_sha256=manifest, runtime=RUNTIME)
        np.testing.assert_array_equal(restored.predict(held.inputs, runtime=RUNTIME), expected)
        export = samples[0].export
        fresh = snapshot_inputs(export)
        np.testing.assert_array_equal(
            restored.predict(fresh, runtime=RUNTIME), historical.predict(matrix(original, fresh))
        )
        for key, item in (
            ("dataset", dataset),
            ("training-labels", labels),
            ("validation-references", validation),
        ):
            write_new_file(stage / f"{key}.json", json_bytes(item.model_dump(mode="json")))
        write_new_file(
            stage / "verification.json",
            json_bytes(
                {
                    **replay,
                    "manifest_sha256": manifest,
                    "selection_score": score,
                    "validation_experiments": len(samples),
                    "validation_rows_compared": held.inputs.row_count,
                    "export_sha256": export.content_sha256,
                    "export_rows_compared": fresh.row_count,
                    "maximum_probability_difference": 0.0,
                }
            ),
        )
    register_completed_bundle(destination / "bundle", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparison", type=Path)
    parser.add_argument("transfer", type=Path)
    parser.add_argument("training", type=Path)
    parser.add_argument(
        "source", type=Path, help="Original adaptive_direction_model.py from the fitted comparison"
    )
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(package(args.comparison, args.transfer, args.training, args.source, args.output))


if __name__ == "__main__":
    main()
