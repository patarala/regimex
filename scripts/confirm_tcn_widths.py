"""Evaluate the selected TCN-width bundles once on independent synthetic experiments."""

import argparse
import json
from pathlib import Path

import numpy as np
from compare_profile_21 import RUNTIME, sha256, write
from compare_tcn_widths import DATASET_SHA256, PROFILE, WIDTHS, generate

from regimex.admin_contracts import GenerationSpec, RawGenerationPlan, SplitPolicy
from regimex.application.training import SharedSelectionEvaluator, _references, observation_inputs
from regimex.backends.baseline import BaselineBackend
from regimex.backends.tcn import TCNBackend
from regimex.benchmark_contracts import dataset_identity_keys
from regimex.data_catalog import DatasetManifest, _load_partition, content_hash
from regimex.evaluation import paired_bootstrap

POPULATIONS = tuple(
    GenerationSpec(
        dataset_id=f"tcn-confirmation-{length}",
        plan=RawGenerationPlan(seed=1604583900 + length, experiment_count=34, step_count=length),
        splits=SplitPolicy(seed=821177000 + length, train_fraction=0.01, validation_fraction=0.95),
    )
    for length in (256, 512)
)
DRAWS, BOOTSTRAP_SEED = 2000, 1729
CONFIRMATION_SHA256 = "33826a9c877dd4da6215fcb8b6a602ccc1eab18f87d6ca5f0b41c0bc2365db1d"


def load_models(fitted: Path, development: DatasetManifest):
    models = {}
    manifests = {}
    for name in ("baseline", *(f"tcn-{width}-channels" for width in WIDTHS)):
        folder = fitted / name
        digest = json.loads((fitted / f"{name}-validation.json").read_bytes())["bundle_sha256"]
        backend = BaselineBackend() if name == "baseline" else TCNBackend()
        predictor = backend.load_bundle(folder, manifest_sha256=digest, runtime=RUNTIME)
        manifest = json.loads((folder / "manifest.json").read_bytes())
        if (
            predictor.analysis != PROFILE
            or manifest["training"]["dataset_sha256"] != development.content_sha256
        ):
            raise ValueError(f"{name} uses a different profile or training corpus")
        if (
            name != "baseline"
            and name != f"tcn-{predictor.configuration.parameters.channels}-channels"
        ):
            raise ValueError(f"{name} does not contain the requested channel width")
        models[name], manifests[name] = predictor, digest
    return models, manifests


def audit_population(
    development: DatasetManifest, fresh: DatasetManifest, exposure: Path | None
) -> dict:
    old_seeds = {m.split_group.seed for m in development.members}
    keys = {content_hash(k) for k in dataset_identity_keys(development.members)}
    if exposure:
        prior = json.loads(exposure.read_bytes())
        old_seeds.update(prior["excluded_seeds"])
        keys.update(prior["identity_keys_sha256"])
    seeds = {m.split_group.seed for m in fresh.members}
    new_keys = {content_hash(k) for k in dataset_identity_keys(fresh.members)}
    if len(seeds) != len(fresh.members) or seeds & old_seeds or keys & new_keys:
        raise ValueError("The confirmation cohort repeats a generator seed or an exposed identity")
    return {
        "dataset_sha256": fresh.content_sha256,
        "new_seeds": len(seeds),
        "new_keys": len(new_keys),
        "prior_seeds": len(old_seeds),
        "prior_keys": len(keys),
        "external_inventory_sha256": sha256(exposure) if exposure else None,
    }


def confirm(fitted: Path, exposure: Path | None) -> None:
    development = DatasetManifest.for_modeling((fitted / "dataset.json").read_bytes())
    if development.content_sha256 != DATASET_SHA256:
        raise ValueError("The models must come from the original width comparison")
    models, manifests = load_models(fitted, development)
    output = fitted / "confirmation"
    output.mkdir()
    write(
        output / "plan.json",
        {
            "manifests": manifests,
            "training_dataset_sha256": development.content_sha256,
            "populations": [p.model_dump(mode="json") for p in POPULATIONS],
            "partition": "validation",
            "profile": PROFILE.model_dump(mode="json"),
            "runtime": RUNTIME.model_dump(mode="json"),
            "primary_difference": "TCN 48 minus TCN 32",
            "metric": "decoded interval macro-F1 at IoU >= 0.5",
            "bootstrap_draws": DRAWS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "script_sha256": sha256(Path(__file__)),
        },
    )
    corpus = generate(output, POPULATIONS, "tcn-width-confirmation", CONFIRMATION_SHA256)
    write(output / "independence.json", audit_population(development, corpus.dataset, exposure))
    refs, labels = corpus.references("validation", profile=PROFILE)
    samples = _load_partition(
        corpus.dataset,
        refs,
        snapshot_reader=corpus.snapshot,
        reference_reader=lambda _, member: labels[member.source_namespace],
    )
    lengths = [s.export.time_grid.step_count for s in samples]
    if len(samples) != 64 or lengths.count(256) != 32 or lengths.count(512) != 32:
        raise ValueError("Unexpected confirmation cohort size or experiment lengths")
    inputs = observation_inputs(samples)
    evaluator = SharedSelectionEvaluator(inputs, _references(samples), PROFILE, refs.kind)
    scores, counts = {}, {}
    for name, model in models.items():
        report = evaluator.evaluate(model.predict(inputs, runtime=RUNTIME))
        write(output / f"{name}.json", report)
        scores[name] = report["segment_macro_f1"]
        if name in ("tcn-32-channels", "tcn-48-channels"):
            counts[name] = {
                item["experiment_id"]: tuple(
                    np.array(item[field], dtype=np.int64)
                    for field in (
                        "segment_counts",
                        "point_counts",
                        "boundary_counts",
                        "typed_boundary_counts",
                    )
                )
                for item in report["experiment_statistics"]
            }
    difference = paired_bootstrap(
        counts["tcn-32-channels"], counts["tcn-48-channels"], repetitions=DRAWS, seed=BOOTSTRAP_SEED
    )["metrics"]["segment_macro_f1"]
    result = {
        "reference_sha256": refs.content_sha256,
        "experiments": len(samples),
        "instrument_sequences": len(inputs.info),
        "interval_macro_f1": scores,
        "tcn_48_minus_32": difference,
    }
    write(output / "results.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fitted", type=Path, help="Output of compare_tcn_widths.py")
    parser.add_argument(
        "--exposure-index", type=Path, help="Additional prior seed and identity digests."
    )
    args = parser.parse_args()
    confirm(args.fitted, args.exposure_index)
