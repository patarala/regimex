"""Compare TCN widths on a common set of training and validation experiments."""

import argparse
from pathlib import Path
from statistics import mean

from compare_profile_21 import RUNTIME, Corpus, sha256, software, write

from regimex.admin_contracts import GenerationSpec, RawGenerationPlan, SplitPolicy
from regimex.application.admin_computation import _generation
from regimex.application.training import TrainingRequest, train_candidate
from regimex.backends.baseline import BaselineBackend
from regimex.backends.tcn import TCNBackend
from regimex.backends.tcn_update_records import UpdateHistoryRecord
from regimex.data_catalog import DatasetManifest, RawTrainingData, load_training_data, seal_dataset
from regimex.directions import AnalysisConfig
from regimex.model_registry import ArchitectureRegistry, FixedUpdatePolicy, TrainingRuntime
from regimex.storage import json_bytes
from regimex.terminal_training import register_completed_bundle

WIDTHS = (16, 32, 48, 64)
PROFILE = AnalysisConfig(version="2.2")
PROFILE_ID = f"direction-32/{PROFILE.version}"
DATASET_SHA256 = "cb5335b8217b9326ddbb4737d6f1f7653e91110d36b23ab334389a0d57f77062"
POLICY = FixedUpdatePolicy(
    seeds=(11, 23, 47), updates=96, validation_updates=(32, 64, 96), batch_size=8
)
POPULATIONS = tuple(
    GenerationSpec(
        dataset_id=f"tcn-width-{length}",
        plan=RawGenerationPlan(seed=2311647000 + length, experiment_count=64, step_count=length),
        splits=SplitPolicy(seed=41183000 + length, train_fraction=0.75, validation_fraction=0.24),
    )
    for length in (256, 512)
)


def generate(
    output: Path,
    populations: tuple[GenerationSpec, ...],
    name: str,
    expected_digest: str | None = None,
) -> Corpus:
    members = []
    locations = {}
    for spec in populations:
        directory = output / spec.dataset_id
        directory.mkdir()
        result = _generation(spec, directory, name)
        manifest = DatasetManifest.model_validate_json(json_bytes(result["manifest"]))
        members.extend(manifest.members)
        for member, snapshot in zip(manifest.members, result["snapshots"], strict=True):
            locations[member.source_namespace] = f"{spec.dataset_id}/{snapshot['path']}"
        print(f"Generated {spec.dataset_id}", flush=True)
    dataset = seal_dataset(name, tuple(members))
    write(output / "dataset.json", dataset.model_dump(mode="json"))
    write(output / "locations.json", locations)
    write(output / "reservation.json", {"dataset_sha256": dataset.content_sha256})
    return Corpus(output, expected_digest=expected_digest or dataset.content_sha256)


def best_checkpoints(model_dir: Path) -> list[dict[str, int | float]]:
    history = UpdateHistoryRecord.model_validate_json((model_dir / "history.json").read_bytes())
    rows = []
    for seed in POLICY.seeds:
        scores = [
            (entry.update, entry.validation_segment_macro_f1)
            for entry in history.updates
            if entry.seed == seed and entry.validation_segment_macro_f1 is not None
        ]
        if tuple(step for step, _ in scores) != POLICY.validation_updates:
            raise ValueError(f"Incomplete validation history for seed {seed}")
        step, score = max(scores, key=lambda item: item[1])
        rows.append({"seed": seed, "update": step, "interval_macro_f1": score})
    return rows


def training_data(corpus: Corpus) -> RawTrainingData:
    train, training_labels = corpus.references("train", profile=PROFILE)
    validation, validation_labels = corpus.references("validation", profile=PROFILE)
    annotations = training_labels | validation_labels
    return load_training_data(
        corpus.dataset,
        train,
        validation,
        snapshot_reader=corpus.snapshot,
        reference_reader=lambda _, member: annotations[member.source_namespace],
    )


def baseline_reference(
    data: RawTrainingData, output: Path, registry: ArchitectureRegistry, build: str
) -> None:
    request = TrainingRequest(
        run_id="baseline-reference",
        model_version_id="baseline-reference",
        architecture_id="baseline",
        parameters={},
        policy=registry.get("baseline").default_training_policy,
        training_runtime=TrainingRuntime(threads=2),
        verification_runtime=RUNTIME,
        analysis_profile_id=PROFILE_ID,
        build_id=build,
    )
    model_dir = output / "baseline"
    result = train_candidate(
        data, request, model_dir, registry=registry, worker_id="tcn-width-comparison"
    )
    register_completed_bundle(model_dir, result.manifest_sha256)
    model = (
        BaselineBackend()
        .load_bundle(model_dir, manifest_sha256=result.manifest_sha256, runtime=RUNTIME)
        .model
    )
    row = {
        "parameters": model.coefficients.size + model.intercept.size,
        "interval_macro_f1": result.validation_segment_macro_f1,
        "bundle_sha256": result.manifest_sha256,
    }
    write(output / "baseline-validation.json", row)
    print("Logistic baseline", row, flush=True)


def compare(output: Path) -> None:
    output.mkdir(parents=True)
    environment = software()
    registry = ArchitectureRegistry.discover()
    write(
        output / "plan.json",
        {
            "widths": list(WIDTHS),
            "comparison_script_sha256": sha256(Path(__file__)),
            "dataset_sha256": DATASET_SHA256,
            "populations": [p.model_dump(mode="json") for p in POPULATIONS],
            "policy": POLICY.model_dump(mode="json"),
            "profile": PROFILE.model_dump(mode="json"),
            "checkpoint_rule": "Highest validation interval macro-F1 per seed, earliest on a tie.",
            "width_comparison": "Arithmetic mean over all seed-specific checkpoint scores.",
            "environment": environment,
        },
    )
    data = training_data(generate(output, POPULATIONS, "tcn-width-comparison", DATASET_SHA256))
    build = "source-" + environment["source_sha256"]
    baseline_reference(data, output, registry, build)
    results = []
    for width in WIDTHS:
        identifier = f"tcn-{width}-channels"
        req = TrainingRequest(
            run_id=identifier,
            model_version_id=identifier,
            architecture_id="tcn",
            parameters={"channels": width},
            policy=POLICY,
            training_runtime=TrainingRuntime(threads=2),
            verification_runtime=RUNTIME,
            analysis_profile_id=PROFILE_ID,
            build_id=build,
        )
        print(f"Fitting {identifier}", flush=True)
        fitted = train_candidate(
            data,
            req,
            output / identifier,
            registry=registry,
            worker_id="tcn-width-comparison",
        )
        register_completed_bundle(output / identifier, fitted.manifest_sha256)
        predictor = TCNBackend().load_bundle(
            output / identifier, manifest_sha256=fitted.manifest_sha256, runtime=RUNTIME
        )
        count = sum(p.numel() for p in predictor.model.parameters())
        per_seed = best_checkpoints(output / identifier)
        row = {
            "channels": width,
            "parameters": count,
            "per_seed": per_seed,
            "mean_interval_macro_f1": mean(r["interval_macro_f1"] for r in per_seed),
            "bundle_sha256": fitted.manifest_sha256,
        }
        write(output / f"{identifier}-validation.json", row)
        results.append(row)
        print(row, flush=True)
    write(output / "validation.json", results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Unused directory for data, bundles and scores.")
    compare(parser.parse_args().output)
