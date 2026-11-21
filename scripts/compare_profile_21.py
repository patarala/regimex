"""Reproduce the synthetic direction-32/2.1 baseline and TCN comparison."""

import argparse
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np
from pydantic import JsonValue

from regimex.admin_contracts import GenerationSpec, RawGenerationPlan, SplitPolicy
from regimex.application.admin_computation import _generation
from regimex.application.training import (
    SharedSelectionEvaluator,
    TrainingRequest,
    observation_inputs,
    train_candidate,
)
from regimex.benchmark_contracts import dataset_identity_keys
from regimex.benchmarks import DirectionSemantics
from regimex.data_catalog import (
    DatasetManifest,
    Partition,
    ReferencePayload,
    ReferenceSet,
    SnapshotGrid,
    SnapshotMember,
    SourceSnapshot,
    _load_partition,
    analysis_profile_hash,
    content_hash,
    create_reference_member,
    load_training_data,
    seal_dataset,
    seal_reference_set,
)
from regimex.directions import DIRECTION_POLICY, AnalysisConfig, DirectionAnnotations
from regimex.exports import load_export_v2
from regimex.model_registry import (
    ArchitectureRegistry,
    FixedUpdatePolicy,
    InferenceRuntime,
    TrainingPolicy,
    TrainingPolicySpec,
    TrainingRuntime,
)
from regimex.segmentation import DECODER_VERSION, annotate
from regimex.storage import json_bytes

LENGTHS = (256, 512, 1024)
ARCHITECTURES = ("baseline", "tcn")
PROFILE = AnalysisConfig(version="2.1")
RUNTIME = InferenceRuntime(threads=2, max_batch_size=8)
DATASET_SHA256 = "682b79aa7915a2d864ec35b3f781c254f643570b38ba8e6b1e9630de3a29ec5e"
SOURCE = Path(__file__).resolve().parents[1] / "src" / "regimex"


def write(path: Path, value: object) -> None:
    with path.open("xb") as stream:
        stream.write(json_bytes(value))


def record(path: Path, value: object) -> None:
    """Resume scoring only when the already written evidence agrees."""
    if path.exists():
        if path.read_bytes() != json_bytes(value):
            raise ValueError(f"Previous evidence differs: {path}")
    else:
        write(path, value)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(output, history):
    output.mkdir(parents=True)
    datasets = []
    locations = {}
    for length in LENGTHS:
        folder = output / str(length)
        folder.mkdir()
        spec = GenerationSpec(
            dataset_id=f"profile21-{length}",
            plan=RawGenerationPlan(seed=217031100 + length, experiment_count=64, step_count=length),
            splits=SplitPolicy(
                seed=517031100 + length, train_fraction=0.5, validation_fraction=0.125
            ),
        )
        result = _generation(spec, folder, "profile21-comparison")
        dataset = DatasetManifest.model_validate_json(json_bytes(result["manifest"]))
        datasets.extend(dataset.members)
        for member, snapshot in zip(dataset.members, result["snapshots"], strict=True):
            locations[member.source_namespace] = str(Path(str(length)) / snapshot["path"])
        print(f"Generated {length}-step experiments", flush=True)
    dataset = seal_dataset("profile21-comparison", tuple(datasets))
    if dataset.content_sha256 != DATASET_SHA256:
        raise ValueError("Generated population differs from the recorded comparison")
    receipt = {"dataset_sha256": dataset.content_sha256}
    if history:
        index = json.loads(history.read_bytes())
        seeds = {member.split_group.seed for member in dataset.members}
        keys = {content_hash(key) for key in dataset_identity_keys(dataset.members)}
        if seeds.intersection(index["excluded_seeds"]) or keys.intersection(
            index["identity_keys_sha256"]
        ):
            raise ValueError("The comparison overlaps the supplied history")
        receipt.update(
            history_sha256=sha256(history),
            checked_seeds=len(index["excluded_seeds"]),
            checked_identity_keys=len(index["identity_keys_sha256"]),
        )
    write(output / "locations.json", locations)
    write(output / "dataset.json", dataset.model_dump(mode="json"))
    write(output / "reservation.json", receipt)


def request(architecture: str, build: str) -> TrainingRequest:
    parameters: dict[str, JsonValue]
    policy: TrainingPolicySpec
    if architecture == "baseline":
        parameters = {"regularization": [0.1, 1.0, 10.0], "context_sizes": [32, 128]}
        policy = TrainingPolicy(optimizer="lbfgs", seeds=(11,))
    else:
        parameters = {"channels": 32, "interval_weight_power": 0.5}
        policy = FixedUpdatePolicy(
            seeds=(29,),
            batch_size=8,
            updates=240,
            validation_updates=(80, 160, 240),
            learning_rate=0.0007,
            weight_decay=0.001,
        )
    return TrainingRequest(
        run_id=f"profile21-{architecture}",
        model_version_id=f"profile21-{architecture}",
        architecture_id=architecture,
        parameters=parameters,
        policy=policy,
        training_runtime=TrainingRuntime(threads=2),
        verification_runtime=RUNTIME,
        analysis_profile_id="direction-32/2.1",
        build_id=build,
    )


def software():
    return {
        "python": platform.python_version(),
        **{name: version(name) for name in ("numpy", "scipy", "scikit-learn", "torch", "pydantic")},
        "source_sha256": content_hash(
            {str(p.relative_to(SOURCE)): sha256(p) for p in sorted(SOURCE.rglob("*.py"))}
        ),
        "script_sha256": sha256(Path(__file__)),
        "lockfile_sha256": sha256(SOURCE.parents[1] / "uv.lock"),
    }


class Corpus:
    def __init__(self, output: Path, *, expected_digest: str = DATASET_SHA256) -> None:
        self.output = output
        self.dataset = DatasetManifest.model_validate_json((output / "dataset.json").read_bytes())
        self.locations = json.loads((output / "locations.json").read_bytes())
        reservation = json.loads((output / "reservation.json").read_bytes())
        if (
            reservation["dataset_sha256"] != self.dataset.content_sha256
            or self.dataset.content_sha256 != expected_digest
        ):
            raise ValueError("Dataset differs from the reservation")

    def snapshot(self, member: SnapshotMember) -> SourceSnapshot:
        return SourceSnapshot(
            source_namespace=member.source_namespace,
            split_group=member.split_group,
            export=load_export_v2(self.output / self.locations[member.source_namespace]),
        )

    def references(
        self, partition: Partition, *, profile: AnalysisConfig = PROFILE
    ) -> tuple[ReferenceSet, dict[str, DirectionAnnotations]]:
        annotations: dict[str, DirectionAnnotations] = {}
        members = []
        for member in self.dataset.members:
            if member.partition == partition:
                export = self.snapshot(member).export
                annotation = annotate(SnapshotGrid.from_export(export), profile)
                annotations[member.source_namespace] = annotation
                members.append(create_reference_member(member, annotation))
        references = seal_reference_set(
            ReferencePayload(
                reference_set_id=f"profile{profile.version.replace('.', '')}-{partition}",
                dataset_sha256=self.dataset.content_sha256,
                partition=partition,
                kind="automatic",
                policy=DIRECTION_POLICY,
                semantics=DirectionSemantics(
                    profile_sha256=analysis_profile_hash(profile), decoder_version=DECODER_VERSION
                ),
                analysis_profile=profile,
                members=tuple(members),
            )
        )
        return references, annotations


def fit(output):
    corpus = Corpus(output)
    runtime = software()
    build = "source-" + runtime["source_sha256"]
    requests = {name: request(name, build) for name in ARCHITECTURES}
    write(
        output / "training-plan.json",
        {
            "environment": runtime,
            "dataset_sha256": corpus.dataset.content_sha256,
            "requests": {name: req.model_dump(mode="json") for name, req in requests.items()},
            "analysis_profile": PROFILE.model_dump(mode="json"),
        },
    )
    train, labels = corpus.references("train")
    validation, observations = corpus.references("validation")
    labels.update(observations)
    data = load_training_data(
        corpus.dataset,
        train,
        validation,
        snapshot_reader=corpus.snapshot,
        reference_reader=lambda _, m: labels[m.source_namespace],
    )
    manifests = {}
    for name, req in requests.items():
        print(f"Fitting {name}", flush=True)
        result = train_candidate(
            data,
            req,
            output / name,
            registry=ArchitectureRegistry.discover(),
            worker_id="offline-comparison",
        )
        manifests[name] = result.manifest_sha256
        write(output / f"{name}-validation.json", result.model_dump(mode="json"))
        print(
            f"{name} validation interval macro-F1 {result.validation_segment_macro_f1:.4f}",
            flush=True,
        )
    write(
        output / "freeze.json",
        {
            "training_plan_sha256": sha256(output / "training-plan.json"),
            "bundles": manifests,
            "reservation_sha256": sha256(output / "reservation.json"),
        },
    )


def score(output):
    from regimex.application.training import _references
    from regimex.backends.baseline import BaselineBackend
    from regimex.backends.tcn import TCNBackend
    from regimex.evaluation import paired_bootstrap

    corpus = Corpus(output)
    freeze = json.loads((output / "freeze.json").read_bytes())
    plan = json.loads((output / "training-plan.json").read_bytes())
    if (
        freeze["training_plan_sha256"] != sha256(output / "training-plan.json")
        or plan["environment"] != software()
        or freeze["reservation_sha256"] != sha256(output / "reservation.json")
        or plan["dataset_sha256"] != corpus.dataset.content_sha256
    ):
        raise ValueError("Training or generation identity has changed since the freeze")
    models = {
        name: backend.load_bundle(
            output / name, manifest_sha256=freeze["bundles"][name], runtime=RUNTIME
        )
        for name, backend in (("baseline", BaselineBackend()), ("tcn", TCNBackend()))
    }
    record(output / "opened.json", {"freeze_sha256": sha256(output / "freeze.json")})
    references, annotations = corpus.references("test")
    samples = _load_partition(
        corpus.dataset,
        references,
        snapshot_reader=corpus.snapshot,
        reference_reader=lambda _, m: annotations[m.source_namespace],
    )
    result = {}
    for length in (*LENGTHS, None):
        group = tuple(
            s for s in samples if length is None or s.export.time_grid.step_count == length
        )
        inputs = observation_inputs(group)
        evaluator = SharedSelectionEvaluator(inputs, _references(group), PROFILE, "automatic")
        metrics = {}
        counts = {}
        for name, model in models.items():
            probabilities = model.predict(inputs, runtime=RUNTIME)
            report = evaluator.evaluate(probabilities, bootstrap_repetitions=1000)
            record(output / f"{name}-{length or 'pooled'}.json", report)
            metrics[name] = {
                k: report[k]
                for k in (
                    "segment_macro_f1",
                    "point_macro_f1",
                    "coverage",
                    "instrument_sequence_count",
                    "experiment_count",
                    "observed_step_count",
                )
            }
            counts[name] = {
                entry["experiment_id"]: tuple(
                    np.asarray(entry[field], dtype=np.int64)
                    for field in (
                        "segment_counts",
                        "point_counts",
                        "boundary_counts",
                        "typed_boundary_counts",
                    )
                )
                for entry in report["experiment_statistics"]
            }
        paired = paired_bootstrap(counts["baseline"], counts["tcn"], repetitions=1000)
        result[str(length or "pooled")] = {**metrics, "paired": paired}
        print(f"Scored {length or 'pooled'}", flush=True)
    record(
        output / "results.json",
        {
            "test_references_sha256": references.content_sha256,
            "dataset_sha256": corpus.dataset.content_sha256,
            "results": result,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "fit", "score"))
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--exposure-index",
        type=Path,
        help="Check known previous seeds and observation fingerprints during preparation.",
    )
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args.output, args.exposure_index)
    elif args.exposure_index:
        parser.error("--exposure-index is only valid during preparation")
    elif args.phase == "fit":
        fit(args.output)
    else:
        score(args.output)


if __name__ == "__main__":
    main()
