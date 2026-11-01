"""Build an artificial saved baseline for deployment smoke, using production codecs.

This helper is mounted read-only into the CPU image; it is never packaged in the
runtime wheel or copied into the Docker build context. Automatic controlled
fixtures do not establish independent accuracy, market realism or transfer.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from regimex.application.training import TrainingRequest, train_candidate
from regimex.benchmarks import DirectionSemantics
from regimex.data_catalog import (
    Partition,
    ReferenceMember,
    ReferencePayload,
    ReferenceSet,
    SnapshotMember,
    SourceSnapshot,
    SplitGroup,
    analysis_profile_hash,
    create_reference_member,
    create_snapshot_member,
    load_training_data,
    seal_dataset,
    seal_reference_set,
)
from regimex.directions import AnalysisConfig, DirectionAnnotations
from regimex.exports import (
    ExportInstrument,
    ExportObservation,
    ExportPayloadV2,
    ExportV2,
    seal_export,
)
from regimex.model_registry import (
    ArchitectureRegistry,
    InferenceRuntime,
    TrainingPolicy,
    TrainingRuntime,
)
from regimex.segmentation import DECODER_VERSION, annotate
from regimex.storage import json_bytes


@dataclass(frozen=True)
class Grid:
    experiment_id: str
    export: ExportV2

    @property
    def step_count(self) -> int:
        return self.export.time_grid.step_count

    @property
    def instruments(self) -> tuple[ExportInstrument, ...]:
        return self.export.instruments

    @property
    def observations(self) -> tuple[ExportObservation, ...]:
        return self.export.observations


def source(template: Path, index: int) -> SourceSnapshot:
    value = json.loads(template.read_bytes())
    value.pop("content_sha256")
    identifier = f"deployment-behavior-{index}"
    for name in ("snapshot_id", "run_id", "configuration_id"):
        value["source"][name] = identifier
    value["producer"]["name"] = "controlled-deployment-fixture"
    value["time_grid"].update(origin=0.0, bin_width=1.0, completed_horizon=192.0, step_count=192)
    row = value["observations"][0]
    initial = 100.0 + 10 * index
    value["instruments"][0]["initial_price"] = initial
    value["observations"] = [
        dict(
            row,
            step=step,
            last_price=initial + (step if step < 64 else 128 - step if step < 128 else 0),
            trade_count=1,
            trade_volume=1.0,
            price_state="observed",
        )
        for step in range(192)
    ]
    return SourceSnapshot(
        source_namespace="deployment-behavior",
        split_group=SplitGroup(namespace="deployment-fixture", group_id=identifier, seed=index),
        export=seal_export(ExportPayloadV2.model_validate_json(json.dumps(value))),
    )


def build(template: Path, output: Path) -> None:
    output.mkdir(mode=0o700)
    sources = tuple(source(template, index) for index in range(3))
    partitions: tuple[Partition, ...] = ("train", "validation", "test")
    dataset = seal_dataset(
        "deployment-fixture",
        tuple(
            create_snapshot_member(item, partition)
            for item, partition in zip(sources, partitions, strict=True)
        ),
    )
    profile = AnalysisConfig()
    documents: dict[str, DirectionAnnotations] = {}
    references = []
    for member, item in zip(dataset.members[:2], sources[:2], strict=True):
        annotations = annotate(Grid(member.snapshot_id, item.export), profile)
        documents[member.snapshot_id] = annotations
        references.append(
            seal_reference_set(
                ReferencePayload(
                    reference_set_id=f"deployment-{member.partition}",
                    dataset_sha256=dataset.content_sha256,
                    partition=member.partition,
                    kind="automatic",
                    policy=annotations.policy,
                    semantics=DirectionSemantics(
                        profile_sha256=analysis_profile_hash(profile),
                        decoder_version=DECODER_VERSION,
                    ),
                    analysis_profile=profile,
                    members=(create_reference_member(member, annotations),),
                )
            )
        )

    def snapshot_reader(member: SnapshotMember) -> SourceSnapshot:
        if member.partition == "test":
            raise AssertionError("deployment training must not read final observations")
        return next(
            item for item in sources if item.export.source.snapshot_id == member.snapshot_id
        )

    def reference_reader(reference: ReferenceSet, member: ReferenceMember) -> DirectionAnnotations:
        if reference.partition == "test":
            raise AssertionError("deployment training must not read final references")
        return documents[member.snapshot_id]

    data = load_training_data(
        dataset,
        references[0],
        references[1],
        snapshot_reader=snapshot_reader,
        reference_reader=reference_reader,
    )
    candidate = train_candidate(
        data,
        TrainingRequest(
            run_id="deployment-fixture",
            model_version_id="deployment-fixture-baseline",
            architecture_id="baseline",
            parameters={"regularization": [1.0], "context_sizes": [4, 16]},
            policy=TrainingPolicy(optimizer="lbfgs", seeds=(11,)),
            training_runtime=TrainingRuntime(threads=1, time_budget_seconds=90),
            verification_runtime=InferenceRuntime(threads=1),
            analysis_profile_id="direction-32/2.0",
            build_id="deployment-behavior-only",
        ),
        output / "bundle",
        registry=ArchitectureRegistry.discover(),
        worker_id="deployment-smoke",
    )
    (output / "export.json").write_bytes(
        json_bytes(source(template, 3).export.model_dump(mode="json"))
    )
    (output / "fixture.json").write_bytes(json_bytes(candidate.model_dump(mode="json")))
    print(
        json.dumps(
            {
                "fixture": "artificial-automatic",
                "model_version_id": candidate.model_version_id,
                "manifest_sha256": candidate.manifest_sha256,
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.template, args.output)
