"""Score a saved bundle on the simulator references used in its original fit."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from regimex.application.analysis_computation import restore_bundle, snapshot_inputs
from regimex.data_catalog import (
    DatasetManifest,
    ReferenceSet,
    SnapshotGrid,
    SourceSnapshot,
    content_hash,
    create_reference_member,
    create_snapshot_member,
    validate_reference_membership,
)
from regimex.direction_quality import DirectionQualityReport, GroupCounts, MeasuredCohort
from regimex.directions import DIRECTION_NAMES
from regimex.evaluation import _segment_counts, probabilities_valid
from regimex.exports import load_export_v2
from regimex.model_bundles import MAX_BUNDLE_MANIFEST_BYTES, ModelBundle
from regimex.model_registry import ArchitectureRegistry, InferenceRuntime
from regimex.segmentation import annotate, decode
from regimex.service_client import read_file
from regimex.storage import digest, json_bytes, write_new_file


def record_group(member, exposures):
    group = member.split_group
    identity = (
        content_hash(["group", group.namespace, group.group_id])
        if group.seed is None
        else content_hash(["seed", group.namespace, str(group.seed)])
    )
    keys = member.run_and_group_keys()
    keys.add(("export", member.export_sha256))
    keys.update(("observations", fingerprint) for fingerprint in member.observation_hashes)
    for key in keys:
        if exposures.setdefault(key, (member.partition, identity)) != (member.partition, identity):
            raise ValueError("Related observations have conflicting group or split identities.")
    return identity


def measure(dataset_path, references_path, locations_path, predictor, bundle, runtime, exposures):
    dataset = DatasetManifest.model_validate_json(dataset_path.read_bytes())
    references = ReferenceSet.model_validate_json(references_path.read_bytes())
    split = references.partition
    if split not in {"train", "validation"} or references.kind != "automatic":
        raise ValueError("Choose an automatic training or validation reference set.")
    training = bundle.training
    expected_dataset = training.dataset_sha256
    expected_reference = training.labels
    if split == "validation":
        expected_dataset = training.validation_dataset_sha256 or expected_dataset
        expected_reference = training.validation_references
    if dataset.content_sha256 != expected_dataset or references.identity() != expected_reference:
        raise ValueError("Dataset or references differ from the original fit.")
    validate_reference_membership(dataset, references)
    dataset.require_native_seed_groups()
    descriptor = ArchitectureRegistry.discover().get(bundle.architecture_id)
    locations = json.loads(locations_path.read_bytes())
    counts = defaultdict(lambda: np.zeros((3, 3), dtype=np.int64))
    members = (member for member in dataset.members if member.partition == split)
    for member, reference in zip(members, references.members, strict=True):
        identity = record_group(member, exposures)
        export = load_export_v2(locations_path.parent / locations[member.source_namespace])
        source = SourceSnapshot(
            source_namespace=member.source_namespace, split_group=member.split_group, export=export
        )
        if create_snapshot_member(source, split) != member:
            raise ValueError("Export or observations differ from the registered member.")
        descriptor.validate_observation_fields(export.capabilities.observation_fields())
        labels = annotate(SnapshotGrid.from_export(export), references.analysis_profile)
        if create_reference_member(member, labels) != reference:
            raise ValueError("Regenerated annotations differ from the original references.")
        inputs = snapshot_inputs(export)
        scores = predictor.predict(inputs, runtime=runtime)
        probabilities_valid(scores, inputs.row_count)
        offset = 0
        for info, rows in zip(inputs.info, inputs.observations, strict=True):
            expected = tuple(i for i in labels.intervals if i.instrument_id == info.instrument_id)
            end = offset + len(rows)
            predicted = decode(scores[offset:end], rows, predictor.analysis)
            counts[identity] += _segment_counts(expected, predicted)
            offset = end
    return MeasuredCohort(
        reference="simulator",
        split=split,
        reference_sha256=references.content_sha256,
        groups=tuple(
            GroupCounts(group_sha256=key, counts=tuple(tuple(row) for row in value.tolist()))
            for key, value in sorted(counts.items())
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument(
        "--manifest-sha256", required=True, help="Recorded when the bundle was saved"
    )
    parser.add_argument("--protocol", type=Path, required=True, help="Existing evaluation plan")
    parser.add_argument(
        "--cohort",
        nargs=3,
        type=Path,
        action="append",
        required=True,
        metavar=("DATASET", "REFERENCES", "LOCATIONS"),
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="New direction quality JSON file"
    )
    args = parser.parse_args()
    protocol_sha256 = digest(args.protocol)
    bundle = ModelBundle.model_validate_json(
        read_file(args.bundle / "manifest.json", MAX_BUNDLE_MANIFEST_BYTES)
    )
    runtime = InferenceRuntime(threads=1)
    predictor, bundle = restore_bundle(
        args.bundle,
        args.manifest_sha256,
        bundle.model_version_id,
        bundle.architecture_id,
        runtime,
        ArchitectureRegistry.discover(),
        worker_id="direction-quality",
    )
    exposures = {}
    report = DirectionQualityReport(
        metric=bundle.training.selection_metric,
        class_names=DIRECTION_NAMES,
        bundle_sha256=args.manifest_sha256,
        protocol_sha256=protocol_sha256,
        rows=tuple(measure(*paths, predictor, bundle, runtime, exposures) for paths in args.cohort),
    )
    evidence = report.summarize(content_hash(report.model_dump(mode="json")))
    evidence.validate_for(args.manifest_sha256, bundle.training)
    write_new_file(args.output, json_bytes(report.model_dump(mode="json")))
    for row in evidence.rows:
        print(f"{row.split}: {row.interval_macro_f1:.5f} across {row.group_count} groups")


if __name__ == "__main__":
    main()
