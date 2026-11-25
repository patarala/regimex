"""Fit three direction forests from sealed training exports, one instrument at a time."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from multiprocessing import get_context
from pathlib import Path
from time import monotonic

import numpy as np

from regimex.application.analysis_computation import snapshot_inputs
from regimex.backends import adaptive_forest, multiscale_segments, price_segments
from regimex.backends.adaptive import environment
from regimex.data_catalog import (
    DatasetManifest,
    ReferenceSet,
    SnapshotGrid,
    create_reference_member,
    validate_reference_membership,
)
from regimex.directions import AnalysisConfig, DirectionAnnotations
from regimex.evaluation import dense_labels
from regimex.exports import load_export_v2
from regimex.model_registry import AdaptiveParameters
from regimex.storage import digest, output_directory, write_json

SEED = 173
PROFILE = AnalysisConfig(version="2.2")
_CAPACITY = {"trees": 80, "max_depth": 15}
CANDIDATES = {
    "greedy-large": AdaptiveParameters(**_CAPACITY),
    "multiscale-large": AdaptiveParameters(feature_set="multiscale", **_CAPACITY),
    "multiscale-balanced": AdaptiveParameters(
        feature_set="multiscale", sample_weighting="interval", **_CAPACITY
    ),
}


def extract(item, *, folder, feature_sets=("greedy", "multiscale")):
    member, reference, location = item
    export = load_export_v2(folder / location)
    labels = DirectionAnnotations.model_validate_json(
        (folder / "annotations" / f"{member.split_group.seed}.json").read_bytes()
    )
    if (
        export.content_sha256 != member.export_sha256
        or create_reference_member(member, labels) != reference
        or AnalysisConfig.model_validate(labels.policy_config) != PROFILE
        or labels.reference_kind != "automatic"
    ):
        raise ValueError(f"Unregistered export or training targets for {member.snapshot_id}")
    labels.validate_for(SnapshotGrid.from_export(export), export_sha256=member.export_sha256)
    inputs = snapshot_inputs(export)
    extractors = {"greedy": adaptive_forest.features, "multiscale": multiscale_segments.features}
    output = {name: [] for name in feature_sets}
    for info, rows in zip(inputs.info, inputs.observations, strict=True):
        intervals = tuple(i for i in labels.intervals if i.instrument_id == info.instrument_id)
        targets = dense_labels(intervals, info.step_count)
        for name, tables in output.items():
            tables.append(adaptive_forest.training_rows(extractors[name](rows, PROFILE), targets))
    return output


def train(data: Path, workers: int) -> None:
    folder = data / "train"
    dataset = DatasetManifest.model_validate_json((folder / "dataset.json").read_bytes())
    references = ReferenceSet.model_validate_json((folder / "references.json").read_bytes())
    receipt = json.loads((folder / "reservation.json").read_bytes())
    reservation = json.loads((data / "reservation.json").read_bytes())
    locations = json.loads((folder / "locations.json").read_bytes())
    if (
        dataset.content_sha256 != receipt["dataset_sha256"]
        or dataset.content_sha256 != reservation["dataset_sha256"]
        or references.content_sha256 != reservation["references_sha256"]["train"]
        or digest(folder / "locations.json") != reservation["locations_sha256"]["train"]
        or references.partition != "train"
        or references.analysis_profile != PROFILE
    ):
        raise ValueError("Unexpected training population or reference profile")
    validate_reference_membership(dataset, references)
    members = [m for m in dataset.members if m.partition == "train"]
    reference_by_source = {ref.source_namespace: ref for ref in references.members}
    items = [
        (m, reference_by_source[m.source_namespace], locations[m.source_namespace]) for m in members
    ]
    tables = {"greedy": [], "multiscale": []}
    started = monotonic()
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        results = pool.map(
            partial(extract, folder=folder), items, chunksize=2, buffersize=workers * 2
        )
        for i, result in enumerate(results, 1):
            for name in tables:
                tables[name].extend(result[name])
            if i % 128 == 0:
                print(f"Extracted {i}/{len(items)} experiments", flush=True)
    print(f"Feature extraction took {monotonic() - started:.1f} s", flush=True)
    census = np.sum(
        [
            np.bincount(row.table[:, -1].astype(np.int64), weights=row.counts, minlength=3)
            for row in tables["greedy"]
        ],
        axis=0,
    )
    evidence = {
        "seed": SEED,
        "threads": 1,
        "feature_workers": workers,
        "training_receipt_sha256": digest(folder / "reservation.json"),
        "population_receipt_sha256": digest(data / "reservation.json"),
        "dataset_sha256": dataset.content_sha256,
        "references_sha256": references.content_sha256,
        "training_experiments": len(items),
        "training_sequences": len(tables["greedy"]),
        "labeled_steps_by_class": census.astype(np.int64).tolist(),
        "environment": environment(),
        "source_sha256": {
            p.name: digest(p)
            for p in (
                Path(__file__),
                Path(adaptive_forest.__file__),
                Path(multiscale_segments.__file__),
                Path(price_segments.__file__),
            )
        },
    }
    with output_directory(data / "fitted") as stage:
        for name, settings in CANDIDATES.items():
            print(f"Fitting {name}", flush=True)
            started = monotonic()
            table = tables[settings.feature_set]
            forest = adaptive_forest.fit_training_rows(
                table,
                interval_weighting=settings.sample_weighting == "interval",
                trees=settings.trees,
                max_depth=settings.max_depth,
                seed=SEED,
            )
            path = stage / f"{name}.npz"
            forest.validate(trees=settings.trees, max_depth=settings.max_depth)
            forest.save(path)
            write_json(
                stage / f"{name}.json",
                {
                    **evidence,
                    "parameters": settings.model_dump(mode="json"),
                    "weights_sha256": digest(path),
                    "numeric_bytes": forest.numeric_bytes,
                    "compressed_training_rows": sum(len(row.table) for row in table),
                },
            )
            print(
                f"{name} trained in {monotonic() - started:.1f} s "
                f"({forest.numeric_bytes:,} numeric bytes)",
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=3)
    args = parser.parse_args()
    train(args.data, args.workers)


if __name__ == "__main__":
    main()
