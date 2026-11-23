"""Fit forests against mixtures of native profile labels and independent latent drift."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from functools import partial
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from latent_direction_training import (
    FIRST_SEED,
    LENGTHS,
    PATHS,
    SOURCE_ID,
    generate,
    observations,
    recipe,
)
from train_large_direction import PROFILE, SEED, extract

from regimex.backends import adaptive_forest, multiscale_segments
from regimex.backends.adaptive import environment
from regimex.data_catalog import (
    DatasetManifest,
    ReferenceSet,
    content_hash,
    validate_reference_membership,
)
from regimex.model_registry import AdaptiveParameters
from regimex.storage import digest, output_directory, write_json

MIXTURES = {"mixed-10": 0.10, "mixed-25": 0.25, "mixed-45": 0.45}
PARAMETERS = AdaptiveParameters(
    trees=80, max_depth=15, feature_set="multiscale", sample_weighting="interval"
)


def hashes(prices, counts, targets):
    observation_hash = content_hash(
        {
            "last_price": [None if np.isnan(p) else float(p) for p in prices],
            "trade_count": [None if c < 0 else int(c) for c in counts],
        }
    )
    return observation_hash, content_hash(targets.tolist())


def build_path(index, *, folder):
    seed, steps = FIRST_SEED + index, LENGTHS[index % len(LENGTHS)]
    prices, counts, _, targets = generate(seed, steps, PROFILE)
    path = folder / f"{seed}.npz"
    np.savez_compressed(path, prices=prices, counts=counts, targets=targets)
    observation_hash, target_hash = hashes(prices, counts, targets)
    return {
        "seed": seed,
        "steps": steps,
        "observations_sha256": observation_hash,
        "targets_sha256": target_hash,
        "file_sha256": digest(path),
        "classified_steps": int((targets >= 0).sum()),
    }


def prepare(data, workers):
    with output_directory(data / "latent-training") as dest:
        write_json(dest / "recipe.json", recipe())
        paths = dest / "paths"
        paths.mkdir()
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
            records = list(
                pool.map(
                    partial(build_path, folder=paths),
                    range(PATHS),
                    chunksize=8,
                    buffersize=workers * 2,
                )
            )
        manifest = {
            "source_id": SOURCE_ID,
            "experiment_count": len(records),
            "classified_steps": sum(r["classified_steps"] for r in records),
            "recipe_sha256": digest(dest / "recipe.json"),
            "exports_sha256": content_hash(
                [(r["seed"], r["observations_sha256"]) for r in records]
            ),
            "references_sha256": content_hash([(r["seed"], r["targets_sha256"]) for r in records]),
            "members": records,
        }
        write_json(dest / "manifest.json", manifest)


def latent_table(record, *, folder):
    path = folder / "paths" / f"{record['seed']}.npz"
    if digest(path) != record["file_sha256"]:
        raise ValueError("Latent training file differs from its manifest")
    with np.load(path, allow_pickle=False) as source:
        prices, counts, targets = (source[key] for key in ("prices", "counts", "targets"))
    if hashes(prices, counts, targets) != (record["observations_sha256"], record["targets_sha256"]):
        raise ValueError("Unexpected latent training observations or targets")
    rows = observations(prices, counts)
    return adaptive_forest.training_rows(multiscale_segments.features(rows, PROFILE), targets)


def class_totals(tables, field):
    return np.sum(
        [
            np.bincount(t.table[:, -1].astype(np.int64), weights=getattr(t, field), minlength=3)
            for t in tables
        ],
        axis=0,
    )


def mixture(native, latent, fraction):
    """Keep the class prior fixed while changing each source's share of every class."""
    combined_counts = class_totals(native, "counts") + class_totals(latent, "counts")
    output = []
    for rows, share in ((native, 1 - fraction), (latent, fraction)):
        measured = class_totals(rows, "interval_weights")
        if np.any(measured <= 0):
            raise ValueError("Each training population must contain all three directions")
        multipliers = share * combined_counts / measured
        output.extend(
            replace(
                row,
                interval_weights=row.interval_weights
                * multipliers[row.table[:, -1].astype(np.int64)],
            )
            for row in rows
        )
    return output


def fit(data, workers):
    folder = data / "train"
    dataset = DatasetManifest.model_validate_json((folder / "dataset.json").read_bytes())
    labels = ReferenceSet.model_validate_json((folder / "references.json").read_bytes())
    reserve = json.loads((data / "reservation.json").read_bytes())
    if (
        labels.content_sha256 != reserve["references_sha256"]["train"]
        or labels.partition != "train"
        or labels.analysis_profile != PROFILE
        or dataset.content_sha256 != reserve["dataset_sha256"]
        or digest(folder / "locations.json") != reserve["locations_sha256"]["train"]
    ):
        raise ValueError("Native training partition differs from the reserved population")
    validate_reference_membership(dataset, labels)
    ref = {r.source_namespace: r for r in labels.members}
    locations = json.loads((folder / "locations.json").read_bytes())
    members = [m for m in dataset.members if m.partition == "train"]
    items = [(m, ref[m.source_namespace], locations[m.source_namespace]) for m in members]
    latent_dir = data / "latent-training"
    manifest = json.loads((latent_dir / "manifest.json").read_bytes())
    records = manifest["members"]
    seeds = [r["seed"] for r in records]
    if (
        seeds != list(range(FIRST_SEED, FIRST_SEED + PATHS))
        or manifest["source_id"] != SOURCE_ID
        or manifest["experiment_count"] != len(records)
        or manifest["recipe_sha256"] != content_hash(recipe())
        or manifest["recipe_sha256"] != digest(latent_dir / "recipe.json")
        or manifest["classified_steps"] != sum(r["classified_steps"] for r in records)
        or manifest["exports_sha256"]
        != content_hash([(r["seed"], r["observations_sha256"]) for r in records])
        or manifest["references_sha256"]
        != content_hash([(r["seed"], r["targets_sha256"]) for r in records])
        or set(seeds).intersection(m.split_group.seed for m in dataset.members)
    ):
        raise ValueError("Unexpected or overlapping latent train membership")
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        native = []
        for i, block in enumerate(
            pool.map(partial(extract, folder=folder), items, chunksize=2, buffersize=workers * 2), 1
        ):
            native.extend(block["multiscale"])
            if i % 384 == 0:
                print(f"Extracted {i}/{len(items)} native experiments", flush=True)
        latent = list(
            pool.map(
                partial(latent_table, folder=latent_dir),
                records,
                chunksize=3,
                buffersize=workers * 2,
            )
        )
    census = class_totals(native, "counts").astype(np.int64).tolist()
    additional = {k: v for k, v in manifest.items() if k != "members"}
    additional["manifest_sha256"] = digest(latent_dir / "manifest.json")
    evidence = {
        "parameters": PARAMETERS.model_dump(mode="json"),
        "seed": SEED,
        "threads": 1,
        "feature_workers": workers,
        "environment": environment(),
        "dataset_sha256": dataset.content_sha256,
        "references_sha256": labels.content_sha256,
        "training_receipt_sha256": digest(folder / "reservation.json"),
        "population_receipt_sha256": digest(data / "reservation.json"),
        "training_experiments": len(members),
        "training_sequences": len(native),
        "labeled_steps_by_class": census,
        "additional_labeled_steps_by_class": class_totals(latent, "counts")
        .astype(np.int64)
        .tolist(),
        "source_sha256": {
            p.name: digest(p)
            for p in (
                Path(__file__),
                Path(__file__).with_name("train_large_direction.py"),
                Path(adaptive_forest.__file__),
                Path(multiscale_segments.__file__),
            )
        },
    }
    with output_directory(data / "fitted") as stage:
        for name, fraction in MIXTURES.items():
            rows = mixture(native, latent, fraction)
            forest = adaptive_forest.fit_training_rows(
                rows,
                interval_weighting=True,
                trees=PARAMETERS.trees,
                max_depth=PARAMETERS.max_depth,
                seed=SEED,
            )
            forest.validate(trees=PARAMETERS.trees, max_depth=PARAMETERS.max_depth)
            path = stage / f"{name}.npz"
            forest.save(path)
            write_json(
                stage / f"{name}.json",
                {
                    **evidence,
                    "weights_sha256": digest(path),
                    "numeric_bytes": forest.numeric_bytes,
                    "additional_training_source": {**additional, "weight_fraction": fraction},
                },
            )
            print(f"Trained {name} ({forest.numeric_bytes:,} numeric bytes)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "fit"))
    parser.add_argument("data", type=Path)
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=3)
    args = parser.parse_args()
    (prepare if args.phase == "prepare" else fit)(args.data, args.workers)


if __name__ == "__main__":
    main()
