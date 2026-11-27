"""Collect previously used generator seeds and dataset identities from named sources.

python scripts/build_exposure_index.py var/prior.json --dataset var/tcn-widths/dataset.json
"""

import argparse
from pathlib import Path

from compare_tcn_slopes import ExposureIndex
from pydantic import TypeAdapter

from regimex.benchmark_contracts import dataset_identity_keys
from regimex.contracts import Seed
from regimex.data_catalog import DatasetManifest, content_hash
from regimex.storage import json_bytes, write_new_file


def build(datasets: list[Path], indexes: list[Path]) -> ExposureIndex:
    seeds: set[int] = set()
    keys: set[str] = set()
    for path in datasets:
        manifest = DatasetManifest.for_modeling(path.read_bytes())
        keys.update(content_hash(key) for key in dataset_identity_keys(manifest.members))
        seeds.update(m.split_group.seed for m in manifest.members if m.split_group.seed is not None)
    index_type = TypeAdapter(ExposureIndex)
    seed_type = TypeAdapter(list[Seed])
    for path in indexes:
        prior = index_type.validate_json(path.read_bytes(), strict=True)
        seeds.update(seed_type.validate_python(prior["excluded_seeds"], strict=True))
        keys.update(prior["identity_keys_sha256"])
    if not seeds and not keys:
        raise ValueError("The supplied sources contain no known exposures.")
    return {"excluded_seeds": sorted(seeds), "identity_keys_sha256": sorted(keys)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="New file for the merged inventory.")
    parser.add_argument(
        "--dataset", type=Path, action="append", default=[], help="Dataset manifest (repeatable)."
    )
    parser.add_argument(
        "--index", type=Path, action="append", default=[], help="Prior exposure index (repeatable)."
    )
    args = parser.parse_args()
    if not args.dataset and not args.index:
        parser.error("Supply at least one --dataset or --index.")
    try:
        merged = build(args.dataset, args.index)
        write_new_file(args.output, json_bytes(merged))
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    print(
        f"{len(merged['excluded_seeds'])} seeds and "
        f"{len(merged['identity_keys_sha256'])} identity keys"
    )
