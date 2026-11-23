"""Reserve fresh market validation and test cohorts for the ensemble comparison."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from latent_direction_training import observations
from prepare_mixed_direction import NativeRound, prepare

from regimex.contracts import Observation
from regimex.data_catalog import content_hash
from regimex.storage import digest

ENSEMBLE = NativeRound(
    "78108a9fdb1ac0cf757a046f8bb7b07e7c89365bc5d919a612e1f9e578e51704",
    2026112407,
    (range(3_961_177_300, 3_961_177_332), range(3_961_177_400, 3_961_177_464)),
    "direction-ensemble/1",
    "direction-ensemble-v3",
    "ensemble-native",
)


def observation_fingerprint(rows: tuple[Observation, ...]) -> str:
    """Match the catalog's whole-instrument observations/1.0 hash for a non-export source."""
    fields = Observation.model_fields.keys() - {"instrument_id"}
    values = []
    for row in rows:
        observed = {name: getattr(row, name) for name in fields}
        for name, value in observed.items():
            if isinstance(value, float) and value == 0:
                observed[name] = 0.0
        values.append(observed)
    return content_hash({"version": "observations/1.0", "rows": values})


def latent_fingerprint(record, *, folder):
    path = folder / "paths" / f"{record['seed']}.npz"
    if digest(path) != record["file_sha256"]:
        raise ValueError("Earlier latent path differs from its registered file")
    with np.load(path, allow_pickle=False) as data:
        return observation_fingerprint(observations(data["prices"], data["counts"]))


def controlled_fingerprints(item):
    from check_adaptive_gap_transfer import pairs

    index, seed = item
    return [
        observation_fingerprint(view.rows)
        for full, masked, _ in pairs(seed, index)
        for view in (full, masked)
    ]


def added_exposures(source: Path, workers: int) -> frozenset[str]:
    from check_large_direction_transfer import GROUPS as first_transfer
    from check_mixed_direction_transfer import GROUPS as second_transfer

    folder = source / "latent-training"
    records = json.loads((folder / "manifest.json").read_bytes())["members"]
    controls = [
        item
        for cohort in (first_transfer, second_transfer["validation"], second_transfer["test"])
        for item in enumerate(cohort)
    ]
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        fingerprints = set(
            pool.map(partial(latent_fingerprint, folder=folder), records, chunksize=8)
        )
        for sample in pool.map(controlled_fingerprints, controls):
            fingerprints.update(sample)
    print(f"Indexed {len(fingerprints)} earlier latent and controlled paths", flush=True)
    return frozenset(content_hash(("observation", fingerprint)) for fingerprint in fingerprints)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source", type=Path, default=Path("var/verification/direction-mixed-v2"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    keys = added_exposures(args.source, args.workers)
    prepare(args.source, args.output, args.workers, ENSEMBLE, keys)
