"""Generate native market snapshots and annotate the training and validation partitions."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from itertools import cycle, islice, product
from multiprocessing import get_context
from pathlib import Path
from typing import TypedDict

import numpy as np

from regimex.benchmark_contracts import dataset_identity_keys
from regimex.benchmarks import DirectionSemantics
from regimex.data_catalog import (
    Partition,
    ReferenceMember,
    ReferencePayload,
    SnapshotGrid,
    SnapshotMember,
    SourceSnapshot,
    analysis_profile_hash,
    content_hash,
    create_reference_member,
    create_snapshot_member,
    seal_dataset,
    seal_reference_set,
    validate_reference_membership,
)
from regimex.directions import DIRECTION_POLICY, AnalysisConfig
from regimex.market_export import sample_snapshot
from regimex.market_identity import market_seed_group
from regimex.market_sampling import GENERATOR_VERSION
from regimex.segmentation import DECODER_VERSION, annotate
from regimex.storage import digest, json_bytes, write_new_file

PROFILE = AnalysisConfig(version="2.2")
COUNTS: dict[Partition, int] = {"train": 1536, "validation": 192, "test": 384}
LENGTHS = (256, 512, 1024)
PLAN_SEED = 2026112301
TRANSFER_SEEDS = range(3_961_177_024, 3_961_177_056)


class PopulationRow(TypedDict):
    partition: Partition
    seed: int
    steps: int
    instruments: int


class Replacement(TypedDict):
    excluded: PopulationRow
    duplicate_keys_sha256: list[str]
    replacement_seed: int


def publish(path: Path, value: object) -> None:
    data = json_bytes(value)
    try:
        write_new_file(path, data)
    except FileExistsError:
        if path.read_bytes() != data:
            raise ValueError(f"Existing file differs from the registered content: {path}") from None


def reserve(
    output: Path, inventory: Path, dataset: Path, transfer: Path
) -> tuple[list[PopulationRow], set[str]]:
    from build_exposure_index import build

    prior = build([dataset], [inventory])
    seen = set(prior["excluded_seeds"])
    seen.update(json.loads(transfer.read_bytes())["protocol"]["groups"])
    if seen.intersection(TRANSFER_SEEDS):
        raise ValueError("Transfer seed reservation overlaps prior exposures")
    seen.update(TRANSFER_SEEDS)
    prior["excluded_seeds"] = sorted(seen)
    publish(output / "prior.json", prior)
    rng = np.random.Generator(np.random.PCG64(PLAN_SEED))
    rows: list[PopulationRow] = []
    for role, count in COUNTS.items():
        for instruments, steps in islice(cycle(product((1, 2), LENGTHS)), count):
            while (seed := int(rng.integers(0, 2**32, dtype=np.uint32))) in seen:
                continue
            seen.add(seed)
            rows.append(
                {
                    "partition": role,
                    "seed": seed,
                    "steps": steps,
                    "instruments": instruments,
                }
            )
    publish(
        output / "candidate-plan.json",
        {
            "version": "direction-large/1",
            "generator_version": GENERATOR_VERSION,
            "analysis_profile": PROFILE.model_dump(mode="json"),
            "rng": "PCG64",
            "plan_seed": PLAN_SEED,
            "prior_sha256": content_hash(prior),
            "rows": rows,
        },
    )
    return rows, set(prior["identity_keys_sha256"])


def generate(row: PopulationRow, *, output: Path) -> None:
    role, seed = row["partition"], row["seed"]
    folder = output / role
    filename = f"{seed}.json"
    export = sample_snapshot(seed=seed, steps=row["steps"], instruments=row["instruments"])
    snapshot = SourceSnapshot(
        source_namespace=f"dir-large-{seed}",
        split_group=market_seed_group(seed),
        export=export,
    )
    member = create_snapshot_member(snapshot, role)
    publish(folder / "exports" / filename, export.model_dump(mode="json"))
    if role != "test":
        grid = SnapshotGrid.from_export(export)
        annotation = annotate(grid, PROFILE)
        annotation.validate_for(grid, export_sha256=member.export_sha256)
        reference = create_reference_member(member, annotation)
        publish(folder / "annotations" / filename, annotation.model_dump(mode="json"))
        publish(folder / "reference-members" / filename, reference.model_dump(mode="json"))
    # Consumers may begin preparing a training sample as soon as its member appears.
    publish(folder / "members" / filename, member.model_dump(mode="json"))


def remove_duplicate(output: Path, row: PopulationRow) -> None:
    role = output / row["partition"]
    for folder in ("members", "reference-members", "annotations", "exports"):
        path = role / folder / f"{row['seed']}.json"
        if path.exists():
            path.unlink()


def resolve_duplicates(output: Path, rows: list[PopulationRow], prior_keys: set[str]) -> None:
    prior = json.loads((output / "prior.json").read_bytes())
    used_seeds = set(prior["excluded_seeds"]).union(row["seed"] for row in rows)
    used_keys = prior_keys.copy()
    # Keep replacement draws independent of how many candidates the plan requested.
    rng = np.random.Generator(np.random.PCG64(PLAN_SEED + 1))
    replacements: list[Replacement] = []
    for index, row in enumerate(rows):
        while True:
            path = output / row["partition"] / "members" / f"{row['seed']}.json"
            member = SnapshotMember.model_validate_json(path.read_bytes())
            keys = {content_hash(k) for k in dataset_identity_keys((member,))}
            duplicates = keys & used_keys
            if not duplicates:
                used_keys.update(keys)
                rows[index] = row
                break
            while (seed := int(rng.integers(0, 2**32, dtype=np.uint32))) in used_seeds:
                continue
            used_seeds.add(seed)
            replacements.append(
                {
                    "excluded": row,
                    "duplicate_keys_sha256": sorted(duplicates),
                    "replacement_seed": seed,
                }
            )
            row = {**row, "seed": seed}
            generate(row, output=output)
    plan = json.loads((output / "candidate-plan.json").read_bytes())
    publish(
        output / "population.json",
        {
            **plan,
            "rows": rows,
            "candidate_plan_sha256": digest(output / "candidate-plan.json"),
            "replacements": replacements,
        },
    )
    for replacement in replacements:
        remove_duplicate(output, replacement["excluded"])


def seal(output: Path, rows: list[PopulationRow], prior_keys: set[str]) -> None:
    members = tuple(
        SnapshotMember.model_validate_json(
            (output / row["partition"] / "members" / f"{row['seed']}.json").read_bytes()
        )
        for row in rows
    )
    if any(content_hash(key) in prior_keys for key in dataset_identity_keys(members)):
        raise ValueError("Generated population overlaps a recorded exposure")
    dataset = seal_dataset("direction-large-v1", members)
    manifest = dataset.model_dump(mode="json")
    publish(output / "dataset.json", manifest)
    reference_hashes = {}
    location_hashes = {}
    for role in COUNTS:
        folder = output / role
        role_members = tuple(m for m in members if m.partition == role)
        locations = {m.source_namespace: f"exports/{m.split_group.seed}.json" for m in role_members}
        publish(folder / "dataset.json", manifest)
        publish(folder / "locations.json", locations)
        location_hashes[role] = content_hash(locations)
        if role != "test":
            references = seal_reference_set(
                ReferencePayload(
                    reference_set_id=f"large-direction-{role}",
                    dataset_sha256=dataset.content_sha256,
                    partition=role,
                    kind="automatic",
                    policy=DIRECTION_POLICY,
                    analysis_profile=PROFILE,
                    semantics=DirectionSemantics(
                        profile_sha256=analysis_profile_hash(PROFILE),
                        decoder_version=DECODER_VERSION,
                    ),
                    members=tuple(
                        ReferenceMember.model_validate_json(
                            (
                                folder / "reference-members" / f"{m.split_group.seed}.json"
                            ).read_bytes()
                        )
                        for m in role_members
                    ),
                ),
            )
            validate_reference_membership(dataset, references)
            publish(folder / "references.json", references.model_dump(mode="json"))
            reference_hashes[role] = references.content_sha256
        publish(folder / "reservation.json", {"dataset_sha256": dataset.content_sha256})
    publish(
        output / "reservation.json",
        {
            "dataset_sha256": dataset.content_sha256,
            "prior_sha256": digest(output / "prior.json"),
            "population_sha256": digest(output / "population.json"),
            "references_sha256": reference_hashes,
            "locations_sha256": location_hashes,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--transfer", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    rows, prior_keys = reserve(args.output, args.inventory, args.dataset, args.transfer)
    print(f"Reserved {len(rows)} experiments", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
        for index, _ in enumerate(pool.map(partial(generate, output=args.output), rows), start=1):
            if index % 96 == 0:
                print(f"Completed {index}/{len(rows)}", flush=True)
    resolve_duplicates(args.output, rows, prior_keys)
    seal(args.output, rows, prior_keys)


if __name__ == "__main__":
    main()
