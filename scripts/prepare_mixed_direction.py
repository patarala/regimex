"""Reuse native training observations and generate fresh evaluation markets for mixed training."""

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from itertools import cycle, islice, product
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from prepare_large_direction import (
    LENGTHS,
    PROFILE,
    PopulationRow,
    Replacement,
    generate,
    publish,
    remove_duplicate,
)

from regimex.benchmark_contracts import dataset_identity_keys
from regimex.benchmarks import DirectionSemantics
from regimex.data_catalog import (
    DatasetManifest,
    Partition,
    ReferenceMember,
    ReferencePayload,
    ReferenceSet,
    SnapshotMember,
    analysis_profile_hash,
    content_hash,
    seal_dataset,
    seal_reference_set,
    validate_reference_membership,
)
from regimex.directions import DECODER_VERSION, DIRECTION_POLICY
from regimex.exports import ExportV2
from regimex.market_sampling import GENERATOR_VERSION
from regimex.storage import digest, write_new_file

SOURCE_DATASET = "88f2b8d3f5340c6f04b49793f2ff2bdaf8a12faaefd8e53395a77cd634e91649"
PLAN_SEED = 2026112389
CONTROLLED_TRAIN = range(2_651_019_000, 2_651_020_152)
CONTROLLED_VALIDATION = range(3_961_177_100, 3_961_177_164)
CONTROLLED_TEST = range(3_961_177_200, 3_961_177_264)
FRESH_COUNTS: dict[Partition, int] = {"validation": 192, "test": 384}


@dataclass(frozen=True)
class NativeRound:
    source_dataset: str
    seed: int
    controlled_groups: tuple[range, ...]
    version: str
    dataset_id: str
    reference_prefix: str


MIXED = NativeRound(
    SOURCE_DATASET,
    PLAN_SEED,
    (CONTROLLED_TRAIN, CONTROLLED_VALIDATION, CONTROLLED_TEST),
    "direction-mixed/1",
    "direction-mixed-v2",
    "mixed-native",
)


def source_manifest(source: Path, design: NativeRound = MIXED) -> DatasetManifest:
    manifest = DatasetManifest.for_modeling((source / "dataset.json").read_bytes())
    receipt = json.loads((source / "reservation.json").read_bytes())
    if (
        manifest.content_sha256 != design.source_dataset
        or receipt["dataset_sha256"] != design.source_dataset
        or digest(source / "population.json") != receipt["population_sha256"]
        or digest(source / "prior.json") != receipt["prior_sha256"]
        or digest(source / "candidate-plan.json")
        != json.loads((source / "population.json").read_bytes())["candidate_plan_sha256"]
    ):
        raise ValueError("Earlier native population differs from the expected reservation")
    return manifest


def prepare_rows(
    output: Path,
    source: Path,
    dataset: DatasetManifest,
    design: NativeRound = MIXED,
    extra_keys: frozenset[str] = frozenset(),
):
    from build_exposure_index import build

    prior = build([source / "dataset.json"], [source / "prior.json"])
    prior["identity_keys_sha256"] = sorted(set(prior["identity_keys_sha256"]) | extra_keys)
    initial = json.loads((source / "candidate-plan.json").read_bytes())
    used = set(prior["excluded_seeds"]).union(r["seed"] for r in initial["rows"])
    controlled = {seed for cohort in design.controlled_groups for seed in cohort}
    if len(controlled) != sum(map(len, design.controlled_groups)):
        raise ValueError("Controlled cohorts overlap")
    if controlled & used:
        raise ValueError("Controlled cohorts intersect previously used seeds")
    used.update(controlled)
    prior["excluded_seeds"] = sorted(used)
    publish(output / "prior.json", prior)
    training: list[PopulationRow] = [
        {
            "partition": "train",
            "seed": m.split_group.seed,
            "steps": m.step_count,
            "instruments": len(m.instrument_ids),
        }
        for m in dataset.members
        if m.partition == "train" and m.split_group.seed is not None
    ]
    if len(training) != 1536:
        raise ValueError("Incomplete original training membership")
    rng = np.random.Generator(np.random.PCG64(design.seed))
    fresh: list[PopulationRow] = []
    for role, count in FRESH_COUNTS.items():
        for instruments, steps in islice(cycle(product((1, 2), LENGTHS)), count):
            while (seed := int(rng.integers(0, 2**32, dtype=np.uint32))) in used:
                continue
            used.add(seed)
            fresh.append(
                {"partition": role, "seed": seed, "steps": steps, "instruments": instruments}
            )
    plan = {
        "version": design.version,
        "generator_version": GENERATOR_VERSION,
        "analysis_profile": PROFILE.model_dump(mode="json"),
        "rng": "PCG64",
        "plan_seed": design.seed,
        "prior_sha256": content_hash(prior),
        "source_dataset_sha256": design.source_dataset,
        "source_receipt_sha256": digest(source / "reservation.json"),
        "rows": [*training, *fresh],
    }
    publish(output / "candidate-plan.json", plan)
    return training, fresh, prior, plan


def copy_training(source: Path, output: Path, dataset: DatasetManifest) -> None:
    refs = ReferenceSet.model_validate_json((source / "train/references.json").read_bytes())
    validate_reference_membership(dataset, refs)
    receipt = json.loads((source / "reservation.json").read_bytes())
    if refs.content_sha256 != receipt["references_sha256"]["train"] or refs.partition != "train":
        raise ValueError("Retained references differ from the previous training reservation")
    members = (member for member in dataset.members if member.partition == "train")
    for member, reference in zip(members, refs.members, strict=True):
        filename = f"{member.split_group.seed}.json"
        for folder in ("exports", "annotations"):
            name = Path("train", folder, filename)
            raw = (source / name).read_bytes()
            if folder == "exports":
                member.bind_export(ExportV2.model_validate_json(raw))
            elif hashlib.sha256(raw).hexdigest() != reference.annotations_sha256:
                raise ValueError(f"Retained training input differs from its manifest: {name}")
            path = output / name
            if path.exists():
                if path.read_bytes() != raw:
                    raise ValueError(f"Retained training input changed: {name}")
            else:
                write_new_file(path, raw)
        publish(output / "train/reference-members" / filename, reference.model_dump(mode="json"))
        publish(output / "train/members" / filename, member.model_dump(mode="json"))


def accept_fresh(
    output: Path,
    fresh: list[PopulationRow],
    prior,
    design: NativeRound = MIXED,
) -> list[Replacement]:
    used_keys = set(prior["identity_keys_sha256"])
    used_seeds = set(prior["excluded_seeds"]).union(row["seed"] for row in fresh)
    rng = np.random.Generator(np.random.PCG64(design.seed + 1))
    replacements: list[Replacement] = []
    for index, row in enumerate(fresh):
        while True:
            member = load_member(output, row)
            keys = {content_hash(k) for k in dataset_identity_keys((member,))}
            duplicates = used_keys & keys
            if not duplicates:
                fresh[index] = row
                used_keys.update(keys)
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
    return replacements


def load_member(output: Path, row: PopulationRow) -> SnapshotMember:
    path = output / row["partition"] / "members" / f"{row['seed']}.json"
    return SnapshotMember.model_validate_json(path.read_bytes())


def close_dataset(
    output: Path,
    source: Path,
    rows: list[PopulationRow],
    original: DatasetManifest,
    design: NativeRound = MIXED,
):
    members = tuple(load_member(output, row) for row in rows)
    retained = tuple(m for m in members if m.partition == "train")
    if tuple(m for m in original.members if m.partition == "train") != retained:
        raise ValueError("Training members differ from the previous native population")
    original_refs = ReferenceSet.model_validate_json(
        (source / "train/references.json").read_bytes()
    )
    validate_reference_membership(original, original_refs)
    dataset = seal_dataset(design.dataset_id, members)
    manifest = dataset.model_dump(mode="json")
    publish(output / "dataset.json", manifest)
    references = {}
    locations = {}
    for role in ("train", "validation", "test"):
        folder = output / role
        selected = [m for m in members if m.partition == role]
        paths = {m.source_namespace: f"exports/{m.split_group.seed}.json" for m in selected}
        publish(folder / "dataset.json", manifest)
        publish(folder / "locations.json", paths)
        locations[role] = content_hash(paths)
        if role != "test":
            refs = tuple(
                ReferenceMember.model_validate_json(
                    (folder / "reference-members" / f"{m.split_group.seed}.json").read_bytes()
                )
                for m in selected
            )
            if role == "train" and refs != original_refs.members:
                raise ValueError("Retained training targets differ from the earlier references")
            ref = seal_reference_set(
                ReferencePayload(
                    reference_set_id=f"{design.reference_prefix}-{role}",
                    dataset_sha256=dataset.content_sha256,
                    partition=role,
                    kind="automatic",
                    policy=DIRECTION_POLICY,
                    analysis_profile=PROFILE,
                    semantics=DirectionSemantics(
                        profile_sha256=analysis_profile_hash(PROFILE),
                        decoder_version=DECODER_VERSION,
                    ),
                    members=refs,
                )
            )
            validate_reference_membership(dataset, ref)
            publish(folder / "references.json", ref.model_dump(mode="json"))
            references[role] = ref.content_sha256
        publish(folder / "reservation.json", {"dataset_sha256": dataset.content_sha256})
    publish(
        output / "reservation.json",
        {
            "dataset_sha256": dataset.content_sha256,
            "prior_sha256": digest(output / "prior.json"),
            "population_sha256": digest(output / "population.json"),
            "references_sha256": references,
            "locations_sha256": locations,
        },
    )


def prepare(
    source: Path,
    output: Path,
    workers: int,
    design: NativeRound,
    extra_keys: frozenset[str] = frozenset(),
) -> None:
    original = source_manifest(source, design)
    training, fresh, prior, plan = prepare_rows(output, source, original, design, extra_keys)
    print(
        f"Retaining {len(training)} training and generating {len(fresh)} fresh markets", flush=True
    )
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        submitted = pool.map(partial(generate, output=output), fresh)
        copy_training(source, output, original)
        for index, _ in enumerate(submitted, 1):
            if index % 96 == 0:
                print(f"Completed {index}/{len(fresh)} fresh markets", flush=True)
    replacements = accept_fresh(output, fresh, prior, design)
    rows = [*training, *fresh]
    publish(
        output / "population.json",
        {
            **plan,
            "candidate_plan_sha256": digest(output / "candidate-plan.json"),
            "rows": rows,
            "replacements": replacements,
        },
    )
    for replacement in replacements:
        remove_duplicate(output, replacement["excluded"])
    close_dataset(output, source, rows, original, design)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source", type=Path, default=Path("var/verification/direction-large-v1"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    prepare(args.source, args.output, args.workers, MIXED)
