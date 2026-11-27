"""Measure changes to frozen direction-32/2.1 predictions on paired incomplete exports."""

import argparse
import hashlib
import json
import platform
from importlib.metadata import version
from itertools import batched
from pathlib import Path

import numpy as np

from regimex.admin_contracts import GenerationSpec, RawGenerationPlan
from regimex.application.admin_computation import _generation
from regimex.application.analysis_computation import snapshot_inputs
from regimex.backends.baseline import BaselineBackend
from regimex.backends.tcn import TCNBackend
from regimex.benchmark_contracts import dataset_identity_keys
from regimex.data_catalog import (
    DatasetManifest,
    SnapshotGrid,
    SourceSnapshot,
    content_hash,
    create_snapshot_member,
)
from regimex.directions import DIRECTION_NAMES, AnalysisConfig
from regimex.export_robustness import (
    INSUFFICIENT,
    NO_TRADES,
    SCENARIOS,
    WINDOW,
    masked_export,
    paired_metrics,
    statuses,
)
from regimex.exports import load_export_v2
from regimex.model_registry import InferenceRuntime
from regimex.segmentation import annotate, decode
from regimex.storage import json_bytes

DATASET_SHA256 = "a8f58746e20f3a55809233e0c18671aaf700e257b489b470e3966d68c3657d79"
GENERATION = GenerationSpec(
    dataset_id="incomplete-exports-21",
    plan=RawGenerationPlan(seed=318751406, experiment_count=48, step_count=512),
)
PROFILE = AnalysisConfig(version="2.1")
RUNTIME = InferenceRuntime(threads=2, max_batch_size=8)


def prepare_masked_members(paths, members, *, scenarios=SCENARIOS):
    variants = []
    for path, member in zip(paths, members, strict=True):
        pristine = member.bind_export(load_export_v2(path)).export
        for scenario in scenarios:
            export = masked_export(pristine, scenario)
            source = SourceSnapshot(
                source_namespace=member.source_namespace,
                split_group=member.split_group,
                export=export,
            )
            variants.append(create_snapshot_member(source, member.partition))
            directory = path.parent / scenario
            directory.mkdir(exist_ok=True)
            (directory / path.name).write_bytes(json_bytes(export.model_dump(mode="json")))
    return tuple(variants)


def registered_views(paths, members, variants, *, scenarios=SCENARIOS):
    """Load the reserved pristine export and its masked counterparts for scoring."""
    batches = batched(variants, len(scenarios), strict=True)
    for path, member, group in zip(paths, members, batches, strict=True):
        pristine = member.bind_export(load_export_v2(path)).export
        masked = {
            scenario: variant.bind_export(load_export_v2(path.parent / scenario / path.name)).export
            for scenario, variant in zip(scenarios, group, strict=True)
        }
        yield pristine, masked


def prepare(output, bundles, history):
    index_bytes = history.read_bytes() if history else None
    freeze = json.loads((bundles / "freeze.json").read_bytes())
    protocol = {
        "generation": GENERATION.model_dump(mode="json"),
        "profile": PROFILE.model_dump(mode="json"),
        "runtime": RUNTIME.model_dump(mode="json"),
        "bundles": freeze["bundles"],
        "window": WINDOW,
        "scenarios": SCENARIOS,
        "reference": "Automatic direction-32/2.1 intervals on each pristine instrument",
        "metrics": [
            "Common reference-eligible direction coverage: point macro-F1 and changed directions",
            "Original grid: lost and gained coverage, no_trades and insufficient_data",
            "Window: statuses on all steps, originally zero executions, and pristine no_trades",
        ],
        "history_sha256": hashlib.sha256(index_bytes).hexdigest() if index_bytes else None,
        "versions": {
            "python": platform.python_version(),
            **{name: version(name) for name in ("numpy", "scikit-learn", "torch")},
        },
    }
    output.mkdir(parents=True)
    (output / "protocol.json").write_bytes(json_bytes(protocol))
    generated = _generation(GENERATION, output, "incomplete-exports")
    dataset = DatasetManifest.for_modeling(json_bytes(generated["manifest"]))
    if dataset.content_sha256 != DATASET_SHA256:
        raise ValueError("Generated data differ from the recorded population")
    paths = [output / item["path"] for item in generated["snapshots"]]
    variants = prepare_masked_members(paths, dataset.members)
    (output / "masked-members.json").write_bytes(
        json_bytes([member.model_dump(mode="json") for member in variants])
    )
    seeds = {member.split_group.seed for member in dataset.members}
    identities = {content_hash(key) for key in dataset_identity_keys(dataset.members)}
    all_identities = identities | {content_hash(key) for key in dataset_identity_keys(variants)}
    receipt = {
        "pristine_keys": len(identities),
        "masked_exports": len(variants),
        "dataset_sha256": dataset.content_sha256,
        "new_seeds": sorted(seeds),
        "new_identity_keys_sha256": sorted(all_identities),
    }
    if index_bytes is not None:
        index = json.loads(index_bytes)
        if seeds.intersection(index["excluded_seeds"]) or all_identities.intersection(
            index["identity_keys_sha256"]
        ):
            raise ValueError("Generated experiments overlap the exposure index; scoring stopped")
        successor = {
            **index,
            "previous_sha256": protocol["history_sha256"],
            "scope": "Prior exposures and the pristine and masked incomplete-export snapshots.",
            "current_dataset_paths": ["dataset.json"],
            "excluded_seeds": sorted(set(index["excluded_seeds"]).union(seeds)),
            "identity_keys_sha256": sorted(
                set(index["identity_keys_sha256"]).union(all_identities)
            ),
        }
        (output / "exposure-index.json").write_bytes(json_bytes(successor))
        receipt.update(
            prior_seeds=len(index["excluded_seeds"]),
            prior_identity_keys=len(index["identity_keys_sha256"]),
        )
    (output / "dataset.json").write_bytes(json_bytes(generated["manifest"]))
    (output / "reservation.json").write_bytes(json_bytes(receipt))
    return freeze["bundles"], registered_views(paths, dataset.members, variants)


def dense_states(intervals, steps):
    states = np.full(steps, -1, dtype=np.int64)
    for item in intervals:
        if item.direction is None:
            state = NO_TRADES if item.status == "no_trades" else INSUFFICIENT
        else:
            state = DIRECTION_NAMES.index(item.direction.value)
        states[item.start : item.stop] = state
    if (states < 0).any():
        raise ValueError("Decoded intervals do not cover the original grid")
    return states


def predictions(model, inputs):
    probabilities = model.predict(inputs, runtime=RUNTIME)
    start = 0
    states = []
    for rows in inputs.observations:
        stop = start + len(rows)
        intervals = decode(probabilities[start:stop], rows, model.analysis)
        states.append(dense_states(intervals, len(rows)))
        start = stop
    return np.concatenate(states)


def evaluate(views, models):
    series = {name: {key: [] for key in ("pristine", *SCENARIOS)} for name in models}
    reference, windows, zeros = [], [], []
    instruments = 0
    experiments = 0
    for export, masked in views:
        experiments += 1
        inputs = snapshot_inputs(export)
        annotations = annotate(SnapshotGrid.from_export(export), PROFILE)
        for info, rows in zip(inputs.info, inputs.observations, strict=True):
            intervals = [i for i in annotations.intervals if i.instrument_id == info.instrument_id]
            reference.append(dense_states(intervals, len(rows)))
            windows.append([WINDOW[0] <= row.step < WINDOW[1] for row in rows])
            zeros.append([row.trade_count == 0 and row.trade_volume == 0 for row in rows])
            instruments += 1
        for name, model in models.items():
            series[name]["pristine"].append(predictions(model, inputs))
        for scenario, variant in masked.items():
            masked_inputs = snapshot_inputs(variant)
            for name, model in models.items():
                series[name][scenario].append(predictions(model, masked_inputs))
    target = np.concatenate(reference)
    window = np.concatenate(windows)
    zero = np.concatenate(zeros)
    result = {}
    for name, variants in series.items():
        pristine = np.concatenate(variants["pristine"])
        result[name] = {
            scenario: paired_metrics(
                target, pristine, np.concatenate(variants[scenario]), window, zero
            )
            for scenario in SCENARIOS
        }
    return {
        "experiments": experiments,
        "instrument_series": instruments,
        "models": result,
        "reference_status": statuses(target),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundles", type=Path, required=True, help="Comparison folder with freeze.json"
    )
    parser.add_argument(
        "--exposure-index",
        type=Path,
        help="Historical seed and identity hashes for an independent overlap audit",
    )
    parser.add_argument("--output", type=Path, required=True, help="New directory for run evidence")
    args = parser.parse_args()
    digests, views = prepare(args.output, args.bundles, args.exposure_index)
    models = {
        name: backend.load_bundle(
            args.bundles / name, manifest_sha256=digests[name], runtime=RUNTIME
        )
        for name, backend in (("baseline", BaselineBackend()), ("tcn", TCNBackend()))
    }
    if any(model.analysis != PROFILE for model in models.values()):
        raise ValueError("The bundles must use direction-32/2.1 with the frozen configuration")
    result = evaluate(views, models)
    (args.output / "results.json").write_bytes(json_bytes(result))
    print(args.output / "results.json")


if __name__ == "__main__":
    main()
