"""Time warm model prediction, interval decoding and the optional price-based second pass."""

import argparse
import json
from itertools import permutations
from pathlib import Path
from statistics import median
from time import perf_counter

from compare_profile_21 import RUNTIME, sha256
from compare_selection_panels import checked, experiment_id, open_corpus, panel_ids
from compare_tcn_widths import PROFILE
from fit_local_pause_features import load_predictors
from refine_model_intervals import refine_decoded

from regimex.application.analysis_computation import snapshot_inputs
from regimex.model_contracts import ObservationInputs
from regimex.segmentation import decode
from regimex.storage import json_bytes, write_new_file

METHODS = ("control", "candidate", "hybrid")


def validation_inputs(panels, populations):
    receipt, _ = checked(panels, populations)
    corpus = open_corpus(panels, receipt, "validation")
    selected = set(panel_ids(corpus.dataset.members, (4, 4, 4)))
    snapshots = []
    members = [member for member in corpus.dataset.members if experiment_id(member) in selected]
    # Keep the instrument batch order used in the ensemble timing comparison.
    partitions = ("train", "validation", "test")
    for member in sorted(members, key=lambda m: partitions.index(m.partition)):
        snapshot = corpus.snapshot(member)
        snapshots.append(snapshot_inputs(snapshot.export))
    inputs = ObservationInputs(
        tuple(item for s in snapshots for item in s.info),
        tuple(rows for s in snapshots for rows in s.observations),
        tuple(grid for s in snapshots for grid in s.time_grids),
        tuple(instrument for s in snapshots for instrument in s.instruments),
    )
    return inputs, sorted(selected), corpus.dataset.content_sha256


def measure(inputs, model, *, hybrid):
    started = perf_counter()
    probabilities = model.predict(inputs, runtime=RUNTIME)
    predicted = perf_counter()
    offset = 0
    for rows in inputs.observations:
        stop = offset + len(rows)
        intervals = decode(probabilities[offset:stop], rows, PROFILE)
        if hybrid:
            refine_decoded(rows, intervals, PROFILE)
        offset = stop
    return predicted - started, perf_counter() - started


def benchmark(output, panels, populations, previous):
    inputs, selected, dataset_sha256 = validation_inputs(panels, populations)
    models = load_predictors(output, populations, previous)
    models["hybrid"] = models["control"]
    for name in METHODS:
        measure(inputs, models[name], hybrid=name == "hybrid")
    timings = {name: [] for name in METHODS}
    orders = list(permutations(METHODS))
    for order in orders:
        for name in order:
            timings[name].append(measure(inputs, models[name], hybrid=name == "hybrid"))
    return {
        "validation_dataset_sha256": dataset_sha256,
        "freeze_sha256": sha256(output / "fitted" / "freeze.json"),
        "validation_experiment_ids": selected,
        "experiments": len(selected),
        "instrument_sequences": len(inputs.info),
        "instrument_steps": inputs.row_count,
        "runtime": RUNTIME.model_dump(mode="json"),
        "warmups_per_method": 1,
        "orders": orders,
        "timings": {
            name: {
                "predict_seconds": [p for p, _ in times],
                "end_to_end_seconds": [t for _, t in times],
                "median_predict_seconds": median(p for p, _ in times),
                "median_end_to_end_seconds": median(t for _, t in times),
            }
            for name, times in timings.items()
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparison", type=Path, help="Completed local pause feature comparison")
    parser.add_argument("--panels", type=Path, required=True)
    parser.add_argument("--populations", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True, help="Pause-target training run")
    parser.add_argument("--output", type=Path, help="New file for the individual timings")
    args = parser.parse_args()
    result = benchmark(args.comparison, args.panels, args.populations, args.previous)
    if args.output is not None:
        write_new_file(args.output, json_bytes(result))
    print(json.dumps(result["timings"], indent=2))
