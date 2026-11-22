"""Time warm single-model and three-model predictions on the same validation sequences."""

import argparse
import json
import time
from pathlib import Path
from statistics import median

import numpy as np
from compare_selection_panels import ENSEMBLE, checked, experiment_id, open_corpus, panel_ids
from compare_tcn_widths import PROFILE
from compare_temporal_architectures import RUNTIME, evaluator

from regimex.data_catalog import _load_partition
from regimex.storage import json_bytes, write_new_file


def benchmark(output: Path, source: Path):
    receipt, models = checked(output, source)
    corpus = open_corpus(output, receipt, "validation")
    selected = set(panel_ids(corpus.dataset.members, (4, 4, 4)))
    samples = []
    for partition in ("train", "validation", "test"):
        refs, labels = corpus.references(partition, profile=PROFILE)
        rows = _load_partition(
            corpus.dataset,
            refs,
            snapshot_reader=corpus.snapshot,
            reference_reader=lambda _, member, labels=labels: labels[member.source_namespace],
        )
        samples.extend(sample for sample in rows if experiment_id(sample) in selected)
    validation = evaluator(tuple(samples))

    def measure(names):
        started = time.perf_counter()
        probabilities = [models[name].predict(validation.inputs, runtime=RUNTIME) for name in names]
        p = np.mean(probabilities, axis=0) if len(names) > 1 else probabilities[0]
        predicted = time.perf_counter()
        validation._decode(p)
        return predicted - started, time.perf_counter() - started

    recipes = {"single": ("short-96",), "ensemble": ENSEMBLE}
    for names in recipes.values():
        measure(names)
    timings = {name: [] for name in recipes}
    repeats = 8
    for iteration in range(repeats):
        order = ("single", "ensemble") if iteration % 2 == 0 else ("ensemble", "single")
        for name in order:
            timings[name].append(measure(recipes[name]))

    return {
        "validation_dataset_sha256": corpus.dataset.content_sha256,
        "experiment_ids": sorted(selected),
        "experiments": len(samples),
        "instrument_sequences": len(validation.inputs.info),
        "instrument_steps": validation.inputs.row_count,
        "runtime": RUNTIME.model_dump(mode="json"),
        "repeats": repeats,
        "timings": {
            name: {
                "predict_seconds": [p for p, _ in measured],
                "predict_and_decode_seconds": [d for _, d in measured],
                "median_predict_seconds": median(p for p, _ in measured),
                "median_predict_and_decode_seconds": median(d for _, d in measured),
            }
            for name, measured in timings.items()
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("panels", type=Path, help="Reserved selection-panel comparison.")
    parser.add_argument(
        "--source", type=Path, required=True, help="Previous training-population comparison."
    )
    parser.add_argument("--output", type=Path, help="New file for the individual timings.")
    args = parser.parse_args()
    result = benchmark(args.panels, args.source)
    if args.output is not None:
        write_new_file(args.output, json_bytes(result))
    print(json.dumps(result["timings"], indent=2))
