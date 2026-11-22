"""Inspect the TCN's raw sideways probabilities inside the known transfer-test pauses."""

import argparse
import json
from itertools import groupby
from pathlib import Path
from statistics import median

from check_pause_target_transfer import METHODS, RUNTIME, STRESS, examples
from fit_pause_targets import load_predictors

from regimex.directions import DIRECTION_NAMES
from regimex.storage import digest, json_bytes, write_new_file

PAUSE_SHAPES = ("plateau-32", "plateau-48", "pause-before-reversal")


def inspect(output, populations, training):
    recorded = json.loads((output / "results.json").read_bytes())
    if recorded["training_freeze_sha256"] != digest(training / "fitted" / "freeze.json"):
        raise ValueError("The supplied training freeze differs from the scored comparison")
    models = load_predictors(training, populations)
    sideways = DIRECTION_NAMES.index("sideways")
    result = {name: {} for name in METHODS}
    for shape in PAUSE_SHAPES:
        data = examples(shape, STRESS.index(shape))
        for name, model in models.items():
            probabilities = model.predict(data.inputs, runtime=RUNTIME)
            offset = 0
            runs, means, winning_fractions = [], [], []
            for series, refs in zip(data.inputs.info, data.references, strict=True):
                (pause,) = [ref for ref in refs if ref.direction == "sideways"]
                inside = probabilities[offset + pause.start : offset + pause.stop]
                wins = inside.argmax(axis=1) == sideways
                runs.append(
                    max(
                        (sum(1 for _ in group) for wins_here, group in groupby(wins) if wins_here),
                        default=0,
                    )
                )
                winning_fractions.append(float(wins.mean()))
                means.append(float(inside[:, sideways].mean()))
                offset += series.step_count
            assert offset == len(probabilities)
            result[name][shape] = {
                "pauses": len(runs),
                "no_sideways_winning_step": runs.count(0),
                "majority_sideways": sum(f > 0.5 for f in winning_fractions),
                "at_least_32_consecutive_winning_steps": sum(run >= 32 for run in runs),
                "median_mean_sideways_probability": median(means),
            }
    write_new_file(
        output / "probabilities.json",
        json_bytes({"transfer_sha256": digest(output / "results.json"), "results": result}),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Completed transfer comparison")
    parser.add_argument("--populations", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    args = parser.parse_args()
    inspect(args.output, args.populations, args.training)
