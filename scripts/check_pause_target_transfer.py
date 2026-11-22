"""Measure transfer from exchange training runs to price paths with known direction changes."""

import argparse
import json
from pathlib import Path

import numpy as np
from compare_profile_21 import RUNTIME
from compare_temporal_architectures import experiment_counts, summary
from confirm_trend_pauses import CONFIG, INSTRUMENT, STRESS, stress_case

from regimex.application.training import SharedSelectionEvaluator
from regimex.exports import ExportInstrument, ExportObservation, InstrumentUnits, ModelTimeGrid
from regimex.model_contracts import ObservationInputs, PredictionSequenceInfo
from regimex.storage import digest, json_bytes, write_new_file

SEED = 125516482
REPEATS = 8
METHODS = ("original", "refined")
NOISE = (0.0005, 0.0014)


def protocol():
    return {
        "seed": SEED,
        "replicates_per_noise_and_sign": REPEATS,
        "families": list(STRESS),
        "log_noise_sd": list(NOISE),
        "models": list(METHODS),
        "metrics": [
            "interval_macro_f1",
            "short_sideways_recall",
            "sideways_precision",
            "point_macro_f1",
        ],
        "inputs": "One execution per step, unobserved orders and quotes, cent-denominated closes.",
        "scores_against": "Designed latent directions and full intervals.",
        "implementation": {
            path.name: digest(path)
            for path in (
                Path(__file__),
                Path(__file__).with_name("confirm_trend_pauses.py"),
                Path(__file__).with_name("compare_temporal_architectures.py"),
            )
        },
    }


def examples(family, index):
    instrument = ExportInstrument(
        **INSTRUMENT.model_dump(),
        units=InstrumentUnits(
            price_unit="quote_asset_per_base_asset",
            volume_unit="base_asset",
            price_decimal_places=2,
            volume_decimal_places=0,
        ),
    )
    info, series, grids, references = [], [], [], []
    for noise_index, sigma in enumerate(NOISE):
        for sign in (-1, 1):
            for r in range(REPEATS):
                seed = [SEED, index, noise_index, sign + 1, r]
                grid, expected = stress_case(family, sign, sigma, np.random.default_rng(seed))
                info.append(
                    PredictionSequenceInfo(
                        experiment_id=f"path-{len(info)}",
                        instrument_id=instrument.instrument_id,
                        step_count=grid.step_count,
                    )
                )
                rows = tuple(
                    ExportObservation(
                        **row.model_dump(exclude={"last_price"}),
                        last_price=round(row.last_price, 2),
                        price_state="observed",
                        bid_state="unknown",
                        ask_state="unknown",
                    )
                    for row in grid.observations
                )
                series.append(rows)
                grids.append(
                    ModelTimeGrid(
                        origin=0,
                        unit="step",
                        bin_width=1,
                        completed_horizon=grid.step_count,
                        step_count=grid.step_count,
                        interval_bounds="[start,stop)",
                        terminal_bin="full_bins_only",
                    )
                )
                references.append(expected)
    inputs = ObservationInputs(tuple(info), tuple(series), tuple(grids), (instrument,) * len(info))
    return SharedSelectionEvaluator(inputs, tuple(references), CONFIG, "automatic")


def score(output, populations, training):
    from fit_pause_targets import load_predictors

    models = load_predictors(training, populations)
    groups = {name: {} for name in METHODS}
    for index, family in enumerate(STRESS):
        evaluation = examples(family, index)
        for name, model in models.items():
            probabilities = model.predict(evaluation.inputs, runtime=RUNTIME)
            _, _, counts = experiment_counts(evaluation, probabilities)
            groups[name][family] = counts
    results = {}
    for name, counts in groups.items():
        results[name] = {
            "pooled": summary(np.concatenate(list(counts.values()))),
            "by_shape": {family: summary(rows) for family, rows in counts.items()},
            "path_counts": {family: rows.tolist() for family, rows in counts.items()},
        }
    write_new_file(
        output / "results.json",
        json_bytes(
            {
                "plan_sha256": digest(output / "plan.json"),
                "results": results,
                "paths": len(STRESS) * REPEATS * 4,
                "training_freeze_sha256": digest(training / "fitted" / "freeze.json"),
            }
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    sub.add_parser("reserve").add_argument("output", type=Path)
    scoring = sub.add_parser("score")
    scoring.add_argument("output", type=Path)
    scoring.add_argument("--populations", type=Path, required=True)
    scoring.add_argument("--training", type=Path, required=True)
    args = parser.parse_args()
    plan = args.output / "plan.json"
    if args.phase == "reserve":
        write_new_file(plan, json_bytes(protocol()))
    elif json.loads(plan.read_bytes()) != protocol():
        raise ValueError("The implementation or design changed after reservation")
    else:
        score(args.output, args.populations, args.training)
