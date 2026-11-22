"""Check annotation split penalties against independently specified piecewise price paths."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from regimex.contracts import Instrument, Observation
from regimex.directions import AnalysisConfig, Direction, DirectionInterval, summarize_interval
from regimex.evaluation import _metrics, _segment_counts
from regimex.segmentation import annotate
from regimex.storage import write_new_file

STEPS = 192
INSTRUMENT = Instrument(
    instrument_id="EUR/USD", base_asset="EUR", quote_asset="USD", initial_price=100
)
PENALTIES = (4, 8, 16)
FAMILIES = ("flat", "level-shift", "slow-trend", "reversal", "plateau-32", "plateau-48")
NOISE = (0.00035, 0.0015)
REPEATS = 12


@dataclass(frozen=True)
class PathGrid:
    experiment_id: str
    observations: tuple[Observation, ...]
    instruments: tuple[Instrument, ...] = (INSTRUMENT,)
    step_count: int = STEPS


def case(family: str, sign: int, sigma: float, rng: np.random.Generator):
    rising = Direction.GROWTH if sign == 1 else Direction.DECLINE
    falling = Direction.DECLINE if sign == 1 else Direction.GROWTH
    middle = STEPS // 2 + int(rng.integers(-12, 13))
    slope = np.zeros(STEPS)
    level = np.zeros(STEPS)
    if family in {"flat", "level-shift"}:
        blocks = [(0, STEPS, Direction.SIDEWAYS)]
        if family == "level-shift":
            level[middle:] = sign * 0.012
    elif family == "slow-trend":
        blocks = [(0, STEPS, rising)]
        slope[:] = sign * np.log1p(0.012) / (STEPS - 1)
    elif family == "reversal":
        blocks = [(0, middle, rising), (middle, STEPS, falling)]
        slope[:middle], slope[middle:] = sign * 0.00016, -sign * 0.00016
    else:
        length = int(family.removeprefix("plateau-"))
        start, stop = middle - length // 2, middle + length // 2
        blocks = [(0, start, rising), (start, stop, Direction.SIDEWAYS), (stop, STEPS, rising)]
        slope[:start] = slope[stop:] = sign * 0.00016

    # The latent slope between consecutive steps is zero throughout the plateau.
    latent = np.r_[0, np.cumsum(slope[1:])] + level
    noise = rng.normal(size=STEPS)
    for i in range(1, STEPS):
        noise[i] = 0.7 * noise[i - 1] + np.sqrt(1 - 0.7**2) * noise[i]
    prices = np.round(100 * np.exp(latent + sigma * noise), 2)
    rows = tuple(
        Observation(
            instrument_id=INSTRUMENT.instrument_id,
            step=i,
            last_price=price,
            trade_count=1,
            trade_volume=1,
            buy_order_count=None,
            buy_order_volume=None,
            sell_order_count=None,
            sell_order_volume=None,
            best_bid=None,
            best_ask=None,
        )
        for i, price in enumerate(prices)
    )
    reference = tuple(
        DirectionInterval(
            instrument_id=INSTRUMENT.instrument_id,
            start=start,
            stop=stop,
            status="observed",
            direction=direction,
            statistics=summarize_interval(rows, start, stop),
        )
        for start, stop, direction in blocks
    )
    return PathGrid(family, rows), reference


def check():
    totals = {
        penalty: {f: np.zeros((3, 3), dtype=np.int64) for f in FAMILIES} for penalty in PENALTIES
    }
    rng = np.random.default_rng(481917)
    for family in FAMILIES:
        for sigma in NOISE:
            for sign in (-1, 1):
                for _ in range(REPEATS):
                    grid, expected = case(family, sign, sigma, rng)
                    for penalty in PENALTIES:
                        actual = annotate(
                            grid, AnalysisConfig(version="2.2", boundary_penalty=penalty)
                        )
                        totals[penalty][family] += _segment_counts(expected, actual.intervals)
    return {
        "profile": "2.2",
        "steps": STEPS,
        "stationary_log_noise_sd": list(NOISE),
        "seed": 481917,
        "experiments_per_family": REPEATS * len(NOISE) * 2,
        "class_order": [x.value for x in Direction],
        "results": {
            penalty: {
                "interval_macro_f1": _metrics(sum(families.values()))["macro_f1"],
                "per_family_tp_fp_fn": {name: values.tolist() for name, values in families.items()},
            }
            for penalty, families in totals.items()
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="New file for the pooled and per-family counts")
    args = parser.parse_args()
    result = check()
    write_new_file(args.output, (json.dumps(result, indent=2) + "\n").encode())
    for penalty, row in result["results"].items():
        print(f"Split penalty {penalty}: interval macro-F1 {row['interval_macro_f1']:.4f}")
