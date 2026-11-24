"""Score execution-aware direction labels against predetermined price paths."""

import argparse
from itertools import pairwise
from math import ceil
from pathlib import Path

import numpy as np
from check_execution_labels import METHODS, PROFILE, label_sequence
from confirm_trend_pauses import STRESS, stress_case
from latent_direction_training import observations

from regimex.directions import DIRECTION_NAMES, Direction, observation_blocks
from regimex.evaluation import _segment_counts
from regimex.segmentation import _interval
from regimex.storage import digest, json_bytes, write_new_file

LENGTHS = (256, 512, 1024)
SHAPES = ("steady", "reversal", "pause_then_reversal", "flat_level_shift")


def known_intervals(rows, directions):
    cuts = [0, *(1 + np.flatnonzero(np.diff(directions))).tolist(), len(rows)]
    intervals = []
    labels = {0: Direction.SIDEWAYS, 1: Direction.GROWTH, -1: Direction.DECLINE}
    for start, stop, status in observation_blocks(rows, PROFILE):
        if status != "observed":
            intervals.append(_interval(rows, start, stop, None, status))
        else:
            for left, right in pairwise(cuts):
                a, b = max(start, left), min(stop, right)
                if a < b:
                    intervals.append(_interval(rows, a, b, labels[directions[left]], status))
    return tuple(intervals)


def drift_paths(seed, index, size):
    """Share noise, quantization and execution times across the four latent shapes and gap views."""
    rng = np.random.default_rng(seed)
    scale = size / 1024
    width = PROFILE.min_duration
    gap_length = max(width, round(48 * scale))
    sign = (-1, 1)[index % 2]
    step = np.arange(size)
    noise = rng.standard_t(5, size) * (0.0005, 0.0015)[index // 4 % 2] * np.sqrt(3 / 5)
    for i in range(1, size):
        noise[i] = 0.6 * noise[i - 1] + 0.8 * noise[i]
    rate = rng.uniform(0.00014, 0.00035) * (1 + 0.35 * np.sin(step / rng.uniform(62, 112)))
    # All resulting intervals, including both sides of a masked tail, meet the profile's duration.
    left = rng.integers(
        ceil(390 * scale), min(ceil(525 * scale), size - gap_length - 3 * width + 1)
    )
    right = left + rng.integers(
        max(width, ceil(56 * scale)),
        min(ceil(144 * scale), size - gap_length - 2 * width - left + 1),
    )
    jump = sign * rng.uniform(0.009, 0.021)
    trades = rng.random(size) < (0.85, 0.22)[index // 2 % 2]
    gap = rng.integers(
        max(right + width, ceil(720 * scale)), min(ceil(841 * scale), size - gap_length - width + 1)
    )
    trades[::24] = True
    trades[[0, gap - 1, gap + gap_length]] = True
    tick = (0.01, 0.20)[index // 8 % 2]
    for shape in SHAPES:
        directions = np.full(size, sign, dtype=np.int8)
        offset = np.zeros(size)
        if shape == "reversal":
            directions[left:] = -sign
        elif shape == "pause_then_reversal":
            directions[left:right] = 0
            directions[right:] = -sign
        elif shape == "flat_level_shift":
            directions.fill(0)
            offset[left:] = jump
        quoted = np.round(
            tick * np.rint(100 * np.exp(np.cumsum(directions * rate) + offset + noise) / tick), 2
        )
        last = np.maximum.accumulate(np.where(trades, step, 0))
        prices = quoted[last]
        counts = trades.astype(np.int64)
        for view in ("complete", "masked"):
            if view == "masked":
                prices[gap : gap + gap_length] = np.nan
                counts[gap : gap + gap_length] = -1
            rows = observations(prices, counts)
            yield shape, view, rows, known_intervals(rows, directions)


def stress_paths(seed, index, *, crossed=False):
    for shape_index, shape in enumerate(STRESS):
        sign = (-1, 1)[index % 2]
        noise = (0.0005, 0.0014)[index // 2 % 2]
        rng = np.random.default_rng(np.random.SeedSequence([seed, shape_index]))
        grid, truth = stress_case(shape, sign, noise, rng)
        rows = grid.observations
        yield shape, "stress", rows, truth
        if crossed:
            # Keep the same execution times across shapes wherever their timelines overlap.
            uniforms = np.random.default_rng(np.random.SeedSequence([seed, 29])).random(len(rows))
            for view, probability in (("dense", 0.85), ("sparse", 0.25)):
                trades = uniforms < probability
                trades[::24] = True
                steps = np.arange(len(rows))
                last = np.maximum.accumulate(np.where(trades, steps, 0))
                observed = observations(
                    np.array([r.last_price for r in rows])[last], trades.astype(int)
                )
                directions = np.zeros(len(rows), dtype=np.int8)
                for interval in truth:
                    directions[interval.start : interval.stop] = {
                        Direction.GROWTH: 1,
                        Direction.SIDEWAYS: 0,
                        Direction.DECLINE: -1,
                    }[interval.direction]
                yield shape, view, observed, known_intervals(observed, directions)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("output", type=Path, help="New JSON file for group-level interval counts")
    cli.add_argument("--steps", choices=(*map(str, LENGTHS), "stress"), required=True)
    cli.add_argument("--groups", type=int, default=32)
    cli.add_argument("--first-seed", type=int, required=True)
    cli.add_argument("--methods", nargs="+", choices=METHODS, required=True)
    cli.add_argument(
        "--crossed",
        action="store_true",
        help="Also score dense and sparse executions on the same stress paths",
    )
    args = cli.parse_args()
    if args.crossed and args.steps != "stress":
        cli.error("--crossed applies to stress paths")
    if args.groups < 1 or args.first_seed < 0 or len(set(args.methods)) != len(args.methods):
        cli.error("Supply positive groups, a nonnegative seed and distinct methods")
    records = []
    for index in range(args.groups):
        seed = args.first_seed + index
        paths = (
            stress_paths(seed, index, crossed=args.crossed)
            if args.steps == "stress"
            else drift_paths(seed, index, int(args.steps))
        )
        for shape, view, rows, truth in paths:
            counts = {}
            for method in args.methods:
                actual = label_sequence(rows, method)
                counts[method] = _segment_counts(truth, actual).tolist()
            records.append(dict(seed=seed, shape=shape, view=view, counts=counts))
    files = (
        Path(__file__),
        Path(__file__).with_name("check_execution_labels.py"),
        Path(__file__).with_name("confirm_trend_pauses.py"),
        Path(__file__).with_name("latent_direction_training.py"),
    )
    write_new_file(
        args.output,
        json_bytes(
            dict(
                steps=args.steps,
                methods=args.methods,
                class_order=DIRECTION_NAMES,
                groups=args.groups,
                files={f.name: digest(f) for f in files},
                records=records,
            )
        ),
    )


if __name__ == "__main__":
    main()
