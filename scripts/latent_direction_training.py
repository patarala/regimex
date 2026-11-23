"""Generate training paths with known drift and independently sampled observation noise."""

from itertools import pairwise
from math import exp
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from regimex.contracts import Observation
from regimex.directions import AnalysisConfig, observation_blocks
from regimex.storage import digest

FIRST_SEED = 2_651_019_000
PATHS = 1152
LENGTHS = (256, 512, 1024)
SOURCE_ID = "latent-direction-train-v1"


def recipe():
    return {
        "source_id": SOURCE_ID,
        "first_seed": FIRST_SEED,
        "paths": PATHS,
        "lengths": LENGTHS,
        "rng": "PCG64",
        "labels": "Sign of the latent drift before innovation noise, quote rounding, execution "
        "sampling and missingness. Ineligible blocks and clipped sub-minimum segments are ignored.",
        "observations": "Piecewise drift with varying speed, level shifts, mean-reverting and "
        "integrated residuals. Execution sampling, tick rounding and missing spans follow.",
        "source_sha256": digest(Path(__file__)),
    }


def observations(prices: NDArray[np.float64], counts: NDArray[np.int64]) -> tuple[Observation, ...]:
    return tuple(
        Observation(
            instrument_id="EUR/USD",
            step=i,
            last_price=float(price) if np.isfinite(price) else None,
            trade_count=int(count) if count >= 0 else None,
            trade_volume=None,
            buy_order_count=None,
            buy_order_volume=None,
            sell_order_count=None,
            sell_order_volume=None,
            best_bid=None,
            best_ask=None,
        )
        for i, (price, count) in enumerate(zip(prices, counts, strict=True))
    )


def generate(seed: int, steps: int, profile: AnalysisConfig):
    rng = np.random.Generator(np.random.PCG64(seed))
    minimum = profile.min_duration
    periods = int(rng.integers(1, min(10, steps // (minimum + 8)) + 1))
    sizes = minimum + rng.multinomial(
        steps - minimum * periods, rng.dirichlet(np.full(periods, 1.25))
    )
    drift = np.zeros(steps)
    labels = np.zeros(steps, dtype=np.int64)
    boundaries = np.r_[0, np.cumsum(sizes)]
    direction = int(rng.choice([-1, 0, 1]))
    speed = exp(rng.uniform(np.log(0.00010), np.log(0.00095)))
    for i, (a, b) in enumerate(pairwise(boundaries)):
        if i:
            if direction:
                direction = 0 if rng.random() < 0.52 else -direction
            else:
                direction = int(rng.choice([-1, 1]))
        labels[a:b] = {0: 0, 1: 1, -1: 2}[direction]
        if direction:
            movement = speed * exp(rng.uniform(-0.8, 0.8))
            drift[a:b] = direction * movement
            if b - a >= minimum + 8 and rng.random() < 0.65:
                # A change in speed within a trend leaves the sign of its drift intact.
                length = int(rng.integers(minimum // 2, b - a))
                left = int(rng.integers(a, b - length + 1))
                drift[left : left + length] *= exp(rng.uniform(np.log(0.16), np.log(2.4)))

    noise = exp(rng.uniform(np.log(0.000025), np.log(0.0012)))
    memory = rng.uniform(0, 0.92)
    residual = rng.normal(0, noise * np.sqrt(1 - memory**2), steps)
    for i in range(1, steps):
        residual[i] += memory * residual[i - 1]
    path = np.cumsum(drift + rng.normal(0, noise * rng.uniform(0.02, 0.22), steps)) + residual
    for _ in range(rng.poisson(steps / 420)):
        k = int(rng.integers(1, steps))
        path[k:] += rng.choice([-1, 1]) * exp(rng.uniform(np.log(0.001), np.log(0.025)))
    initial = exp(rng.uniform(np.log(0.1), np.log(2500)))
    tick = initial * exp(rng.uniform(np.log(0.000003), np.log(0.0005)))
    quoted = np.maximum(tick, np.round(initial * np.exp(path) / tick) * tick)
    counts = (rng.random(steps) < rng.uniform(0.42, 1.0)).astype(np.int64)
    counts[0] = 1
    prices = quoted[np.maximum.accumulate(np.where(counts, np.arange(steps), 0))]
    hidden = rng.random(steps) < rng.uniform(0, 0.022)
    for _ in range(rng.poisson(steps / 1100)):
        width = int(rng.integers(8, min(96, steps // 3) + 1))
        k = int(rng.integers(0, steps - width + 1))
        hidden[k : k + width] = True
    prices[hidden] = np.nan
    counts[hidden] = -1
    rows = observations(prices, counts)
    for left, right, status in observation_blocks(rows, profile):
        if status != "observed":
            labels[left:right] = -1
        else:
            knots = np.r_[left, boundaries[(boundaries > left) & (boundaries < right)], right]
            for a, b in pairwise(knots):
                if b - a < min(minimum, right - left):
                    labels[a:b] = -1
    return prices, counts, rows, labels
