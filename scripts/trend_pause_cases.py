"""Paired price paths observed on execution schedules from the native exchange."""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from regimex.directions import AnalysisConfig, Direction, DirectionInterval, observation_blocks
from regimex.exports import ExportObservation
from regimex.market import simulate
from regimex.market_config import MarketConfig, ParticipantProfile
from regimex.segmentation import _interval

STEPS = 512
CONFIG = AnalysisConfig(version="2.2")
MARKETS = {"dense": 0.35, "sparse": 0.04}
SHAPES = ("steady", "one-slowdown", "three-slowdowns", "one-pause", "three-pauses")


class Bounds(NamedTuple):
    start: int
    stop: int


@dataclass(frozen=True)
class Case:
    market: str
    shape: str
    severity: str
    rows: tuple[ExportObservation, ...]
    parents: tuple[DirectionInterval, ...]
    expected: tuple[Bounds, ...]
    tick: float
    active_steps: int


def native_schedule(seed, probability, tick):
    profile = ParticipantProfile(
        count=12,
        buy_probability=probability,
        sell_probability=probability,
        hold_probability=1 - 2 * probability,
        price_offset_bps=(-25.0, 25.0),
    )
    market = MarketConfig(
        tick_size=tick,
        profiles=(profile,),
        test_participant=profile.model_copy(update={"count": 1}),
    )
    return simulate(market, seed, STEPS).observations


def observe(schedule, prices):
    last = 100.0
    rows = []
    for row, price in zip(schedule, prices, strict=True):
        active = bool(row.trade_count)
        if active:
            last = float(price)
        rows.append(
            ExportObservation(
                **row.model_dump(exclude={"last_price", "best_bid", "best_ask"}),
                last_price=last,
                price_state="observed" if active else "carried",
                best_bid=None,
                best_ask=None,
                bid_state="unknown",
                ask_state="unknown",
            )
        )
    return tuple(rows)


def cases(group_seed, index):
    rng = np.random.default_rng(np.random.SeedSequence([group_seed, 9041]))
    sign = (-1, 1)[index % 2]
    direction = Direction.GROWTH if sign == 1 else Direction.DECLINE
    tick = (0.01, 0.1, 0.25)[index % 3]
    noise_sd = (0.00035, 0.0014)[index // 2 % 2]
    noise = noise_sd * rng.standard_t(5, STEPS) * np.sqrt(3 / 5)
    for step in range(1, STEPS):
        noise[step] = 0.60 * noise[step - 1] + 0.80 * noise[step]
    slope = sign * rng.uniform(0.00022, 0.00055)
    severity = ("near-flat", "moderate")[index // 4 % 2]
    fraction = rng.uniform(0.08, 0.22) if severity == "near-flat" else rng.uniform(0.35, 0.65)
    bounds = []
    for center in (105, 250, 398):
        width = int(rng.choice((32, 40, 48, 56)))
        start = center + int(rng.integers(-10, 11)) - width // 2
        bounds.append(Bounds(start, start + width))
    for market_index, (market, probability) in enumerate(MARKETS.items()):
        schedule = native_schedule(group_seed + market_index, probability, tick)
        active_steps = sum(bool(row.trade_count) for row in schedule)
        for shape in SHAPES:
            windows = (bounds[1],) if shape.startswith("one") else tuple(bounds)
            rates = np.full(STEPS, slope)
            rates[0] = 0
            if shape != "steady":
                for start, stop in windows:
                    # The first price in a pause is its level; subsequent increments are zero.
                    rates[start + 1 : stop] *= 0 if "pause" in shape else fraction
            prices = tick * np.rint(100 * np.exp(np.cumsum(rates) + noise) / tick)
            rows = observe(schedule, prices)
            parents = tuple(
                _interval(rows, start, stop, direction if status == "observed" else None, status)
                for start, stop, status in observation_blocks(rows, CONFIG)
            )
            yield Case(
                market,
                shape,
                severity,
                rows,
                parents,
                windows if "pause" in shape else (),
                tick,
                active_steps,
            )
