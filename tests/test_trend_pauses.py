"""A pause needs observed prices and sustained movement on both sides."""

import math
import sys
from collections.abc import Sequence

import pytest
from test_segmentation import experiment

from regimex.contracts import Observation
from regimex.directions import AnalysisConfig, Direction
from regimex.exports import ExportObservation
from regimex.segmentation import _interval
from regimex.trend_pauses import PauseScan, find_trend_pauses, scan_trend_pauses


def path(
    size: int,
    plateaus: Sequence[tuple[int, int]],
    *,
    sign: int = 1,
    scale: float = 100,
    slope_fraction: float = 0,
) -> list[float]:
    def price(step: int) -> float:
        paused = sum(max(0, min(step - start, stop - start - 1)) for start, stop in plateaus)
        return scale * math.exp(sign * 0.00035 * (step - (1 - slope_fraction) * paused))

    return [price(step) for step in range(size)]


def scan(
    rows: tuple[Observation, ...],
    direction: Direction = Direction.GROWTH,
    *,
    tick_size: float | None = None,
) -> PauseScan:
    parent = _interval(rows, 0, len(rows), direction, "observed")
    return scan_trend_pauses(rows, (parent,), AnalysisConfig(), tick_size=tick_size)


@pytest.mark.parametrize(("sign", "direction"), [(1, Direction.GROWTH), (-1, Direction.DECLINE)])
@pytest.mark.parametrize("scale", [0.01, 1e100])
def test_three_separated_pauses_in_one_trend(sign: int, direction: Direction, scale: float) -> None:
    expected = [(72, 104), (190, 230), (330, 378)]
    rows = experiment(path(440, expected, sign=sign, scale=scale)).observations
    fitted = scan(rows, direction)
    found = fitted.select()
    assert [(i.start, i.stop) for i in found] == expected
    assert all(i.price_change_percent == 0 and i.direction == direction for i in found)
    single = fitted.select(multiple=False)
    assert len(single) == 1 and single[0] in found


def test_absolute_drift_and_declared_tick_distinguish_slowdown_from_plateau() -> None:
    rows = experiment(path(192, [(72, 120)], slope_fraction=0.13)).observations
    fitted = scan(rows)
    assert [(p.start, p.stop) for p in fitted.select(absolute_limit=False)] == [(72, 120)]
    assert fitted.select() == ()
    assert scan(rows, tick_size=0.01).select() == ()
    assert [(p.start, p.stop) for p in scan(rows, tick_size=0.5).select()] == [(72, 120)]


@pytest.mark.parametrize("known", ["price_state", "trade_count", "trade_volume"])
def test_sparse_executions_are_fitted_on_the_original_grid(known: str) -> None:
    prices = path(192, [(72, 104)])
    available = [prices[step - step % 3] for step in range(192)]
    rows = experiment(available).observations
    sparse = []
    for row in rows:
        active = row.step % 3 == 0
        fields = row.model_dump()
        fields.update(trade_count=int(active) if known == "trade_count" else None)
        fields.update(trade_volume=float(active) if known == "trade_volume" else None)
        sparse.append(
            ExportObservation(
                **fields,
                price_state="observed" if active else "carried",
                bid_state="unknown",
                ask_state="unknown",
            )
            if known == "price_state"
            else Observation(**fields)
        )
    result = scan(tuple(sparse)).select()
    assert [(p.start, p.stop, p.price_change_percent) for p in result] == [(72, 104, 0.0)]


@pytest.mark.parametrize("missing", ["carried", "unknown_aggregates", "no_price", "one_execution"])
def test_unobserved_pause_never_masquerades_as_an_observed_plateau(missing: str) -> None:
    prices: list[float | None] = list(path(192, [(72, 104)]))
    counts: list[int | None] = [1] * 192
    counts[72:104] = [None if missing == "unknown_aggregates" else 0] * 32
    if missing == "no_price":
        prices[72:104] = [None] * 32
    if missing == "one_execution":
        counts[88] = 1
    rows = experiment(prices, counts).observations
    assert not scan(rows).select()


@pytest.mark.parametrize("shape", ["long_flat", "edge", "straight", "one_jump", "reversal"])
def test_no_pause_without_movement_on_both_immediate_flanks(shape: str) -> None:
    prices = {
        "long_flat": path(320, [(72, 186)]),
        "edge": path(192, [(0, 48)]),
        "straight": path(192, []),
        "one_jump": [100.0 + int(step >= 96) for step in range(192)],
        "reversal": [100 + min(step, 191 - step) / 100 for step in range(192)],
    }[shape]
    assert not scan(experiment(prices).observations).select()


def test_parent_grid_gaps_and_direction_constrain_every_pause() -> None:
    rows = experiment(path(192, [(72, 104)])).observations
    config = AnalysisConfig()
    parents = (
        _interval(rows, 0, 72, Direction.GROWTH, "observed"),
        _interval(rows, 72, 104, None, "no_trades"),
        _interval(rows, 104, 192, Direction.GROWTH, "observed"),
    )
    assert find_trend_pauses(rows, parents, config) == ()
    assert not scan(rows, Direction.SIDEWAYS).select()
    assert not scan(rows, Direction.DECLINE).select()
    with pytest.raises(ValueError, match="grid of one instrument"):
        find_trend_pauses(rows[::-1], parents, config)
    with pytest.raises(ValueError, match="without overlaps or gaps"):
        find_trend_pauses(rows, parents[1:], config)
    with pytest.raises(ValueError, match="tick"):
        find_trend_pauses(rows, parents, config, tick_size=math.nan)


@pytest.mark.parametrize("noise_floor", [sys.float_info.max, math.ulp(0.0)])
def test_extreme_positive_noise_floors_preserve_a_steady_trend(noise_floor: float) -> None:
    rows = experiment(path(192, [])).observations
    parent = _interval(rows, 0, len(rows), Direction.GROWTH, "observed")
    config = AnalysisConfig(noise_floor=noise_floor, minimum_relative_change=math.ulp(0.0))
    assert find_trend_pauses(rows, (parent,), config) == ()


def test_sparse_window_moments_match_direct_least_squares() -> None:
    import numpy as np

    start, stop = 72, 104
    # Mixed missing execution timestamps and nonzero residuals exercise every fit term.
    prices = [p * math.exp(0.00009 * math.sin(i)) for i, p in enumerate(path(192, [(start, stop)]))]
    counts = [int(i % 5 not in (1, 3)) for i in range(192)]
    rows = experiment(prices, counts).observations
    fitted = scan(rows)
    ((parent, windows),) = fitted.trends
    window = next(item for item in windows if (item.start, item.stop) == (start, stop))
    config = fitted.config
    steps = np.arange(start - config.min_duration, stop + config.min_duration)
    steps = steps[np.asarray(counts)[steps] > 0]
    y = np.log(np.asarray(prices)[steps])
    plateau = np.column_stack(
        (np.ones(len(steps)), np.minimum(steps - start, 0), np.maximum(steps - stop + 1, 0))
    )
    straight = np.column_stack((np.ones(len(steps)), steps))
    plateau_rss = np.linalg.lstsq(plateau, y, rcond=None)[1].item()
    straight_rss = np.linalg.lstsq(straight, y, rcond=None)[1].item()
    noise_sq = plateau_rss / len(steps)
    gain = (straight_rss - plateau_rss) / (noise_sq + config.noise_floor**2) / math.log(parent.stop)
    middle = (steps >= start) & (steps < stop)
    slope = np.linalg.lstsq(straight[middle], y[middle], rcond=None)[0][1]
    assert window.noise == pytest.approx(math.sqrt(noise_sq), rel=1e-7)
    assert window.gain == pytest.approx(gain, rel=1e-7)
    assert window.movement == pytest.approx(abs(slope) * (stop - start - 1), rel=1e-7)
