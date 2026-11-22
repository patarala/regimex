"""Locate short price plateaus inside previously classified growth and decline intervals."""

import math
from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from regimex.contracts import Observation
from regimex.directions import AnalysisConfig, Direction, DirectionInterval
from regimex.evaluation import validate_interval_grid

Floats = NDArray[np.float64]
Indexes = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class TrendPause:
    start: int
    stop: int
    direction: Direction
    price_change_percent: float | None


class _Sums(NamedTuple):
    """Least-squares moments of execution prices on the original time axis."""

    n: Floats
    x: Floats
    y: Floats
    xx: Floats
    xy: Floats
    yy: Floats

    def centered(self) -> tuple[Floats, Floats, Floats]:
        n = np.maximum(self.n, 1)
        return self.xx - self.x**2 / n, self.xy - self.x * self.y / n, self.yy - self.y**2 / n


def _sums(
    prefix: Floats, start: Indexes | int, stop: Indexes | int, origin: Indexes | int
) -> _Sums:
    n, x, y, xx, xy, yy = (prefix[stop] - prefix[start]).T
    local_x = x - n * origin
    return _Sums(n, local_x, y, xx - origin * (x + local_x), xy - origin * y, yy)


def _divide(numerator: Floats, denominator: Floats) -> Floats:
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)


def _directional(
    side: _Sums, length: Indexes | int, minimum_change: float, config: AnalysisConfig
) -> NDArray[np.bool_]:
    xx, xy, yy = side.centered()
    return (
        (side.n >= config.minimum_observations)
        & (_divide(xy, xx) * (length - 1) >= minimum_change)
        & (_divide(xy**2, xx * yy) >= config.minimum_directional_fit)
    )


def _execution_price(row: Observation) -> float | None:
    if (
        getattr(row, "price_state", None) == "observed"
        or (row.trade_count is not None and row.trade_count > 0)
        or (row.trade_volume is not None and row.trade_volume > 0)
    ):
        return row.last_price
    return None


class _Candidate(NamedTuple):
    start: int
    stop: int
    gain: float
    movement: float
    noise: float
    tick: float


def _candidates(
    rows: Sequence[Observation],
    direction: Direction,
    config: AnalysisConfig,
    tick_size: float | None,
) -> list[_Candidate]:
    flank = config.min_duration
    size = len(rows)
    prices = np.array([_execution_price(row) for row in rows], dtype=np.float64)
    executed = np.isfinite(prices)
    steps = np.flatnonzero(executed)
    if len(steps) < 3 * config.minimum_observations:
        return []
    # Even when activity totals are unknown, a full flank without an execution
    # price cannot support a plateau. Keep gaps in the original time units.
    starts, ends = np.r_[0, steps + 1], np.r_[steps, size]
    long_gaps = ends - starts >= flank
    gap_starts, gap_ends = np.r_[starts[long_gaps], size], ends[long_gaps]
    log_price = np.zeros(size)
    log_price[executed] = np.log(prices[executed])
    origin = log_price[steps[0]]
    sign = 1 if direction == Direction.GROWTH else -1
    y = np.where(executed, sign * (log_price - origin), 0)
    x = np.arange(size, dtype=np.float64) * executed
    moments = np.column_stack((executed, x, y, x * x, x * y, y * y))
    prefix = np.vstack((np.zeros(6), np.cumsum(moments, axis=0)))
    available = np.r_[0, np.cumsum([row.last_price is not None for row in rows])]
    minimum_change = math.log1p(config.minimum_relative_change)
    flank_change = min(minimum_change, config.noise_floor)
    floor_sq = max(config.noise_floor * config.noise_floor, np.finfo(float).tiny)
    found = []
    for width in range(flank, min(2 * flank, size - 2 * flank + 1), max(1, flank // 4)):
        s = np.arange(flank, size - flank - width + 1, dtype=np.int64)
        e = s + width
        a, b = s - flank, e + flank
        left = _sums(prefix, a, s, s)
        mid = _sums(prefix, s, e, s)
        right = _sums(prefix, e, b, e - 1)
        all_points = _sums(prefix, a, b, s)

        # The plateau and both flanks need executions spread through the window.
        # Carried closes may give price coverage without observing its shape.
        next_gap = np.searchsorted(gap_ends, a, side="right")
        eligible = gap_starts[next_gap] >= b
        for start, stop, region in ((a, s, left), (s, e, mid), (e, b, right)):
            half = (start + stop) // 2
            first = prefix[half, 0] - prefix[start, 0]
            eligible &= (
                (region.n >= config.minimum_observations)
                & (first > 0)
                & (first < region.n)
                & (
                    available[stop] - available[start]
                    >= config.minimum_price_coverage * (stop - start)
                )
            )

        # The joined alternative has a shared intercept and two disjoint hinge columns.
        # Slopes are fitted to actual execution timestamps, including on sparse price grids.
        n = np.maximum(all_points.n, 1)
        ll = left.xx - left.x**2 / n
        rr = right.xx - right.x**2 / n
        lr = -left.x * right.x / n
        ly = left.xy - left.x * all_points.y / n
        ry = right.xy - right.x * all_points.y / n
        det = ll * rr - lr**2
        slope_left = _divide(rr * ly - lr * ry, det)
        slope_right = _divide(ll * ry - lr * ly, det)
        explained = slope_left * ly + slope_right * ry
        xx, xy, yy = all_points.centered()
        noise_sq = np.maximum(0, yy - explained) / n
        gain = (explained - _divide(xy**2, xx)) / (noise_sq + floor_sq) / math.log(size)
        mxx, mxy, _ = mid.centered()
        center_slope = np.abs(_divide(mxy, mxx))
        eligible &= (
            (det > 0) & (gain > 0) & (4 * center_slope < np.minimum(slope_left, slope_right))
        )
        # The shared fit can assign a positive slope to a locally flat flank.
        # Require both immediate sides to move, and both longer sides to sustain it.
        eligible &= _directional(left, flank, flank_change, config)
        eligible &= _directional(right, flank, flank_change, config)
        eligible &= _directional(_sums(prefix, 0, s, 0), s, minimum_change, config)
        eligible &= _directional(_sums(prefix, e, size, e), size - e, minimum_change, config)

        tick = np.zeros_like(gain)
        if tick_size is not None:
            center_price_log = origin + sign * _divide(mid.y, mid.n)
            tick = np.logaddexp(0, math.log(tick_size) - center_price_log)
        for i in np.flatnonzero(eligible):
            found.append(
                _Candidate(
                    int(s[i]),
                    int(e[i]),
                    float(gain[i]),
                    float(center_slope[i] * (width - 1)),
                    math.sqrt(noise_sq[i]),
                    float(tick[i]),
                )
            )
    return sorted(found, key=lambda c: (c.stop, c.start))


def _separated(candidates: list[_Candidate], gap: int) -> list[_Candidate]:
    """Maximize total support while retaining at least one trend flank between pauses."""
    ends = [item.stop for item in candidates]
    best = [0.0] * (len(ends) + 1)
    predecessor = [-1] * len(ends)
    for i, item in enumerate(candidates):
        preceding = bisect_right(ends, item.start - gap, 0, i)
        take = best[preceding] + item.gain
        if take > best[i]:
            best[i + 1] = take
            predecessor[i] = preceding
        else:
            best[i + 1] = best[i]
    selected = []
    count = len(ends)
    while count:
        i = count - 1
        if predecessor[i] < 0:
            count -= 1
        else:
            selected.append(candidates[i])
            count = predecessor[i]
    return selected[::-1]


@dataclass(frozen=True)
class PauseScan:
    """Fitted windows for comparing report policies on the same observed price path."""

    rows: Sequence[Observation]
    config: AnalysisConfig
    trends: list[tuple[DirectionInterval, list[_Candidate]]]

    def select(
        self,
        *,
        absolute_limit: bool = True,
        multiple: bool = True,
        gain_multiplier: float = 4.0,
        absolute_multiplier: float = 1.0,
    ) -> tuple[TrendPause, ...]:
        """Select windows by support and, optionally, by drift against noise and tick size."""
        if any(not math.isfinite(v) or v <= 0 for v in (gain_multiplier, absolute_multiplier)):
            raise ValueError("Pause thresholds must be positive and finite")
        maximum_drift = math.log1p(self.config.minimum_relative_change) / 2
        pauses = []
        for interval, candidates in self.trends:
            eligible = []
            for item in candidates:
                noise_limit = absolute_multiplier * max(self.config.noise_floor, item.noise)
                limit = max(item.tick, min(maximum_drift, noise_limit))
                if item.gain > gain_multiplier and (not absolute_limit or item.movement <= limit):
                    eligible.append(item)
            if not eligible:
                continue
            selected = (
                _separated(eligible, self.config.min_duration)
                if multiple
                else [max(eligible, key=lambda item: item.gain)]
            )
            direction = interval.direction
            assert direction is not None
            for item in selected:
                start, stop = interval.start + item.start, interval.start + item.stop
                prices = [
                    p for row in self.rows[start:stop] if (p := _execution_price(row)) is not None
                ]
                change = 100 * (prices[-1] / prices[0] - 1)
                pauses.append(
                    TrendPause(start, stop, direction, change if math.isfinite(change) else None)
                )
        return tuple(pauses)


def scan_trend_pauses(
    rows: Sequence[Observation],
    intervals: Sequence[DirectionInterval],
    config: AnalysisConfig,
    *,
    tick_size: float | None = None,
) -> PauseScan:
    """Scan the original grid once, so alternative thresholds can share the same observations."""
    if not rows or any(
        row.step != i or row.instrument_id != rows[0].instrument_id for i, row in enumerate(rows)
    ):
        raise ValueError("Pause detection requires the complete ordered grid of one instrument")
    validate_interval_grid(tuple(intervals), rows[0].instrument_id, len(rows))
    if tick_size is not None and (not math.isfinite(tick_size) or tick_size <= 0):
        raise ValueError("The declared tick must be positive and finite")
    trends = []
    for interval in intervals:
        if (
            interval.direction in (Direction.GROWTH, Direction.DECLINE)
            and interval.stop - interval.start >= 3 * config.min_duration
        ):
            windows = _candidates(
                rows[interval.start : interval.stop], interval.direction, config, tick_size
            )
            trends.append((interval, windows))
    return PauseScan(rows, config, trends)


def find_trend_pauses(
    rows: Sequence[Observation],
    intervals: Sequence[DirectionInterval],
    config: AnalysisConfig,
    *,
    tick_size: float | None = None,
    absolute_limit: bool = True,
    multiple: bool = True,
    gain_multiplier: float = 4.0,
    absolute_multiplier: float = 1.0,
) -> tuple[TrendPause, ...]:
    """Find half-open execution-price plateaus within decoded trends on one instrument."""
    return scan_trend_pauses(rows, intervals, config, tick_size=tick_size).select(
        absolute_limit=absolute_limit,
        multiple=multiple,
        gain_multiplier=gain_multiplier,
        absolute_multiplier=absolute_multiplier,
    )
