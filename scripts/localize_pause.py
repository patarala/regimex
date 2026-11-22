"""Compare a plateau with a shallow trend across plausible boundary locations."""

from bisect import bisect_left, bisect_right

import numpy as np
from numpy.typing import NDArray

from regimex.trend_pauses import PauseScan, TrendPause, _execution_price, _sums

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
# Context on either side (in minimum durations) and minimum proposal gain.
_POLICIES = {"localize-wide": (2.5, 2.0), "localize-near": (2.0, 4.0)}
SETTINGS = tuple(_POLICIES)


def _residuals(
    x: IntArray, y: FloatArray, first: IntArray, last: IntArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Fit two exterior slopes and a constrained interior slope for each knot pair."""
    # Prefix moments let all hypotheses share the same observed log prices without
    # allocating a matrix of shape (number of knot pairs, number of observations).
    moments = np.column_stack((np.ones(len(x)), x, y, x.astype(float) ** 2, x * y, y**2))
    prefix = np.vstack((np.zeros(6), np.cumsum(moments, axis=0)))
    start = np.searchsorted(x, first)
    stop = np.searchsorted(x, last)
    left = _sums(prefix, 0, start, first)
    center = _sums(prefix, start, stop, first)
    right = _sums(prefix, stop, len(x), last)
    width = last - first
    # The middle basis increases from 0 to width, then stays at width to join
    # both flanks continuously. The left and right bases have disjoint support.
    middle = center.x + width * right.n
    middle_sq = center.xx + width**2 * right.n - middle**2 / len(x)
    middle_y = center.xy + width * right.y
    left_sq = left.xx - left.x**2 / len(x)
    right_sq = right.xx - right.x**2 / len(x)
    left_middle = -left.x * middle / len(x)
    left_right = -left.x * right.x / len(x)
    middle_right = (width - middle / len(x)) * right.x
    total = float(y @ y)

    def fit(fraction: float) -> FloatArray:
        # The interior slope is this fraction of the mean of the two flank slopes.
        h = fraction / 2
        aa = left_sq + 2 * h * left_middle + h**2 * middle_sq
        bb = right_sq + 2 * h * middle_right + h**2 * middle_sq
        ab = left_right + h * (left_middle + middle_right) + h**2 * middle_sq
        ay = left.xy + h * middle_y
        by = right.xy + h * middle_y
        det = aa * bb - ab**2
        explained = np.divide(
            bb * ay**2 - 2 * ab * ay * by + aa * by**2,
            det,
            out=np.full_like(det, -np.inf),
            where=det > np.finfo(float).eps * aa * bb,
        )
        return np.maximum(total - explained, 0)

    return fit(0), fit(1 / 6), fit(-1 / 6)


def _supports_pause(scan: PauseScan, pause: TrendPause, start: int, stop: int) -> bool:
    scale = scan.config.min_duration
    prices = np.array([_execution_price(row) for row in scan.rows[start:stop]], dtype=np.float64)
    x = np.flatnonzero(np.isfinite(prices))
    stride = max(1, scale // 8)
    radius = scale // 2 // stride
    starts = pause.start - start + stride * np.arange(-radius, radius + 1)
    widths = np.union1d(np.arange(scale, 2 * scale, stride), [pause.stop - pause.start])
    first = np.repeat(starts, len(widths))
    last = first + np.tile(widths, len(starts)) - 1
    margin = max(2, scale // 2)
    valid = (first >= margin) & (last < stop - start - margin)
    if not valid.any() or len(x) < 3:
        return False
    y = np.log(prices[x])
    y -= y.mean()
    flat, continuing, opposite = _residuals(x, y, first[valid], last[valid])
    best = float(flat.min())
    if not np.isfinite(best):
        return False
    # A noiseless path can fit exactly. Use a scale-relative numerical floor
    # before comparing the Gaussian likelihoods averaged over the same knot grid.
    variance = max(best / len(y), 1e-12 * float(y @ y), np.finfo(float).tiny)

    def evidence(errors: FloatArray) -> float:
        return float(np.logaddexp.reduce((best - errors) / (2 * variance)))

    plateau = evidence(flat)
    return plateau > evidence(continuing) and evidence(opposite) - plateau <= np.log(len(y))


def detect(scan: PauseScan, setting: str) -> tuple[TrendPause, ...]:
    """Retain supported proposals after marginalizing over uncertain boundary locations."""
    context, gain = _POLICIES[setting]
    flank = int(context * scan.config.min_duration)
    parents = [parent for parent, _ in scan.trends]
    starts = [parent.start for parent in parents]
    accepted = []
    proposals = scan.select(gain_multiplier=gain)
    if not proposals:
        return ()
    # A nearby slowdown or pullback may fail the drift limit for a pause and
    # still interrupt its moving flank. Exclude those intervals from the fit.
    neighbours = scan.select(absolute_limit=False, gain_multiplier=gain)
    neighbour_starts = [item.start for item in neighbours]
    neighbour_stops = [item.stop for item in neighbours]
    for pause in proposals:
        parent = parents[bisect_right(starts, pause.start) - 1]
        start = max(parent.start, pause.start - flank)
        stop = min(parent.stop, pause.stop + flank)
        prior = bisect_right(neighbour_stops, pause.start)
        following = bisect_left(neighbour_starts, pause.stop)
        if prior:
            start = max(
                start, min(pause.start - scan.config.min_duration, neighbour_stops[prior - 1])
            )
        if following < len(neighbours):
            stop = min(
                stop, max(pause.stop + scan.config.min_duration, neighbour_starts[following])
            )
        if _supports_pause(scan, pause, start, stop):
            accepted.append(pause)
    return tuple(accepted)
