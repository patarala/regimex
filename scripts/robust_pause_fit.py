"""Compare plateau and shallow-trend fits with correlated execution-price noise."""

from bisect import bisect_right
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from regimex.pause_shape_confirmation import _best_fit
from regimex.trend_pauses import PauseScan, TrendPause, _execution_price

Floats = NDArray[np.float64]


@dataclass(frozen=True)
class _Policy:
    flank_widths: float
    slope_fraction: float


_POLICIES = {
    "longer-context": _Policy(3, 1 / 6),
    "long-context": _Policy(2.5, 1 / 6),
    "medium-slope": _Policy(2.5, 1 / 6.5),
}
SETTINGS = tuple(_POLICIES)


def _innovations(values: Floats, lagged: Floats) -> Floats:
    """Use elapsed steps between executions to scale the conditional differences."""
    result = values.copy()
    result[..., 1:] = (values[..., 1:] - lagged * values[..., :-1]) / np.sqrt(1 - lagged**2)
    return result


def _accepts(
    x: Floats, prices: Floats, start: int, stop: int, length: int, scale: int, policy: _Policy
) -> bool:
    stride = max(1, scale // 8)
    offsets = np.arange(-(scale // 2 // stride), scale // 2 // stride + 1) * stride
    widths = np.union1d(np.arange(scale, 2 * scale, stride), [stop - start])
    first = np.repeat(start + offsets, len(widths))
    last = first + np.tile(widths, len(offsets)) - 1
    margin = max(2, scale // 2)
    valid = (first >= margin) & (last < length - margin)
    if len(x) < 4 or not valid.any():
        return False
    first, last = first[valid, None], last[valid, None]

    lagged = np.power(0.55, np.diff(x))
    y = _innovations(np.log(prices) - np.log(prices[0]), lagged)

    # Estimate one weight per execution from a free three-slope pilot fit. Every
    # hypothesis then uses the same weights, so a poor fit cannot hide its own errors.
    columns = np.stack((np.ones_like(x), x, np.maximum(x - start, 0), np.maximum(x - stop + 1, 0)))
    pilot = _innovations(columns, lagged).T
    residual = y - pilot @ np.linalg.lstsq(pilot, y, rcond=None)[0]
    absolute = np.abs(residual - np.median(residual))
    cutoff = 3 * 1.4826 * np.median(absolute)
    weights = np.ones_like(y)
    if cutoff > np.finfo(float).eps:
        weights = np.sqrt(cutoff / np.maximum(cutoff, absolute))
    intercept = pilot[:, 0] * weights
    y *= weights
    denominator = float(intercept @ intercept)
    y -= intercept * float(y @ intercept) / denominator

    def prepare(column: Floats) -> Floats:
        weighted = _innovations(column, lagged) * weights
        projection: Floats = np.einsum("ij,j->i", weighted, intercept) / denominator
        return weighted - projection[:, None] * intercept

    # Bound the largest grid-by-execution arrays when a caller increases min_duration.
    batch_size = max(1, 32_768 // len(x))
    flat = continued = opposite = -np.inf
    for offset in range(0, len(first), batch_size):
        begin = first[offset : offset + batch_size]
        end = last[offset : offset + batch_size]
        left = prepare(np.minimum(x - begin, 0))
        middle = prepare(np.clip(x - begin, 0, end - begin))
        right = prepare(np.maximum(x - end, 0))
        flat = max(flat, _best_fit(left, middle, right, y, 0))
        continued = max(continued, _best_fit(left, middle, right, y, policy.slope_fraction))
        opposite = max(opposite, _best_fit(left, middle, right, y, -policy.slope_fraction))
    if not np.isfinite(flat):
        return False
    tolerance = np.finfo(float).eps * 8 * flat
    if flat - tolerance <= continued:
        return False
    variance = max(0.0, float(y @ y) - opposite) / len(y)
    return bool(opposite - flat <= 2 * np.log(len(y)) * variance + tolerance)


def detect(scan: PauseScan, setting: str) -> tuple[TrendPause, ...]:
    """Check proposed pauses using executed prices on the original time grid."""
    policy = _POLICIES[setting]
    scale = scan.config.min_duration
    flank = round(policy.flank_widths * scale)
    parents = [parent for parent, _ in scan.trends]
    starts = [parent.start for parent in parents]
    accepted = []
    proposals = scan.select(gain_multiplier=3, absolute_multiplier=1.5)
    for index, pause in enumerate(proposals):
        parent = parents[bisect_right(starts, pause.start) - 1]
        start = max(parent.start, pause.start - flank)
        stop = min(parent.stop, pause.stop + flank)
        # A neighboring plateau contains no evidence about the intervening trend.
        if index > 0:
            start = max(start, proposals[index - 1].stop)
        if index + 1 < len(proposals):
            stop = min(stop, proposals[index + 1].start)
        prices = np.array(
            [_execution_price(row) for row in scan.rows[start:stop]], dtype=np.float64
        )
        observed = np.isfinite(prices)
        x = np.flatnonzero(observed).astype(np.float64)
        if _accepts(
            x,
            prices[observed],
            pause.start - start,
            pause.stop - start,
            stop - start,
            scale,
            policy,
        ):
            accepted.append(pause)
    return tuple(accepted)
