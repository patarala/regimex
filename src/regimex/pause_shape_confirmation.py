"""Confirm a candidate price pause against a model of continued movement."""

from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from regimex.contracts import Observation
from regimex.directions import DirectionInterval
from regimex.trend_pauses import PauseScan, TrendPause, _execution_price

FloatArray = NDArray[np.float64]
_SLOPE_FRACTIONS = {"shape-4": 1 / 4, "shape-5": 1 / 5, "shape-6": 1 / 6}
SETTINGS = tuple(_SLOPE_FRACTIONS)


@dataclass(frozen=True)
class ConfirmedPauses:
    by_setting: dict[str, tuple[TrendPause, ...]]

    def select(self, setting: str) -> tuple[TrendPause, ...]:
        return self.by_setting[setting]


def _best_fit(
    left: FloatArray, middle: FloatArray, right: FloatArray, prices: FloatArray, fraction: float
) -> float:
    """Maximum explained variation across boundary pairs for one middle slope rule."""
    # If the middle slope is f * (left slope + right slope) / 2, the joined
    # line has just two free slopes. Centering has already fitted its intercept.
    first = left + (fraction / 2) * middle
    last = right + (fraction / 2) * middle
    aa = np.einsum("ij,ij->i", first, first)
    ab = np.einsum("ij,ij->i", first, last)
    bb = np.einsum("ij,ij->i", last, last)
    ay = np.einsum("ij,j->i", first, prices)
    by = np.einsum("ij,j->i", last, prices)
    det = aa * bb - ab**2
    explained = np.divide(
        bb * ay**2 - 2 * ab * ay * by + aa * by**2,
        det,
        out=np.full_like(det, -np.inf),
        where=det > np.finfo(float).eps * aa * bb,
    )
    return float(np.max(explained))


def _fits(
    rows: Sequence[Observation],
    parent: DirectionInterval,
    pause: TrendPause,
    scale: int,
    settings: Sequence[str],
) -> tuple[str, ...]:
    # Changing the hypothesized boundaries must not change the observations used
    # to score either explanation. Clip the shared context to the decoded trend.
    start = max(parent.start, pause.start - 3 * scale // 2)
    stop = min(parent.stop, pause.stop + 3 * scale // 2)
    prices = np.array([_execution_price(row) for row in rows[start:stop]], dtype=np.float64)
    executed = np.isfinite(prices)
    x = np.flatnonzero(executed).astype(np.float64)
    y = np.log(prices[executed])
    y -= y.mean()

    stride = max(1, scale // 8)
    radius = scale // 2 // stride
    starts = pause.start - start + stride * np.arange(-radius, radius + 1)
    widths = np.union1d(np.arange(scale, 2 * scale, stride), [pause.stop - pause.start])
    first = np.repeat(starts, len(widths))
    last = first + np.tile(widths, len(starts)) - 1
    margin = max(2, scale // 2)
    fits = (first >= margin) & (last < stop - start - margin)
    if not fits.any():
        return ()
    first, last = first[fits, None], last[fits, None]
    left = np.minimum(x - first, 0)
    right = np.maximum(x - last, 0)
    middle = np.clip(x - first, 0, last - first)
    for basis in (left, middle, right):
        basis -= basis.mean(axis=1, keepdims=True)

    flat_score = _best_fit(left, middle, right, y, 0)
    if not np.isfinite(flat_score):
        return ()
    tolerance = np.finfo(float).eps * 8 * flat_score
    total = float(y @ y)
    accepted = []
    for setting in settings:
        fraction = _SLOPE_FRACTIONS[setting]
        if flat_score - tolerance <= _best_fit(left, middle, right, y, fraction):
            continue
        # A countertrend can also produce a short flat-looking window. A noise-sized
        # gain for that extra explanation is common on a genuinely flat patch.
        opposite_score = _best_fit(left, middle, right, y, -fraction)
        residual = max(0.0, total - opposite_score) / len(y)
        if opposite_score - flat_score <= 2 * np.log(len(y)) * residual + tolerance:
            accepted.append(setting)
    return tuple(accepted)


def prepare(scan: PauseScan, settings: Sequence[str] = SETTINGS) -> ConfirmedPauses:
    """Select separated candidates once, then check each against a continuing shallow slope."""
    if not settings or any(setting not in _SLOPE_FRACTIONS for setting in settings):
        raise ValueError("Unknown or empty pause shape setting")
    accepted: dict[str, list[TrendPause]] = {setting: [] for setting in settings}
    parents = [parent for parent, _ in scan.trends]
    starts = [parent.start for parent in parents]
    for pause in scan.select(gain_multiplier=4, absolute_multiplier=1):
        parent = parents[bisect_right(starts, pause.start) - 1]
        for setting in _fits(scan.rows, parent, pause, scan.config.min_duration, settings):
            accepted[setting].append(pause)
    return ConfirmedPauses({setting: tuple(pauses) for setting, pauses in accepted.items()})


def confirm_pauses(scan: PauseScan, setting: str) -> tuple[TrendPause, ...]:
    """Check the report's selected policy using only its two alternative slopes."""
    return prepare(scan, (setting,)).select(setting)
