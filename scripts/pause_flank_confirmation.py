"""Check for continuing drift across a candidate pause and its two flanking trends."""

import math
from dataclasses import dataclass

import numpy as np

from regimex.directions import Direction
from regimex.trend_pauses import Floats, PauseScan, TrendPause, _execution_price

_DRIFT_LIMITS = {"drift-08": 0.08, "drift-10": 0.10, "drift-12": 0.12, "drift-14": 0.14}
SETTINGS = tuple(_DRIFT_LIMITS)


def _continuation_ratio(
    times: Floats, prices: Floats, last_central_time: float, minimum_count: int, flank: int
) -> float:
    """Fit connected lines with free slopes and breakpoints to observed log prices."""
    offsets = np.unique(np.rint(np.linspace(-flank / 2, flank / 2, 9))) / flank
    starts, ends = np.meshgrid(offsets, last_central_time + offsets)
    widths = ends - starts
    # A width counts both endpoint bins. Keep the same duration scale as the initial scan.
    allowed = (widths >= 1 - 1 / flank - 1e-12) & (widths < 2 - 1 / flank + 1e-12)
    starts, ends = starts[allowed, None], ends[allowed, None]
    before, after = times < starts, times > ends
    supported = (
        (before.sum(axis=1) >= minimum_count)
        & (after.sum(axis=1) >= minimum_count)
        & ((~(before | after)).sum(axis=1) >= minimum_count)
    )
    if not supported.any():
        return math.inf
    starts, ends = starts[supported], ends[supported]

    # Clipping elapsed time gives one slope per segment with shared prices at both joins.
    left = np.minimum(times - starts, 0)
    middle = np.clip(times - starts, 0, ends - starts)
    right = np.maximum(times - ends, 0)
    design = np.stack((np.ones_like(left), left, middle, right), axis=-1)
    centered = prices - prices.mean()
    transposed = design.transpose(0, 2, 1)
    gram = transposed @ design
    eigenvalues = np.linalg.eigvalsh(gram)
    # Dense counts can still leave nearly identical execution times on one side.
    stable = eigenvalues[:, 0] > np.sqrt(np.finfo(float).eps) * eigenvalues[:, -1]
    if not stable.any():
        return math.inf
    design = design[stable]
    slopes = np.linalg.solve(gram[stable], (transposed[stable] @ centered)[..., None])[..., 0]
    residual = (design @ slopes[..., None])[..., 0] - centered
    _, left_slope, central_slope, right_slope = slopes[np.argmin((residual**2).sum(axis=1))]
    if left_slope <= 0 or right_slope <= 0:
        return math.inf
    return float(abs(central_slope) / ((left_slope + right_slope) / 2))


@dataclass(frozen=True)
class FlankConfirmation:
    candidates: tuple[tuple[TrendPause, float], ...]

    def select(self, setting: str) -> tuple[TrendPause, ...]:
        limit = _DRIFT_LIMITS[setting]
        return tuple(pause for pause, ratio in self.candidates if ratio <= limit)


def prepare(scan: PauseScan) -> FlankConfirmation:
    """Locate possible pauses first, then check their drift on a common span of executions."""
    flank = scan.config.min_duration
    candidates = []
    for pause in scan.select(gain_multiplier=4, absolute_multiplier=2):
        # Each proposed window has a full flank on either side inside its parent interval.
        points = [
            ((row.step - pause.start) / flank, math.log(price))
            for row in scan.rows[pause.start - flank : pause.stop + flank]
            if (price := _execution_price(row)) is not None
        ]
        times, prices = np.asarray(points, dtype=np.float64).T
        if pause.direction == Direction.DECLINE:
            prices = -prices
        central_time = (pause.stop - pause.start - 1) / flank
        ratio = _continuation_ratio(
            times, prices, central_time, scan.config.minimum_observations, flank
        )
        candidates.append((pause, ratio))
    return FlankConfirmation(tuple(candidates))
