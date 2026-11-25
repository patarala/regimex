"""Price-block statistics shared by adaptive forests and Segment MoE."""

import math
from collections.abc import Iterator

import numpy as np
from numpy.typing import NDArray

from regimex.contracts import Observation
from regimex.directions import AnalysisConfig, observation_blocks

Floats = NDArray[np.float64]
Ints = NDArray[np.int64]


def price_blocks(
    rows: tuple[Observation, ...], config: AnalysisConfig
) -> Iterator[tuple[int, Floats, Floats, float]]:
    """Prepare log prices, line-fit moments and noise from each available block."""
    for start, stop, status in observation_blocks(rows, config):
        if status != "observed":
            continue
        raw = np.array([row.last_price for row in rows[start:stop]], dtype=np.float64)
        present = np.isfinite(raw)
        log_price = np.log(raw)
        log_price -= log_price[present][0]
        size = len(raw)
        x = np.arange(size, dtype=np.float64) * present
        y = np.where(present, log_price, 0.0)
        moments = np.column_stack((present, x, y, x * x, x * y, y * y))
        prefix = np.vstack((np.zeros(6), moments.cumsum(axis=0)))
        diff = np.diff(log_price)
        diff = diff[np.isfinite(diff)]
        noise = config.noise_floor
        if len(diff):
            noise = max(noise, 1.4826 * float(np.median(np.abs(diff - np.median(diff)))))
        yield start, log_price, prefix, noise


def line_residuals(prefix: Floats, start: int | Ints, stop: int | Ints) -> Floats:
    """Residual sum of squares for separate intercepts and slopes on each span."""
    return line_residuals_from_moments((prefix[stop] - prefix[start]).T)


def line_residuals_from_moments(moments: Floats) -> Floats:
    """Compute the line-fit error from counts and sums of x, y, x², xy and y²."""
    count, x, y, xx, xy, yy = moments
    n = np.maximum(count, 1.0)
    spread = xx - x * x / n
    covariance = xy - x * y / n
    return np.asarray(np.maximum(0.0, yy - y * y / n - covariance**2 / np.maximum(spread, 1e-12)))


def line_fit(x: Floats, y: Floats) -> tuple[float, float]:
    """Return the slope and R² on centered step coordinates."""
    centered = y - y.mean()
    xy = float(x @ centered)
    xx = max(float(x @ x), 1e-12)
    movement = xy / xx
    fit = min(1.0, xy**2 / max(xx * float(centered @ centered), 1e-24))
    return movement, fit


def scaled_movement(value: float, scale: float) -> float:
    ratio = float(value) / float(scale)
    if math.isfinite(ratio):
        return float(np.arcsinh(ratio))
    # The ratio overflowed, but asinh is still finite and rounds to log(2 * |ratio|).
    return math.copysign(math.log(2) + math.log(abs(value)) - math.log(scale), value)
