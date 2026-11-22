"""Price-slope contrasts across three disjoint windows of a completed instrument series."""

import numpy as np
from numpy.typing import NDArray

FLANK = 32
CENTER_WIDTHS = (32, 48)
PAUSE_FEATURE_NAMES = tuple(
    f"pause_{width}_{measure}"
    for width in CENTER_WIDTHS
    for measure in ("flank_agreement_asinh", "relative_flattening")
)


def _slopes(log_prices: NDArray[np.float64], width: int) -> NDArray[np.float64]:
    """Fit each window on the original time axis with at least 80% price coverage."""
    known = np.isfinite(log_prices).astype(np.float64)
    prices = np.nan_to_num(log_prices, nan=0.0)
    ones = np.ones(width)
    time = np.arange(width) - (width - 1) / 2
    count = np.correlate(known, ones)
    mean_t = np.correlate(known, time) / np.maximum(count, 1)
    xx = np.correlate(known, time**2) - count * mean_t**2
    xy = np.correlate(prices, time) - np.correlate(prices, ones) * mean_t
    valid = (5 * count >= 4 * width) & (xx > 0)
    return np.divide(xy, xx, out=np.full_like(xy, np.nan), where=valid)


def local_pause_features(log_prices: NDArray[np.floating]) -> NDArray[np.float64]:
    """Return two numeric contrasts per width; unavailable windows retain NaNs."""
    prices = np.asarray(log_prices, dtype=np.float64)
    if prices.ndim != 1 or np.isinf(prices).any():
        raise ValueError("Expected one sequence of finite or missing log prices")
    result = np.full((len(prices), len(PAUSE_FEATURE_NAMES)), np.nan)
    if len(prices) < 2 * FLANK + min(CENTER_WIDTHS):
        return result

    flank_slopes = _slopes(prices, FLANK)
    for i, width in enumerate(CENTER_WIDTHS):
        n = len(prices) - (2 * FLANK + width) + 1
        if n <= 0:
            continue
        left = flank_slopes[:n]
        right = flank_slopes[FLANK + width : FLANK + width + n]
        center_slopes = flank_slopes if width == FLANK else _slopes(prices, width)
        center = center_slopes[FLANK : FLANK + n]

        # Put slopes on a common log-change scale. Opposite flanks give negative agreement.
        agreement = np.sign(left) * np.sign(right) * np.minimum(np.abs(left), np.abs(right))
        flank_move = (FLANK - 1) * agreement
        center_move = (FLANK - 1) * np.abs(center)
        # A 0.1% floor keeps a tiny change on an almost flat path from looking like a pause.
        flattening = (flank_move - center_move) / (np.abs(flank_move) + center_move + 0.001)
        positions = slice(FLANK + width // 2, FLANK + width // 2 + n)
        result[positions, 2 * i] = np.arcsinh(flank_move / 0.01)
        result[positions, 2 * i + 1] = flattening
    return result
