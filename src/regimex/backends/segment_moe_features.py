"""Compute Segment MoE features on observed price blocks, on the original time grid."""

from itertools import pairwise

import numpy as np
from numpy.typing import NDArray

from regimex.backends.adaptive_forest import Floats, price_blocks
from regimex.backends.multiscale_segments import _cuts
from regimex.backends.segment_moe_layout import (
    BLOCK_FEATURE_COUNT,
    CONTEXT_LENGTH,
    NEIGHBOR_RADIUS,
    RETURN_LAGS,
    SEGMENT_FEATURE_COUNT,
)
from regimex.contracts import Observation
from regimex.directions import AnalysisConfig

type Float32Array = NDArray[np.float32]


def _piece_evidence(values: Floats) -> tuple[float, float, float, float]:
    """Fit a common slope on either side of a possible price-level change."""
    finite = np.isfinite(values)
    x = np.flatnonzero(finite).astype(float)
    y = values[finite]
    if len(y) < 2:
        return 0.0, 0.0, 0.0, 0.0
    x = (x - x.mean()) / (len(values) - 1)
    y = y - y.mean()
    xx, xy, yy = x @ x, x @ y, y @ y

    # Candidate zero fits one intercept. Subtracting the between-group sums of
    # squares below fits two intercepts with one shared slope at each eligible cut.
    cuts = np.arange(2, len(y) - 1)
    sum_x, sum_y = x.cumsum()[cuts - 1], y.cumsum()[cuts - 1]
    sizes = 1 / cuts + 1 / (len(y) - cuts)
    within_xx = np.maximum(np.r_[xx, xx - sum_x * sum_x * sizes], 1e-24)
    within_xy = np.r_[xy, xy - sum_x * sum_y * sizes]
    within_yy = np.maximum(np.r_[yy, yy - sum_y * sum_y * sizes], 0.0)
    error = np.maximum(within_yy - within_xy * within_xy / within_xx, 0.0)
    tolerance = 32 * np.finfo(float).eps * max(float(yy), 1e-18)
    # Prefer the unsplit fit when a level change does not reduce residual error.
    best = np.flatnonzero(error <= error.min() + tolerance)[0]
    movement = np.clip(100 * within_xy / within_xx, -300, 300)
    fit = np.clip(within_xy * within_xy / within_xx / np.maximum(within_yy, 1e-20), 0, 1)
    return float(movement[best]), float(fit[best]), float(movement[0]), float(fit[0])


def _block_evidence(values: Floats) -> Float32Array:
    """Describe the size and consistency of changes, independent of their direction."""
    prices = values[np.isfinite(values)]
    features: list[float] = []
    for lag in RETURN_LAGS:
        changes = prices[lag:] - prices[:-lag]
        if len(changes) < 2:
            features.extend([0.0] * 7)
            continue
        magnitudes = np.abs(changes)
        nonzero = changes[magnitudes > 1e-12]
        rms = np.sqrt(np.mean(changes * changes))
        median, upper = np.log1p(1000 * np.quantile(magnitudes, [0.5, 0.9]))
        signs = np.sign(nonzero)
        features.extend(
            [
                np.log1p(1000 * rms),
                median,
                upper,
                np.mean(magnitudes) / (rms + 1e-12),
                np.abs(np.mean(changes)) / (rms + 1e-12),
                np.mean(signs[1:] * signs[:-1]) if len(signs) > 1 else 0.0,
                np.max(magnitudes) / (np.linalg.norm(changes) + 1e-12),
            ]
        )
    return np.asarray(features, dtype=np.float32)


def segment_features(
    rows: tuple[Observation, ...], config: AnalysisConfig
) -> tuple[NDArray[np.int64], Float32Array, Float32Array]:
    """Return original-grid spans, adjacent-piece evidence and block evidence per span."""
    spans = []
    contexts = []
    blocks = []
    for start, prices, prefix, noise in price_blocks(rows, config):
        cuts = _cuts(prefix, start, config, noise)[1]
        pieces = np.array([_piece_evidence(prices[a:b]) for a, b in pairwise(cuts)], np.float32)
        # Reset the window at each availability boundary.
        neighbors = np.pad(pieces, ((NEIGHBOR_RADIUS, NEIGHBOR_RADIUS), (0, 0)))
        block = _block_evidence(prices)
        for index, (left, right) in enumerate(pairwise(cuts)):
            spans.append((start + left, start + right))
            contexts.append(neighbors[index : index + CONTEXT_LENGTH])
            blocks.append(block)
    return (
        np.asarray(spans, dtype=np.int64).reshape(-1, 2),
        np.asarray(contexts, dtype=np.float32).reshape(-1, CONTEXT_LENGTH, SEGMENT_FEATURE_COUNT),
        np.asarray(blocks, dtype=np.float32).reshape(-1, BLOCK_FEATURE_COUNT),
    )
