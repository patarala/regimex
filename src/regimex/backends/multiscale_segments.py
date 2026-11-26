"""Multiscale piecewise-linear evidence for the adaptive direction forest."""

import math
from itertools import pairwise

import numpy as np

from regimex.backends.price_segments import (
    Floats,
    Ints,
    line_fit,
    line_residuals,
    line_residuals_from_moments,
    price_blocks,
    scaled_movement,
)
from regimex.contracts import Observation
from regimex.directions import AnalysisConfig

_SCALES = (0.625, 1.0, 1.625)
_MEASURES = (
    "line_movement",
    "line_fit",
    "intercept_adjusted_movement",
    "intercept_adjusted_fit",
    "supported_movement",
    "log_length",
    "previous_movement",
    "previous_fit",
    "next_movement",
    "next_fit",
)
FEATURE_NAMES = tuple(f"scale_{scale:g}_{name}" for scale in _SCALES for name in _MEASURES)
FEATURE_VERSION = "adaptive-multiscale/1.1"
LEGACY_FEATURE_VERSION = "adaptive-multiscale/1.0"


def _required_prices(size: int, config: AnalysisConfig, legacy_coverage: bool) -> Ints:
    """Minimum observed prices for every possible span length."""
    lengths = np.arange(size + 1)
    threshold = config.minimum_price_coverage
    product = lengths * threshold
    if legacy_coverage:
        counts = np.ceil(product).astype(np.int64)
    else:
        counts = product.astype(np.int64)
        # Compare the attainable fraction exactly as the decoder does. Rounding
        # the product upward would, for example, require 56 prices for 55% of 100.
        counts[1:] += counts[1:] / lengths[1:] < threshold
    return np.maximum(counts, config.minimum_observations)


def _best_predecessors(
    prefix: Floats, nodes: Ints, min_duration: int, required_prices: Ints, penalties: Floats
) -> Ints:
    """Choose the lowest-cost previous cut for each penalty and grid endpoint."""
    moments = np.ascontiguousarray(prefix[nodes].T)
    limits = nodes.searchsorted(nodes - min_duration, side="right")
    costs = np.full((len(penalties), len(nodes)), np.inf)
    costs[:, 0] = 0
    parents = np.zeros(costs.shape, dtype=np.int64)
    levels = np.arange(len(penalties))
    # Line-fit errors depend only on prices. Compute eight endpoints at a time to
    # bound the temporary arrays, then update the dependent path costs in order.
    batch_size = 8
    for first in range(1, len(nodes), batch_size):
        stop = min(first + batch_size, len(nodes))
        limit = limits[stop - 1]
        spans = moments[:, first:stop, None] - moments[:, None, :limit]
        lengths = nodes[first:stop, None] - nodes[:limit]
        enough = spans[0] >= required_prices[np.maximum(lengths, 0)]
        residual = line_residuals_from_moments(spans)
        residual[~enough] = np.inf
        for i in range(first, stop):
            count = limits[i]
            options = costs[:, :count] + residual[i - first, :count]
            best = options.argmin(axis=1)
            costs[:, i] = options[levels, best] + penalties
            parents[:, i] = best
    return parents


def _partitions(
    prefix: Floats,
    start: int,
    config: AnalysisConfig,
    noise: float,
    scales: tuple[float, ...],
    *,
    legacy_coverage: bool = False,
) -> list[list[int]]:
    """Fit the requested scales jointly on a bounded grid, then refine cuts at single steps."""
    size = len(prefix) - 1
    width = config.min_duration
    if size < 2 * width:
        return [[0, size] for _ in scales]
    # At most 512 interior candidates keep the dynamic program bounded on long exports.
    stride = max(2, math.ceil(size / 512))
    inside = np.arange((-start) % stride, size, stride)
    nodes = np.r_[0, inside[(inside >= width) & (inside <= size - width)], size]
    with np.errstate(over="ignore"):
        scaled_noise = math.sqrt(config.boundary_penalty) * noise * math.sqrt(math.log(size))
        penalties = np.asarray(scales) * scaled_noise * scaled_noise
    if not np.isfinite(penalties).all():
        return [[0, size] for _ in scales]
    required_prices = _required_prices(size, config, legacy_coverage)
    parents = _best_predecessors(prefix, nodes, width, required_prices, penalties)
    results = []
    for previous in parents:
        i = len(nodes) - 1
        cuts = [size]
        while i:
            i = int(previous[i])
            cuts.append(int(nodes[i]))
        cuts.reverse()
        for j in range(1, len(cuts) - 1):
            left_edge, cut, right_edge = cuts[j - 1 : j + 2]
            choices = np.arange(
                max(left_edge + width, cut - stride), min(right_edge - width, cut + stride) + 1
            )
            left_count = prefix[choices, 0] - prefix[left_edge, 0]
            right_count = prefix[right_edge, 0] - prefix[choices, 0]
            allowed = (left_count >= required_prices[choices - left_edge]) & (
                right_count >= required_prices[right_edge - choices]
            )
            residual = line_residuals(prefix, left_edge, choices) + line_residuals(
                prefix, choices, right_edge
            )
            cuts[j] = int(choices[np.where(allowed, residual, np.inf).argmin()])
        results.append(cuts)
    return results


def profile_partition(
    prefix: Floats,
    start: int,
    config: AnalysisConfig,
    noise: float,
    *,
    legacy_coverage: bool = False,
) -> list[int]:
    """Return the cuts at the profile's own boundary penalty for Segment MoE."""
    return _partitions(prefix, start, config, noise, (1.0,), legacy_coverage=legacy_coverage)[0]


def _measure(values: Floats) -> tuple[float, float, float, float]:
    """Estimate sustained slope with and without an abrupt shift in the price level."""
    x = np.flatnonzero(np.isfinite(values)).astype(np.float64)
    y = values[np.isfinite(values)]
    if len(y) < 2:
        return 0.0, 0.0, 0.0, 0.0
    x = (x - x.mean()) / (len(values) - 1)
    y = y - y.mean()
    slope, fit = line_fit(x, y)
    if len(y) < 4:
        return slope, fit, slope, fit
    # Quantiles cover gradual transitions. Large innovations supply exact candidates
    # for discrete price changes whose position falls between two quantiles.
    changes = np.abs(np.diff(y))
    largest = np.argpartition(changes, -min(4, len(changes)))[-4:] + 1
    cuts = np.unique(np.r_[np.linspace(2, len(y) - 2, min(17, len(y) - 3)).round(), largest])
    cuts = cuts[(cuts >= 2) & (cuts <= len(y) - 2)].astype(np.int64)
    sx = np.cumsum(x)[cuts - 1]
    sy = np.cumsum(y)[cuts - 1]
    inv = 1 / cuts + 1 / (len(y) - cuts)
    xx = np.maximum(x @ x - sx * sx * inv, 1e-24)
    yy = np.maximum(y @ y - sy * sy * inv, 0.0)
    xy = x @ y - sx * sy * inv
    explained = xy**2 / xx
    residual = np.maximum(0.0, yy - explained)
    best = int(residual.argmin())
    if residual[best] >= (1 - fit) * (y @ y) - 1e-14 * max(float(y @ y), 1e-18):
        return slope, fit, slope, fit
    return (
        slope,
        fit,
        float(xy[best] / xx[best]),
        min(1.0, float(explained[best] / max(yy[best], 1e-24))),
    )


def _support(moves: Floats, fits: Floats) -> Floats:
    """Pool adjacent slopes while their sign agrees and their fits remain credible."""

    def cumulative(movement: Floats, fit: Floats) -> Floats:
        sums = movement.copy()
        for i in range(1, len(sums)):
            if movement[i] * movement[i - 1] > 0:
                sums[i] += sums[i - 1] * min(fit[i], fit[i - 1])
        return sums

    return cumulative(moves, fits) + cumulative(moves[::-1], fits[::-1])[::-1] - moves


def features(
    rows: tuple[Observation, ...], config: AnalysisConfig, *, legacy_coverage: bool = False
) -> Floats:
    """Describe each step from observed prices in its own instrument and available block."""
    output = np.zeros((len(rows), len(FEATURE_NAMES)))
    scale = math.log1p(config.minimum_relative_change)
    for start, values, prefix, noise in price_blocks(rows, config):
        measures = {}
        partitions = _partitions(
            prefix, start, config, noise, _SCALES, legacy_coverage=legacy_coverage
        )
        for level, cuts in enumerate(partitions):
            for a, b in pairwise(cuts):
                if (a, b) not in measures:
                    measures[a, b] = _measure(values[a:b])
            piece = np.array([measures[a, b] for a, b in pairwise(cuts)])
            support = _support(piece[:, 2], piece[:, 3])
            first = level * len(_MEASURES)
            for i, (a, b) in enumerate(pairwise(cuts)):
                movement, fit, adjusted, adjusted_fit = piece[i]
                previous = piece[i - 1, 2:] if i else (0.0, 0.0)
                following = piece[i + 1, 2:] if i + 1 < len(piece) else (0.0, 0.0)
                output[start + a : start + b, first : first + len(_MEASURES)] = (
                    scaled_movement(movement, scale),
                    fit,
                    scaled_movement(adjusted, scale),
                    adjusted_fit,
                    scaled_movement(support[i], scale),
                    math.log1p(b - a),
                    scaled_movement(previous[0], scale),
                    previous[1],
                    scaled_movement(following[0], scale),
                    following[1],
                )
    return output
