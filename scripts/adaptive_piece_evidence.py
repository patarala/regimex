"""Train the adaptive forest on joint fits of consecutive price intervals."""

import numpy as np
from numpy.typing import NDArray

from regimex.backends import adaptive_forest
from regimex.backends.adaptive_forest import (
    DirectionContextModel,
    Floats,
    Ints,
    fit,
)
from regimex.backends.price_segments import line_residuals
from regimex.contracts import Observation
from regimex.directions import AnalysisConfig

__all__ = ["DirectionContextModel", "features", "fit"]


def _covered(
    prefix: Floats, left: int | Ints, right: int | Ints, config: AnalysisConfig
) -> NDArray[np.bool_]:
    count = prefix[right, 0] - prefix[left, 0]
    return np.asarray(
        (count >= config.minimum_observations)
        & (count >= (right - left) * config.minimum_price_coverage)
    )


def _refine(prefix: Floats, cuts: list[int], config: AnalysisConfig, stride: int) -> None:
    # The lattice makes joint fitting affordable. Refine each surviving cut within
    # its own cell so an eight-step grid does not quantize short interval boundaries.
    for i in range(1, len(cuts) - 1):
        left, cut, right = cuts[i - 1 : i + 2]
        options = np.arange(
            max(left + config.min_duration, cut - stride + 1),
            min(right - config.min_duration, cut + stride - 1) + 1,
        )
        options = options[
            _covered(prefix, left, options, config) & _covered(prefix, options, right, config)
        ]
        errors = line_residuals(prefix, left, options) + line_residuals(prefix, options, right)
        cuts[i] = int(options[errors.argmin()])


def partitions(
    prefix: Floats, start: int, config: AnalysisConfig, noise_variance: float
) -> tuple[list[int], ...]:
    """Find the best joint fit for each of the forest's three penalty levels."""
    size = len(prefix) - 1
    width = config.min_duration
    stride = max(1, min(8, width // 4))
    # Anchor possible interior cuts to the original observation grid. Removing
    # prices before a block must not move the lattice on the rest of the export.
    interior = np.arange((-start) % stride, size, stride)
    nodes = np.r_[0, interior[(interior >= width) & (interior <= size - width)], size]
    penalties = np.asarray(adaptive_forest._PENALTIES) * np.log1p(size) * noise_variance
    costs = np.full((len(nodes), len(penalties)), np.inf)
    parents = np.zeros(costs.shape, dtype=np.int64)
    costs[0] = -penalties
    levels = np.arange(len(penalties))
    for i, end in enumerate(nodes[1:], start=1):
        last = nodes.searchsorted(end - width, side="right")
        if not last:
            continue
        begins = nodes[:last]
        residuals = line_residuals(prefix, begins, int(end))
        valid = _covered(prefix, begins, int(end), config)
        totals = costs[:last] + np.where(valid, residuals, np.inf)[:, None]
        best = totals.argmin(axis=0)
        parents[i] = best
        costs[i] = totals[best, levels] + penalties

    result = []
    for level in levels:
        index = len(nodes) - 1
        cuts = [size]
        while index:
            index = int(parents[index, level])
            cuts.append(int(nodes[index]))
        cuts.reverse()
        _refine(prefix, cuts, config, stride)
        result.append(cuts)
    return tuple(result)


def features(rows: tuple[Observation, ...], config: AnalysisConfig) -> Floats:
    return adaptive_forest.features(rows, config, partition=partitions)
