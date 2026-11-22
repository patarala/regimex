"""Offline second pass for a short flat interval inside a sustained price trend."""

import math
from dataclasses import dataclass

import numpy as np

from regimex.directions import AnalysisConfig, AnnotationGrid, Direction, DirectionInterval
from regimex.segmentation import _fit, _interval, _log_prices, annotate


@dataclass(frozen=True)
class PauseAnnotations:
    """Carry a distinct policy identity for scoring outside the production reference catalog."""

    experiment_id: str
    intervals: tuple[DirectionInterval, ...]
    policy = "trend-pause-refinement/1.0"


def _pause(
    values: np.ndarray, direction: Direction, config: AnalysisConfig
) -> tuple[int, int] | None:
    """Find the strongest supported pause between two trends of the original sign."""
    flank = config.min_duration
    size = len(values)
    if size < 3 * flank:
        return None
    known = np.isfinite(values)
    # Missing prices outside a local window may still belong to a well-observed flank.
    signal = np.where(known, values, 0.0)
    if direction == Direction.DECLINE:
        signal = -signal
    ranked = []
    for width in range(flank, min(2 * flank, size - 2 * flank + 1), max(1, flank // 4)):
        if width < config.minimum_observations:
            continue
        window = width + 2 * flank
        x = np.arange(window)
        design = np.column_stack((np.minimum(x - flank, 0), np.maximum(x - flank - width + 1, 0)))
        # Centering accounts for the shared intercept. Both flanks meet the same horizontal line.
        design = design - design.mean(axis=0)
        coefficients = np.linalg.pinv(design)
        windows = np.lib.stride_tricks.sliding_window_view(signal, window)
        complete = np.lib.stride_tricks.sliding_window_view(known, window).all(axis=1)
        left_slope, right_slope = (np.correlate(signal, weights) for weights in coefficients)
        x_centered = x - x.mean()
        linear_cov = np.correlate(signal, x_centered)
        linear_explained = linear_cov**2 / (x_centered @ x_centered)
        fitted_explained = left_slope * np.correlate(
            signal, design[:, 0]
        ) + right_slope * np.correlate(signal, design[:, 1])
        total = windows.var(axis=1) * window
        residual = np.maximum(total - fitted_explained, 0.0)
        gain = (fitted_explained - linear_explained) / (residual / window + config.noise_floor**2)
        center = windows[:, flank : flank + width]
        centered_time = np.arange(width) - (width - 1) / 2
        center_slope = np.einsum("ij,j->i", center, centered_time) / (centered_time @ centered_time)
        # Demand a fourfold slowdown and a gain over a single line that accounts
        # for residual noise and the number of possible splits in the trend.
        slower = np.abs(center_slope) < np.minimum(left_slope, right_slope) / 4
        valid = complete & slower & (gain > 4 * math.log(size))
        for offset in np.flatnonzero(valid):
            start = int(offset + flank)
            ranked.append((float(gain[offset]), start, start + width))
    threshold = math.log1p(config.minimum_relative_change)
    sign = 1 if direction == Direction.GROWTH else -1
    for _, start, stop in sorted(ranked, reverse=True):
        for side in (values[:start], values[stop:]):
            count = np.isfinite(side).sum()
            if (
                count < config.minimum_observations
                or count < len(side) * config.minimum_price_coverage
            ):
                break
            slope, fit = _fit(side)
            if fit < config.minimum_directional_fit or sign * slope * (len(side) - 1) < threshold:
                break
        else:
            return start, stop
    return None


def annotate_pauses(grid: AnnotationGrid, config: AnalysisConfig) -> PauseAnnotations:
    """Apply the offline pause candidate to the observed trends in a baseline annotation."""
    baseline = annotate(grid, config)
    by_instrument = {
        item.instrument_id: tuple(
            row for row in grid.observations if row.instrument_id == item.instrument_id
        )
        for item in grid.instruments
    }
    intervals = []
    for run in baseline.intervals:
        if (
            run.direction not in (Direction.GROWTH, Direction.DECLINE)
            or run.stop - run.start < 3 * config.min_duration
        ):
            intervals.append(run)
            continue
        rows = by_instrument[run.instrument_id]
        limits = _pause(_log_prices(rows[run.start : run.stop], config), run.direction, config)
        if limits is None:
            intervals.append(run)
            continue
        start, stop = (run.start + i for i in limits)
        intervals.extend(
            (
                _interval(rows, run.start, start, run.direction, "observed"),
                _interval(rows, start, stop, Direction.SIDEWAYS, "observed"),
                _interval(rows, stop, run.stop, run.direction, "observed"),
            )
        )
    result = baseline.model_copy(update={"intervals": tuple(intervals)})
    result.validate_for(grid)
    return PauseAnnotations(result.experiment_id, result.intervals)
