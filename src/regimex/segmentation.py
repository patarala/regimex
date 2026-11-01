"""Retrospective log-linear segmentation and duration-constrained decoding."""

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from regimex.contracts import Observation
from regimex.directions import (
    DIRECTION_NAMES,
    AnalysisConfig,
    AnnotationGrid,
    Direction,
    DirectionAnnotations,
    DirectionInterval,
    observation_blocks,
    summarize_interval,
)

DECODER_VERSION = "duration-viterbi/1.0"


def eligible_mask(
    observations: tuple[Observation, ...], config: AnalysisConfig
) -> NDArray[np.bool_]:
    mask = np.zeros(len(observations), dtype=np.bool_)
    for start, stop, status in observation_blocks(observations, config):
        mask[start:stop] = status == "observed"
    return mask


def _fit(values: NDArray[np.float64]) -> tuple[float, float]:
    """Estimate sustained slope while allowing one unreported level offset.

    Each candidate has a common time slope and separate intercepts on either
    side of an observed split. Its directional fit uses within-side variation;
    the level difference cannot provide evidence for a sustained trend. This
    affects interpretation inside a piece, never its output boundaries.
    """
    known = np.isfinite(values)
    x = np.flatnonzero(known).astype(np.float64)
    y = values[known]
    if len(y) < 2:
        return 0.0, 0.0
    x -= np.mean(x)
    y = y - np.mean(y)
    moments = np.array([[x @ x, y @ y, x @ y]])
    if len(y) >= 4:
        prefix = np.vstack(
            (np.zeros((1, 5)), np.column_stack((x, y, x * x, y * y, x * y)).cumsum(axis=0))
        )
        # Both intercept groups need two observed timestamps to distinguish
        # within-side movement from a level difference. Missing bins keep x.
        counts = np.arange(2, len(y) - 1, dtype=np.float64)
        left = prefix[2:-2]
        right = prefix[-1] - left
        centered = []
        for sums, count in ((left, counts), (right, len(y) - counts)):
            sx, sy, sxx, syy, sxy = sums.T
            centered.append(
                np.column_stack(
                    (sxx - sx * sx / count, syy - sy * sy / count, sxy - sx * sy / count)
                )
            )
        moments = np.vstack((moments, centered[0] + centered[1]))
    xx = np.maximum(moments[:, 0], np.finfo(np.float64).tiny)
    yy = np.maximum(moments[:, 1], 0.0)
    xy = moments[:, 2]
    errors = np.maximum(0.0, yy - xy * xy / xx)
    tolerance = 64 * np.finfo(np.float64).eps * max(float(yy[0]), 1e-18)
    # Prefer the simpler line when improvement is only numerical roundoff.
    best = int(np.flatnonzero(errors <= errors.min() + tolerance)[0])
    slope = float(xy[best] / xx[best])
    r_squared = min(1.0, max(0.0, slope**2 * xx[best] / yy[best])) if yy[best] > 1e-18 else 0.0
    return slope, r_squared


def _pieces(values: NDArray[np.float64], config: AnalysisConfig) -> list[tuple[int, int]]:
    """Exact penalized independent line fits; jumps need not become trend classes."""
    size = len(values)
    if size < 2 * config.min_duration:
        return [(0, size)]
    known = np.isfinite(values)
    x = np.arange(size, dtype=np.float64)
    y = np.where(known, values - values[known][0], 0.0)
    prefix = [
        np.concatenate((np.zeros(1), np.cumsum(array)))
        for array in (known, x * known, x * x * known, y, y * y, x * y)
    ]
    returns = np.diff(values)
    returns = returns[np.isfinite(returns)]
    scale = config.noise_floor
    if len(returns):
        scale = max(scale, 1.4826 * float(np.median(np.abs(returns - np.median(returns)))))
    penalty = config.boundary_penalty * math.log(size) * scale**2
    cost = np.full(size + 1, np.inf)
    cost[0] = -penalty
    previous = np.full(size + 1, -1, dtype=np.int64)
    for stop in range(config.min_duration, size + 1):
        starts = np.arange(stop - config.min_duration + 1)
        count, sx, sxx, sy, syy, sxy = (array[stop] - array[starts] for array in prefix)
        count_safe = np.maximum(count, 1)
        xx = sxx - sx**2 / count_safe
        xy = sxy - sx * sy / count_safe
        yy = syy - sy**2 / count_safe
        error = np.maximum(0.0, yy - xy**2 / np.maximum(xx, 1e-12))
        sufficient = (count >= config.minimum_observations) & (
            count / (stop - starts) >= config.minimum_price_coverage
        )
        options = np.where(sufficient, cost[starts] + error + penalty, np.inf)
        best = int(np.argmin(options))
        cost[stop] = options[best]
        previous[stop] = starts[best]
    if not np.isfinite(cost[-1]):
        raise ValueError(
            "eligible observations cannot be partitioned with the configured constraints"
        )
    result = []
    stop = size
    while stop:
        start = int(previous[stop])
        result.append((start, stop))
        stop = start
    result.reverse()
    return result


@dataclass
class _FittedRun:
    start: int
    stop: int
    sign: int
    log_movement: float


def _directions(
    values: NDArray[np.float64], config: AnalysisConfig
) -> list[tuple[int, int, Direction]]:
    groups: list[_FittedRun] = []
    for start, stop in _pieces(values, config):
        slope, fit = _fit(values[start:stop])
        sign = int(np.sign(slope)) if fit >= config.minimum_directional_fit else 0
        movement = slope * (stop - start - 1) if sign else 0.0
        if groups and groups[-1].sign == sign:
            groups[-1].stop = stop
            groups[-1].log_movement += movement
        else:
            groups.append(_FittedRun(start, stop, sign, movement))

    # Supported slopes can accumulate into a sustained trend. Sum only movement
    # within each fit: refitting across pieces would turn their level offsets into
    # direction evidence, even when every piece has only a subthreshold drift.
    result: list[tuple[int, int, Direction]] = []
    for group in groups:
        direction = Direction.SIDEWAYS
        if group.sign and abs(group.log_movement) >= math.log1p(config.minimum_relative_change):
            direction = Direction.GROWTH if group.sign > 0 else Direction.DECLINE
        if result and result[-1][2] == direction:
            result[-1] = (result[-1][0], group.stop, direction)
        else:
            result.append((group.start, group.stop, direction))
    return result


def _interval(
    rows: tuple[Observation, ...], start: int, stop: int, direction: Direction | None, status: str
) -> DirectionInterval:
    if status not in {"observed", "no_trades", "insufficient_data"}:
        raise ValueError("unknown availability status")
    return DirectionInterval(
        instrument_id=rows[0].instrument_id,
        start=start,
        stop=stop,
        direction=direction,
        status="observed"
        if status == "observed"
        else ("no_trades" if status == "no_trades" else "insufficient_data"),
        statistics=summarize_interval(rows, start, stop),
    )


def annotate(
    experiment: AnnotationGrid, config: AnalysisConfig | None = None
) -> DirectionAnnotations:
    """Create automatic references, never independent human ground truth."""
    config = config or AnalysisConfig()
    intervals = []
    for instrument in experiment.instruments:
        rows = tuple(
            row for row in experiment.observations if row.instrument_id == instrument.instrument_id
        )
        for start, stop, status in observation_blocks(rows, config):
            if status != "observed":
                intervals.append(_interval(rows, start, stop, None, status))
                continue
            values = np.array(
                [
                    math.log(row.last_price) if row.last_price is not None else np.nan
                    for row in rows[start:stop]
                ],
                dtype=np.float64,
            )
            for left, right, direction in _directions(values, config):
                intervals.append(_interval(rows, start + left, start + right, direction, status))
    result = DirectionAnnotations(
        experiment_id=experiment.experiment_id,
        policy_config=config.model_dump(mode="json"),
        intervals=tuple(intervals),
    )
    result.validate_for(experiment)
    return result


def _decode_block(
    probabilities: NDArray[np.float64],
    observations: tuple[Observation, ...],
    config: AnalysisConfig,
) -> list[tuple[int, int, Direction]]:
    size = len(probabilities)
    losses = -np.log(np.maximum(probabilities, 1e-12))
    cumulative = np.vstack((np.zeros((1, len(DIRECTION_NAMES))), np.cumsum(losses, axis=0)))
    minimum = min(size, config.min_duration)
    cost = np.full((size + 1, len(DIRECTION_NAMES)), np.inf)
    starts = np.full((size + 1, len(DIRECTION_NAMES)), -1, dtype=np.int64)
    classes = np.full_like(starts, -1)
    known = np.concatenate(
        (
            np.zeros(1, dtype=np.int64),
            np.cumsum([row.last_price is not None for row in observations]),
        )
    )
    for stop in range(minimum, size + 1):
        for label in range(len(DIRECTION_NAMES)):
            if (
                known[stop] >= config.minimum_observations
                and known[stop] / stop >= config.minimum_price_coverage
            ):
                cost[stop, label] = cumulative[stop, label]
            starts[stop, label] = 0
            if stop < 2 * minimum:
                continue
            candidates = np.arange(minimum, stop - minimum + 1)
            prior = cost[candidates].copy()
            prior[:, label] = np.inf
            prior_label = np.argmin(prior, axis=1)
            options = (
                prior[np.arange(len(candidates)), prior_label]
                + cumulative[stop, label]
                - cumulative[candidates, label]
                + config.decoder_switch_penalty
            )
            counts = known[stop] - known[candidates]
            sufficient = (counts >= config.minimum_observations) & (
                counts / (stop - candidates) >= config.minimum_price_coverage
            )
            options = np.where(sufficient, options, np.inf)
            best = int(np.argmin(options))
            if options[best] < cost[stop, label]:
                cost[stop, label] = options[best]
                starts[stop, label] = candidates[best]
                classes[stop, label] = prior_label[best]
    result = []
    if not np.isfinite(cost[-1]).any():
        raise ValueError("scores cannot be decoded with the observation constraints")
    stop = size
    label = int(np.argmin(cost[-1]))
    while stop:
        start = int(starts[stop, label])
        result.append((start, stop, Direction(DIRECTION_NAMES[label])))
        label = int(classes[stop, label])
        stop = start
    result.reverse()
    return result


def decode(
    probabilities: NDArray[np.float32] | NDArray[np.float64],
    observations: tuple[Observation, ...],
    config: AnalysisConfig | None = None,
) -> tuple[DirectionInterval, ...]:
    """Jointly decode scores; only availability gaps can exempt minimum duration."""
    config = config or AnalysisConfig()
    values = np.asarray(probabilities, dtype=np.float64)
    if values.shape != (len(observations), len(DIRECTION_NAMES)):
        raise ValueError("scores must have one row per observation and one column per direction")
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("scores must be finite values between zero and one")
    if not np.allclose(values.sum(axis=1), 1.0, atol=1e-6, rtol=0):
        raise ValueError("each score row must sum to one")
    result = []
    for start, stop, status in observation_blocks(observations, config):
        if status != "observed":
            result.append(_interval(observations, start, stop, None, status))
            continue
        for left, right, direction in _decode_block(
            values[start:stop], observations[start:stop], config
        ):
            result.append(_interval(observations, start + left, start + right, direction, status))
    return tuple(result)
