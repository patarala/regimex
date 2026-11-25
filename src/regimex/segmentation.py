"""Retrospective log-linear segmentation and duration-constrained decoding."""

import math
from dataclasses import dataclass
from sys import float_info

import numpy as np
from numpy.typing import NDArray

from regimex.contracts import Observation
from regimex.directions import (
    DECODER_VERSION as DECODER_VERSION,
)
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

# Keep a batch's candidate scores under 512 KiB unless one endpoint alone needs more.
_DECODER_COST_BUFFER_ELEMENTS = 65_536


def eligible_mask(
    observations: tuple[Observation, ...], config: AnalysisConfig
) -> NDArray[np.bool_]:
    mask = np.zeros(len(observations), dtype=np.bool_)
    for start, stop, status in observation_blocks(observations, config):
        mask[start:stop] = status == "observed"
    return mask


def _fit(values: NDArray[np.float64]) -> tuple[float, float]:
    """Return the slope and R² of the best fit within a piece.

    Allowing a separate intercept on each side of a level change keeps that
    change from counting as sustained movement.
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
        # Two observed prices per side let us distinguish a slope from a level change.
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


def _split_penalty(weight: float, size: int, noise: float) -> float:
    log_size = math.log(size)
    try:
        penalty = weight * log_size * noise**2
    except OverflowError:
        penalty = math.inf
    if 0 < penalty < math.inf:
        return penalty
    # A small weight can offset a noise variance too large to represent on its
    # own. Combine the exponents before rounding the product to float64.
    coefficient, exponent = math.frexp(weight)
    scale, shift = math.frexp(noise)
    try:
        return math.ldexp(coefficient * log_size * scale * scale, exponent + 2 * shift)
    except OverflowError:
        return math.inf


def _pieces(values: NDArray[np.float64], config: AnalysisConfig) -> list[tuple[int, int]]:
    """Choose piece boundaries by minimizing line-fit error plus a penalty per split."""
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
    penalty = _split_penalty(config.boundary_penalty, size, scale)
    if math.isinf(penalty):
        # A single eligible piece has no split cost, even when every split is prohibitive.
        count = int(known.sum())
        if count < config.minimum_observations or count / size < config.minimum_price_coverage:
            raise ValueError(
                "eligible observations cannot be partitioned with the configured constraints"
            )
        return [(0, size)]
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

    # Summing within-piece movement keeps level changes between pieces out of the trend.
    result: list[tuple[int, int, Direction]] = []
    threshold = math.log1p(config.minimum_relative_change)
    if config.version == "2.2":
        # A change of quote units can round the same fitted log movement either way.
        threshold -= min(16 * math.ulp(1.0), threshold * 1e-10)
    for group in groups:
        direction = Direction.SIDEWAYS
        if group.sign and abs(group.log_movement) >= threshold:
            direction = Direction.GROWTH if group.sign > 0 else Direction.DECLINE
        if result and result[-1][2] == direction:
            result[-1] = (result[-1][0], group.stop, direction)
        else:
            result.append((group.start, group.stop, direction))
    return result


def _log_prices(rows: tuple[Observation, ...], config: AnalysisConfig) -> NDArray[np.float64]:
    reference = next(row.last_price for row in rows if row.last_price is not None)
    values = np.full(len(rows), np.nan)
    for index, row in enumerate(rows):
        price = row.last_price
        if price is None:
            continue
        if config.version == "2.2":
            ratio = price / reference
            # Fall back to separate logs when extreme endpoints lose precision in the ratio.
            if math.isfinite(ratio) and ratio >= float_info.min:
                values[index] = math.log(ratio)
            else:
                values[index] = math.log(price) - math.log(reference)
        else:
            values[index] = math.log(price)
    return values


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
    """Label sustained price movements in each instrument's available blocks."""
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
            values = _log_prices(rows[start:stop], config)
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
    label_count = len(DIRECTION_NAMES)
    cumulative = np.vstack((np.zeros((1, label_count)), np.cumsum(losses, axis=0))).T.copy()
    minimum = min(size, config.min_duration)
    starts = np.zeros((size + 1, label_count), dtype=np.int64)
    # At each possible split, cache the cheapest predecessor distinct from the
    # next label. Its cost and class are independent of the end of the new interval.
    previous_cost = np.full((label_count, size + 1), np.inf)
    previous_label = np.zeros((label_count, size + 1), dtype=np.int64)
    known = np.concatenate(
        (
            np.zeros(1, dtype=np.int64),
            np.cumsum([row.last_price is not None for row in observations]),
        )
    )
    steps = np.arange(size + 1)
    all_prices_known = known[-1] == size
    # Each split precedes its endpoint by at least minimum steps. Endpoints within
    # that distance share only already-computed predecessors and can run together.
    first_stop = minimum
    while first_stop <= size:
        batch_size = min(
            minimum, max(1, _DECODER_COST_BUFFER_ELEMENTS // (label_count * first_stop))
        )
        end = min(first_stop + batch_size, size + 1)
        stops = steps[first_stop:end]
        end_cost = cumulative[:, first_stop:end].T
        eligible = (known[stops] >= config.minimum_observations) & (
            known[stops] / stops >= config.minimum_price_coverage
        )
        cost = np.where(eligible[:, None], end_cost, np.inf)
        if stops[-1] >= 2 * minimum:
            candidates = slice(minimum, end - minimum)
            lengths = stops[:, None] - steps[candidates]
            if all_prices_known:
                sufficient = lengths >= max(minimum, config.minimum_observations)
            else:
                counts = known[stops, None] - known[candidates]
                sufficient = (
                    (lengths >= minimum)
                    & (counts >= config.minimum_observations)
                    & (counts / lengths >= config.minimum_price_coverage)
                )
            options = previous_cost[None, :, candidates] + end_cost[:, :, None]
            options -= cumulative[None, :, candidates]
            options += config.decoder_switch_penalty
            np.copyto(options, np.inf, where=~sufficient[:, None, :])
            offsets = options.argmin(axis=2)
            split_cost = np.take_along_axis(options, offsets[:, :, None], axis=2)[:, :, 0]
            improved = split_cost < cost
            starts[first_stop:end] = np.where(improved, minimum + offsets, 0)
            np.copyto(cost, split_cost, where=improved)

        # Stable class order also decides ties when reconstructing the path.
        first, second = np.argsort(cost, axis=1, kind="stable")[:, :2].T
        row = np.arange(len(stops))
        previous_cost[:, first_stop:end] = cost[row, first]
        previous_label[:, first_stop:end] = first
        previous_cost[first, stops] = cost[row, second]
        previous_label[first, stops] = second
        first_stop = end
    result = []
    if not np.isfinite(cost[-1]).any():
        raise ValueError("scores cannot be decoded with the observation constraints")
    stop = size
    label = int(np.argmin(cost[-1]))
    while stop:
        start = int(starts[stop, label])
        result.append((start, stop, Direction(DIRECTION_NAMES[label])))
        label = int(previous_label[label, start])
        stop = start
    result.reverse()
    return result


def decode(
    probabilities: NDArray[np.float32] | NDArray[np.float64],
    observations: tuple[Observation, ...],
    config: AnalysisConfig | None = None,
) -> tuple[DirectionInterval, ...]:
    """Decode each available block on the original grid, including short edge blocks."""
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
