"""Contracts for retrospective direction intervals and their observed descriptors."""

import math
from collections.abc import Sequence
from enum import StrEnum
from itertools import pairwise
from statistics import pstdev
from typing import Annotated, Literal, Protocol, Self

from pydantic import Field, JsonValue, model_validator

from regimex.contracts import MAX_STEPS, Identifier, Instrument, Observation, StrictModel

DIRECTION_POLICY = "retrospective-directions/3.0"


class AnnotationGrid(Protocol):
    """Observation view needed for annotation validation, without simulator metadata."""

    @property
    def experiment_id(self) -> str: ...

    @property
    def step_count(self) -> int: ...

    @property
    def instruments(self) -> Sequence[Instrument]: ...

    @property
    def observations(self) -> Sequence[Observation]: ...


class Direction(StrEnum):
    SIDEWAYS = "sideways"
    GROWTH = "growth"
    DECLINE = "decline"


DIRECTION_NAMES = tuple(direction.value for direction in Direction)
Fraction = Annotated[float, Field(ge=0, le=1)]
Nonnegative = Annotated[float, Field(ge=0)]


class AnalysisConfig(StrictModel):
    """Versioned analysis scale; thresholds are provisional annotation conventions.

    The fit penalty is boundary_penalty * log(block length) * noise_scale**2,
    where noise_scale is the robust scale of adjacent log returns, floored at
    noise_floor. minimum_relative_change applies to a fitted sustained trend.
    Short missing spans and trading pauses remain inside intervals and lower
    their coverage.
    """

    version: Literal["2.0"] = "2.0"
    min_duration: Annotated[int, Field(ge=2, le=MAX_STEPS)] = 32
    boundary_penalty: Annotated[float, Field(gt=0)] = 8.0
    noise_floor: Annotated[float, Field(gt=0)] = 0.001
    minimum_relative_change: Annotated[float, Field(gt=0, lt=1)] = 0.005
    minimum_directional_fit: Fraction = 0.5
    minimum_observations: Annotated[int, Field(ge=2, le=MAX_STEPS)] = 4
    minimum_price_coverage: Annotated[float, Field(gt=0, le=1)] = 0.8
    decoder_switch_penalty: Annotated[float, Field(ge=0)] = 3.0


class IntervalStatistics(StrictModel):
    """Observed totals, adjacent one-step returns, and explicit coverage.

    Price change uses the first and last available prices at the reported steps.
    Return standard deviation uses population normalization and only adjacent
    observed prices; it is absent when fewer than two returns are available.
    Trade totals include known values only and are absent at zero coverage.
    """

    price_change_percent: float | None
    first_price_step: Annotated[int, Field(ge=0)] | None
    last_price_step: Annotated[int, Field(ge=0)] | None
    return_std: Nonnegative | None
    return_count: Annotated[int, Field(ge=0)]
    trade_count: Annotated[int, Field(ge=0)] | None
    trade_volume: Nonnegative | None
    trade_active_fraction: Fraction | None
    price_coverage: Fraction
    trade_count_coverage: Fraction
    trade_volume_coverage: Fraction
    longest_no_trade_run: Annotated[int, Field(ge=0)]


class DirectionInterval(StrictModel):
    instrument_id: Identifier
    start: Annotated[int, Field(ge=0, lt=MAX_STEPS)]
    stop: Annotated[int, Field(gt=0, le=MAX_STEPS)]
    direction: Direction | None
    status: Literal["observed", "no_trades", "insufficient_data"]
    statistics: IntervalStatistics

    @model_validator(mode="after")
    def meaningful_interval(self) -> Self:
        if self.stop <= self.start:
            raise ValueError("interval stop must exceed start")
        if (self.direction is not None) != (self.status == "observed"):
            raise ValueError("only observed intervals have a direction")
        return self


def summarize_interval(
    observations: tuple[Observation, ...], start: int, stop: int
) -> IntervalStatistics:
    """Summarize an interval in a complete, step-ordered instrument sequence."""
    if not 0 <= start < stop <= len(observations):
        raise ValueError("invalid interval bounds")
    rows = observations[start:stop]
    prices = [(row.step, row.last_price) for row in rows if row.last_price is not None]
    counts = [row.trade_count for row in rows if row.trade_count is not None]
    volumes = [row.trade_volume for row in rows if row.trade_volume is not None]
    returns = [
        math.log(right.last_price) - math.log(left.last_price)
        for left, right in pairwise(rows)
        if left.last_price is not None and right.last_price is not None
    ]
    run = longest = 0
    for row in rows:
        run = run + 1 if row.trade_count == 0 else 0
        longest = max(longest, run)
    return IntervalStatistics(
        price_change_percent=(100 * (prices[-1][1] / prices[0][1] - 1))
        if len(prices) >= 2
        else None,
        first_price_step=prices[0][0] if prices else None,
        last_price_step=prices[-1][0] if prices else None,
        return_std=pstdev(returns) if len(returns) >= 2 else None,
        return_count=len(returns),
        trade_count=sum(counts) if counts else None,
        trade_volume=math.fsum(volumes) if volumes else None,
        trade_active_fraction=sum(value > 0 for value in counts) / len(counts) if counts else None,
        price_coverage=len(prices) / len(rows),
        trade_count_coverage=len(counts) / len(rows),
        trade_volume_coverage=len(volumes) / len(rows),
        longest_no_trade_run=longest,
    )


def observation_blocks(
    observations: tuple[Observation, ...], config: AnalysisConfig
) -> tuple[tuple[int, int, Literal["observed", "no_trades", "insufficient_data"]], ...]:
    """Share the same availability policy between annotations and model outputs."""
    if not observations:
        raise ValueError("an instrument sequence must not be empty")
    instrument_id = observations[0].instrument_id
    if any(
        row.instrument_id != instrument_id or row.step != i for i, row in enumerate(observations)
    ):
        raise ValueError("observations must be one complete ordered instrument sequence")
    size = len(observations)
    statuses: list[Literal["observed", "no_trades", "insufficient_data"]] = ["observed"] * size
    for predicate, unavailable_status in (
        ([row.trade_count == 0 for row in observations], "no_trades"),
        ([row.last_price is None for row in observations], "insufficient_data"),
    ):
        start = 0
        while start < size:
            if not predicate[start]:
                start += 1
                continue
            stop = start + 1
            while stop < size and predicate[stop]:
                stop += 1
            if stop - start >= config.min_duration or (start == 0 and stop == size):
                for i in range(start, stop):
                    statuses[i] = (
                        "no_trades" if unavailable_status == "no_trades" else "insufficient_data"
                    )
            start = stop
    result: list[tuple[int, int, Literal["observed", "no_trades", "insufficient_data"]]] = []
    start = 0
    while start < size:
        status: Literal["observed", "no_trades", "insufficient_data"] = statuses[start]
        stop = start + 1
        while stop < size and statuses[stop] == status:
            stop += 1
        known = sum(row.last_price is not None for row in observations[start:stop])
        if status == "observed" and (
            known < config.minimum_observations
            or known / (stop - start) < config.minimum_price_coverage
        ):
            status = "insufficient_data"
        if result and result[-1][2] == status:
            result[-1] = (result[-1][0], stop, status)
        else:
            result.append((start, stop, status))
        start = stop
    return tuple(result)


def _validate_interval_availability(
    interval: DirectionInterval,
    observations: tuple[Observation, ...],
    availability: Sequence[tuple[int, int, str]],
    config: AnalysisConfig,
) -> None:
    for start, stop, status in availability:
        if start < interval.stop and stop > interval.start and status != interval.status:
            raise ValueError("interval status contradicts observed availability")
        if (
            status == "observed"
            and start <= interval.start < stop
            and interval.stop - interval.start < min(config.min_duration, stop - start)
        ):
            raise ValueError("observed interval is shorter than the minimum duration")
    if interval.status == "observed":
        known = sum(
            row.last_price is not None for row in observations[interval.start : interval.stop]
        )
        if (
            known < config.minimum_observations
            or known / (interval.stop - interval.start) < config.minimum_price_coverage
        ):
            raise ValueError("observed interval has insufficient price coverage")


def _validate_interval_statistics(
    interval: DirectionInterval, observations: tuple[Observation, ...]
) -> None:
    expected = summarize_interval(observations, interval.start, interval.stop).model_dump()
    for name, actual in interval.statistics.model_dump().items():
        value = expected[name]
        if isinstance(actual, float) and isinstance(value, float):
            if not math.isclose(actual, value, rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError(f"interval statistic {name} does not match observations")
        elif actual != value:
            raise ValueError(f"interval statistic {name} does not match observations")


def _validate_instrument_intervals(
    intervals: Sequence[DirectionInterval],
    observations: tuple[Observation, ...],
    step_count: int,
    config: AnalysisConfig,
) -> None:
    availability = observation_blocks(observations, config)
    expected_start = 0
    previous_interval: DirectionInterval | None = None
    for interval in intervals:
        if interval.start != expected_start or interval.stop > step_count:
            raise ValueError("intervals must be ordered and cover the complete grid without gaps")
        if (
            previous_interval is not None
            and previous_interval.status == interval.status
            and previous_interval.direction == interval.direction
        ):
            raise ValueError("adjacent intervals with the same interpretation must be merged")
        _validate_interval_availability(interval, observations, availability, config)
        _validate_interval_statistics(interval, observations)
        expected_start = interval.stop
        previous_interval = interval
    if expected_start != step_count:
        raise ValueError("intervals must cover the complete grid")


class DirectionAnnotations(StrictModel):
    schema_version: Literal["3.0"] = "3.0"
    experiment_id: Identifier
    policy: Identifier = DIRECTION_POLICY
    policy_config: dict[str, JsonValue] = Field(default_factory=dict)
    reference_kind: Literal["automatic", "human"] = "automatic"
    reviewer: Identifier | None = None
    intervals: Annotated[tuple[DirectionInterval, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def reference_provenance(self) -> Self:
        AnalysisConfig.model_validate(self.policy_config)
        if self.reference_kind == "human":
            if (
                self.reviewer is None
                or not self.reviewer.strip()
                or self.policy != "human-directions/1.0"
            ):
                raise ValueError(
                    "human references require a reviewer and human-directions/1.0 policy"
                )
        elif self.reviewer is not None or self.policy != DIRECTION_POLICY:
            raise ValueError(
                "automatic annotations require the automatic policy and no human reviewer"
            )
        return self

    def validate_for(self, experiment: AnnotationGrid) -> None:
        """Check instrument coverage, interval scale, and observed evidence."""
        if self.experiment_id != experiment.experiment_id:
            raise ValueError("annotations reference a different experiment")
        config = AnalysisConfig.model_validate(self.policy_config)
        expected_ids = [instrument.instrument_id for instrument in experiment.instruments]
        if set(expected_ids) != {interval.instrument_id for interval in self.intervals}:
            raise ValueError("intervals must cover every instrument and no unknown instruments")
        for instrument_id in expected_ids:
            rows = tuple(
                row for row in experiment.observations if row.instrument_id == instrument_id
            )
            intervals = [item for item in self.intervals if item.instrument_id == instrument_id]
            _validate_instrument_intervals(intervals, rows, experiment.step_count, config)
