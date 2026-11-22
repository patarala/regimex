"""Apply the frozen price-based pause check inside decoded TCN trends."""

from refine_trend_pauses import _pause

from regimex.contracts import Observation
from regimex.directions import AnalysisConfig, Direction, DirectionInterval
from regimex.evaluation import validate_interval_grid
from regimex.segmentation import _interval, _log_prices


def refine_decoded(
    rows: tuple[Observation, ...], intervals: tuple[DirectionInterval, ...], config: AnalysisConfig
) -> tuple[DirectionInterval, ...]:
    """Check each predicted trend once, without recomputing the model's segmentation."""
    if not rows:
        raise ValueError("A complete instrument sequence is required")
    instrument = rows[0].instrument_id
    if any(row.instrument_id != instrument or row.step != idx for idx, row in enumerate(rows)):
        raise ValueError("Rows must follow the original grid of one instrument")
    validate_interval_grid(intervals, instrument, len(rows))
    result = []
    for interval in intervals:
        direction = interval.direction
        left, right = interval.start, interval.stop
        if (
            direction in (Direction.GROWTH, Direction.DECLINE)
            and right - left >= 3 * config.min_duration
        ):
            pause = _pause(_log_prices(rows[left:right], config), direction, config)
            if pause is not None:
                start, stop = (left + bound for bound in pause)
                result.extend(
                    (
                        _interval(rows, left, start, direction, "observed"),
                        _interval(rows, start, stop, Direction.SIDEWAYS, "observed"),
                        _interval(rows, stop, right, direction, "observed"),
                    )
                )
                continue
        result.append(interval)
    return tuple(result)
