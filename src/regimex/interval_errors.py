"""Describe agreement for each reference sideways interval on one instrument's grid."""

from dataclasses import dataclass
from enum import StrEnum

from regimex.directions import Direction, DirectionInterval
from regimex.evaluation import interval_iou, matched_interval_indices, validate_interval_grid


class Outcome(StrEnum):
    MATCHED = "matched"
    SPLIT = "split"
    MERGED = "merged"
    LOW_OVERLAP = "low_overlap"
    ABSORBED = "absorbed_by_trend"
    WRONG_DIRECTION = "wrong_direction"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class SidewaysInterval:
    reference: DirectionInterval
    outcome: Outcome
    best_sideways_iou: float
    overlapping: tuple[DirectionInterval, ...]


def _overlaps(first: DirectionInterval, second: DirectionInterval) -> bool:
    return first.start < second.stop and second.start < first.stop


def sideways_intervals(
    references: tuple[DirectionInterval, ...], predictions: tuple[DirectionInterval, ...]
) -> list[SidewaysInterval]:
    """Classify matches first, then describe the geometry of each missed reference."""
    if not references:
        raise ValueError("A reference instrument grid is required")
    for intervals in (references, predictions):
        validate_interval_grid(intervals, references[0].instrument_id, references[-1].stop)
    sideways = [item for item in references if item.direction == Direction.SIDEWAYS]
    predicted = [item for item in predictions if item.direction == Direction.SIDEWAYS]
    matches = matched_interval_indices(sideways, predicted)
    merged_starts = {
        item.start for item in predicted if sum(_overlaps(item, other) for other in sideways) > 1
    }
    records = []
    for index, expected in enumerate(sideways):
        overlaps = tuple(item for item in predictions if _overlaps(item, expected))
        candidates = [item for item in overlaps if item.direction == Direction.SIDEWAYS]
        best_iou = max((interval_iou(expected, item) for item in candidates), default=0.0)
        if index in matches:
            outcome = Outcome.MATCHED
        elif len(candidates) > 1:
            outcome = Outcome.SPLIT
        elif candidates:
            outcome = (
                Outcome.MERGED if candidates[0].start in merged_starts else Outcome.LOW_OVERLAP
            )
        elif all(item.direction is None for item in overlaps):
            outcome = Outcome.UNAVAILABLE
        elif len(overlaps) == 1 and any(
            neighbor.direction == overlaps[0].direction and _overlaps(neighbor, overlaps[0])
            for neighbor in references
            if neighbor.stop == expected.start or neighbor.start == expected.stop
        ):
            outcome = Outcome.ABSORBED
        else:
            outcome = Outcome.WRONG_DIRECTION
        records.append(SidewaysInterval(expected, outcome, best_iou, overlaps))
    return records
