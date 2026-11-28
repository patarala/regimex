"""Small, reproducible price windows for missed validation intervals."""

from pydantic import JsonValue

from regimex.directions import DIRECTION_NAMES, DirectionSpan
from regimex.evaluation import matched_interval_indices
from regimex.model_contracts import ObservationInputs


def _visible_spans(intervals: tuple[DirectionSpan, ...], start: int, stop: int) -> list[JsonValue]:
    # Keep the original bounds even when the window cuts through a neighboring interval.
    return [
        {"start": span.start, "stop": span.stop, "label": span.direction or span.status}
        for span in intervals
        if span.stop > start and span.start < stop
    ]


def missed_interval_examples(
    inputs: ObservationInputs,
    references: tuple[tuple[DirectionSpan, ...], ...],
    predictions: tuple[tuple[DirectionSpan, ...], ...],
    *,
    min_duration: int,
) -> dict[str, JsonValue]:
    """Include one actual miss per direction, preferring different experiments."""
    examples: dict[str, JsonValue] = {}
    used_experiments: set[str] = set()
    for name in DIRECTION_NAMES:
        ordered = sorted(
            range(len(inputs.info)),
            key=lambda i: (
                inputs.info[i].experiment_id in used_experiments,
                inputs.info[i].experiment_id,
                inputs.info[i].instrument_id,
            ),
        )
        for i in ordered:
            expected = [item for item in references[i] if item.direction == name]
            actual = [item for item in predictions[i] if item.direction == name]
            matched = matched_interval_indices(expected, actual)
            missed = next((span for j, span in enumerate(expected) if j not in matched), None)
            if missed is None:
                continue
            info = inputs.info[i]
            context = max(min_duration, (missed.stop - missed.start) // 2)
            start = max(0, missed.start - context)
            stop = min(info.step_count, missed.stop + context)
            instrument = inputs.instruments[i]
            examples[name] = {
                "experiment_id": info.experiment_id,
                "instrument_id": info.instrument_id,
                "price_unit": f"{instrument.quote_asset} / {instrument.base_asset}",
                "missed_start": missed.start,
                "missed_stop": missed.stop,
                "window_start": start,
                "prices": [row.last_price for row in inputs.observations[i][start:stop]],
                "reference": _visible_spans(references[i], start, stop),
                "prediction": _visible_spans(predictions[i], start, stop),
            }
            used_experiments.add(info.experiment_id)
            break
    return examples
