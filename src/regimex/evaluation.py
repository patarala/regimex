"""Direction interval matching, point diagnostics, and experiment-level uncertainty."""

import hashlib
import math
from collections.abc import Iterator, Sequence
from itertools import pairwise
from typing import Literal, Protocol, cast

import numpy as np
from pydantic import JsonValue

from regimex.directions import DIRECTION_NAMES, DirectionSpan
from regimex.features import DoubleArray, IntArray
from regimex.storage import json_bytes

IOU_THRESHOLD = 0.5
BOUNDARY_TOLERANCE = 4
BOOTSTRAP_SEED = 1729
ExperimentCounts = tuple[IntArray, IntArray, IntArray, IntArray]


class EvaluationSequence(Protocol):
    @property
    def experiment_id(self) -> str: ...

    @property
    def instrument_id(self) -> str: ...

    @property
    def step_count(self) -> int: ...


class IntervalEvaluationData(Protocol):
    """Evaluation references need grid identity and targets, without generator metadata."""

    @property
    def labels(self) -> IntArray: ...

    @property
    def offsets(self) -> IntArray: ...

    @property
    def info(self) -> tuple[EvaluationSequence, ...]: ...

    @property
    def references(self) -> tuple[tuple[DirectionSpan, ...], ...]: ...


def dense_labels(intervals: tuple[DirectionSpan, ...], length: int) -> IntArray:
    result = np.full(length, -1, dtype=np.int64)
    for interval in intervals:
        if interval.direction is not None:
            result[interval.start : interval.stop] = DIRECTION_NAMES.index(interval.direction.value)
    return result


def probabilities_valid(probabilities: DoubleArray, rows: int) -> None:
    if (
        probabilities.shape != (rows, len(DIRECTION_NAMES))
        or not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
        or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-6, rtol=0)
    ):
        raise ValueError("invalid class scores")


def _metrics(counts: IntArray) -> dict[str, JsonValue]:
    per_class: dict[str, JsonValue] = {}
    f1s = []
    for name, (correct, false_positive, false_negative) in zip(
        DIRECTION_NAMES, counts, strict=True
    ):
        true_positive = int(correct)
        positive = true_positive + int(false_positive)
        support = true_positive + int(false_negative)
        precision = true_positive / positive if positive else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2 * true_positive / (positive + support) if positive + support else 0.0
        f1s.append(f1)
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "predicted": positive,
            "matched": true_positive,
        }
    return {
        "macro_f1": sum(f1s) / len(f1s),
        "per_class": per_class,
        "class_names": list(DIRECTION_NAMES),
        "zero_support_policy": (
            "All three directions remain in macro-F1; undefined class F1 is zero."
        ),
    }


def _point_table(labels: IntArray, predicted: IntArray) -> IntArray:
    """Count (reference, prediction) pairs. Row and column zero represent unavailable steps."""
    if labels.shape != predicted.shape or labels.ndim != 1:
        raise ValueError("metrics require matching one-dimensional target arrays")
    if ((labels < -1) | (labels >= len(DIRECTION_NAMES))).any() or (
        (predicted < -1) | (predicted >= len(DIRECTION_NAMES))
    ).any():
        raise ValueError("metric targets must be direction indices or -1 for unavailable")
    width = len(DIRECTION_NAMES) + 1
    bins = (labels + 1) * width + (predicted + 1)
    return np.bincount(bins, minlength=width * width).reshape(width, width)


def _point_direction_counts(table: IntArray) -> IntArray:
    confusion = table[1:, 1:]
    correct = np.diag(confusion)
    return np.column_stack(
        (correct, confusion.sum(axis=0) - correct, table[1:].sum(axis=1) - correct)
    )


def _point_counts(labels: IntArray, predicted: IntArray) -> tuple[IntArray, IntArray, IntArray]:
    """Return per-class counts, the labeled confusion matrix and abstentions."""
    table = _point_table(labels, predicted)
    return _point_direction_counts(table), table[1:, 1:], table[1:, 0]


def _point_metrics(table: IntArray) -> dict[str, JsonValue]:
    sample_count = int(table[1:].sum())
    result = _metrics(_point_direction_counts(table))
    result.update(
        {
            "confusion_matrix": cast(JsonValue, table[1:, 1:].tolist()),
            "abstentions_by_class": cast(JsonValue, table[1:, 0].tolist()),
            "sample_count": sample_count,
            "unavailable_reference_count": int(table[0].sum()),
            "defined": sample_count > 0,
        }
    )
    return result


def scores(labels: IntArray, predicted: IntArray) -> dict[str, JsonValue]:
    """Point diagnostics exclude unavailable references and penalize abstention."""
    return _point_metrics(_point_table(labels, predicted))


def interval_iou(expected: DirectionSpan, actual: DirectionSpan) -> float:
    overlap = max(0, min(expected.stop, actual.stop) - max(expected.start, actual.start))
    union = expected.stop - expected.start + actual.stop - actual.start - overlap
    return overlap / union


def matched_interval_indices(
    reference: Sequence[DirectionSpan], predicted: Sequence[DirectionSpan]
) -> dict[int, int]:
    """Match two ordered, disjoint interval lists for one instrument and direction."""
    matches: dict[int, int] = {}
    left = right = 0
    while left < len(reference) and right < len(predicted):
        expected, actual = reference[left], predicted[right]
        # Overlaps cannot cross: matching this pair can replace either interval's
        # later match without reducing the maximum number of pairs.
        if interval_iou(expected, actual) >= IOU_THRESHOLD:
            matches[left] = right
            left += 1
            right += 1
        elif expected.stop <= actual.stop:
            # The earlier end cannot overlap the other list's next item.
            left += 1
        else:
            right += 1
    return matches


def _segment_counts(
    reference: tuple[DirectionSpan, ...], predicted: tuple[DirectionSpan, ...]
) -> IntArray:
    result = np.zeros((len(DIRECTION_NAMES), 3), dtype=np.int64)
    for index, name in enumerate(DIRECTION_NAMES):
        expected = [item for item in reference if item.direction == name]
        actual = [item for item in predicted if item.direction == name]
        matched = len(matched_interval_indices(expected, actual))
        result[index] = matched, len(actual) - matched, len(expected) - matched
    return result


def segment_macro_f1(
    references: tuple[tuple[DirectionSpan, ...], ...],
    predictions: tuple[tuple[DirectionSpan, ...], ...],
) -> float:
    """Score complete decoded sequences using the pooled counts from the full report."""
    counts = np.zeros((len(DIRECTION_NAMES), 3), dtype=np.int64)
    for reference, predicted in zip(references, predictions, strict=True):
        counts += _segment_counts(reference, predicted)
    return cast(float, _metrics(counts)["macro_f1"])


def _boundaries(intervals: tuple[DirectionSpan, ...]) -> list[int]:
    return [
        right.start
        for left, right in pairwise(intervals)
        if left.direction is not None and right.direction is not None
    ]


def _boundary_matches(reference: list[int], predicted: list[int]) -> int:
    left = right = matched = 0
    while left < len(reference) and right < len(predicted):
        difference = reference[left] - predicted[right]
        if abs(difference) <= BOUNDARY_TOLERANCE:
            matched += 1
            left += 1
            right += 1
        elif difference < 0:
            left += 1
        else:
            right += 1
    return matched


def _aligned_lags(reference: list[int], predicted: list[int]) -> list[int]:
    """Pair as many transitions as possible, then minimize total absolute lag without a cutoff.

    Transitions supplied here have the same type and common available block.
    An optimal absolute-distance assignment can be ordered without crossings.
    Every transition in the shorter list therefore gets a pair.
    """
    if not reference or not predicted:
        return []
    if len(reference) == len(predicted):
        # Pairing every transition in order is the only full ordered assignment.
        return [b - a for a, b in zip(reference, predicted, strict=True)]
    if len(reference) < len(predicted):
        shorter, longer, sign = reference, predicted, 1
    else:
        shorter, longer, sign = predicted, reference, -1
    surplus = len(longer) - len(shorter)
    steps = np.asarray(longer, dtype=np.int64)
    best = np.zeros(surplus + 1, dtype=np.int64)
    paired = np.empty((len(shorter), surplus + 1), dtype=np.bool_)
    for index, step in enumerate(shorter):
        # Column k allows k skips in the longer prefix. Either pair its last
        # step here or skip it and keep the best cost from an earlier column.
        costs = best + np.abs(steps[index : index + surplus + 1] - step)
        best = np.minimum.accumulate(costs)
        paired[index] = costs == best  # Prefer the latest pair on equal cost.
    lags = []
    skipped = surplus
    for index in reversed(range(len(shorter))):
        while not paired[index, skipped]:
            skipped -= 1
        lags.append(sign * (longer[index + skipped] - shorter[index]))
    return list(reversed(lags))


def _typed_boundary_counts(
    reference: tuple[DirectionSpan, ...],
    predicted: tuple[DirectionSpan, ...],
    labels: IntArray,
    predicted_labels: IntArray,
) -> tuple[IntArray, list[int]]:
    available = (labels >= 0) & (predicted_labels >= 0)
    blocks = np.cumsum(np.r_[True, available[1:] != available[:-1]])
    groups: dict[tuple[int, str, str], tuple[list[int], list[int]]] = {}
    totals = [0, 0]
    for side, intervals in enumerate((reference, predicted)):
        for left, right in pairwise(intervals):
            if (
                left.direction is None
                or right.direction is None
                or left.direction == right.direction
            ):
                continue
            totals[side] += 1
            step = right.start
            if not available[step - 1] or not available[step]:
                continue
            key = (int(blocks[step]), left.direction.value, right.direction.value)
            groups.setdefault(key, ([], []))[side].append(step)
    matched = 0
    lags = []
    for expected, actual in groups.values():
        matched += _boundary_matches(expected, actual)
        lags.extend(_aligned_lags(expected, actual))
    return np.asarray([matched, totals[1] - matched, totals[0] - matched], dtype=np.int64), lags


def lag_summary(lags: list[int]) -> dict[str, JsonValue]:
    """Positive lag means that the predicted transition occurs later."""
    absolute = np.abs(lags)
    return {
        "count": len(lags),
        "mean_steps": float(np.mean(lags)) if lags else None,
        "median_steps": float(np.median(lags)) if lags else None,
        "mean_absolute_steps": float(np.mean(absolute)) if lags else None,
        "p90_absolute_steps": float(np.quantile(absolute, 0.9)) if lags else None,
        "minimum_steps": min(lags) if lags else None,
        "maximum_steps": max(lags) if lags else None,
        "early_count": sum(value < 0 for value in lags),
        "on_time_count": sum(value == 0 for value in lags),
        "late_count": sum(value > 0 for value in lags),
    }


def _typed_summary(counts: IntArray, lags: list[int]) -> dict[str, JsonValue]:
    matched, false_positive, false_negative = (int(value) for value in counts)
    reference, predicted = matched + false_negative, matched + false_positive
    return {
        "tolerance_steps": BOUNDARY_TOLERANCE,
        "precision": matched / predicted if predicted else 0.0,
        "recall": matched / reference if reference else 0.0,
        "f1": _boundary_score(counts),
        "matched": matched,
        "reference_count": reference,
        "predicted_count": predicted,
        "defined": bool(reference + predicted),
        "definition": (
            "Direction changes with identical before/after classes in a common available block."
        ),
        "alignment": {
            "method": (
                "Without a distance cutoff: maximum paired count, then minimum total absolute lag, "
                "independently within each transition type and common available block."
            ),
            "matched": len(lags),
            "unmatched_reference_count": reference - len(lags),
            "unmatched_prediction_count": predicted - len(lags),
            "signed_lag_steps": lag_summary(lags),
            "sign": "predicted_step - reference_step; positive means later",
            "limitation": (
                "Timing describes paired transitions only; unmatched counts remain separate. "
                "Distant pairs may represent distinct missing and extra events, "
                "so this alignment does not establish a pure timing error."
            ),
        },
    }


def validate_interval_grid(
    intervals: tuple[DirectionSpan, ...], instrument_id: str, length: int
) -> None:
    previous = 0
    for interval in intervals:
        if (
            interval.instrument_id != instrument_id
            or interval.start != previous
            or interval.stop > length
        ):
            raise ValueError(
                "evaluation intervals must cover their instrument without overlaps or gaps"
            )
        previous = interval.stop
    if previous != length:
        raise ValueError("evaluation intervals must cover the full sequence")


def _length_summary(lengths: list[int]) -> dict[str, JsonValue]:
    return {
        "interval_count": len(lengths),
        "minimum_steps": min(lengths) if lengths else None,
        "median_steps": float(np.median(lengths)) if lengths else None,
        "maximum_steps": max(lengths) if lengths else None,
        "one_or_two_step_fraction": sum(length <= 2 for length in lengths) / len(lengths)
        if lengths
        else None,
    }


def point_macro_f1_from_counts(counts: IntArray) -> float:
    """Return NaN without labeled steps. An all-abstaining prediction can score zero."""
    if not (counts[:, 0].any() or counts[:, 2].any()):
        return math.nan
    return cast(float, _metrics(counts)["macro_f1"])


def _bootstrap(
    grouped: dict[str, ExperimentCounts], repetitions: int, *, keep_segment_draws: bool = False
) -> dict[str, JsonValue] | None:
    if repetitions == 0:
        return None
    if repetitions < 0:
        raise ValueError("bootstrap repetitions must be nonnegative")
    groups = [grouped[key] for key in sorted(grouped)]
    if not groups:
        raise ValueError("bootstrap requires at least one experiment")
    metadata: dict[str, JsonValue] = {
        "method": "Whole experiments sampled with replacement, including all their instruments.",
        "seed": BOOTSTRAP_SEED,
        "repetitions": repetitions,
        "confidence_level": 0.95,
        "experiment_count": len(groups),
    }
    if len(groups) == 1:
        return {
            **metadata,
            "segment_macro_f1_percentile_interval": None,
            "point_macro_f1_percentile_interval": None,
            "point_macro_f1_undefined_draws": 0,
            "boundary_f1_percentile_interval": None,
            "typed_boundary_f1_percentile_interval": None,
            "boundary_f1_undefined_draws": 0,
            "typed_boundary_f1_undefined_draws": 0,
            "limitation": "A sampling interval requires at least two independent experiments.",
        }
    segment_counts = np.stack([item[0] for item in groups])
    point_counts = np.stack([item[1] for item in groups])
    boundary_counts = np.stack([item[2] for item in groups])
    typed_counts = np.stack([item[3] for item in groups])
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    segment_values = []
    point_values = []
    boundary_values = []
    typed_values = []
    for _ in range(repetitions):
        chosen = rng.integers(0, len(groups), size=len(groups))
        segment_values.append(_metrics(segment_counts[chosen].sum(axis=0))["macro_f1"])
        point_values.append(point_macro_f1_from_counts(point_counts[chosen].sum(axis=0)))
        boundary_values.append(
            _boundary_score(boundary_counts[chosen].sum(axis=0), undefined=math.nan)
        )
        typed_values.append(_boundary_score(typed_counts[chosen].sum(axis=0), undefined=math.nan))
    undefined_point = sum(map(math.isnan, point_values))
    undefined_boundary = sum(map(math.isnan, boundary_values))
    undefined_typed = sum(map(math.isnan, typed_values))
    return {
        **metadata,
        **({"segment_draws_pcg64": cast(JsonValue, segment_values)} if keep_segment_draws else {}),
        "segment_macro_f1_percentile_interval": cast(
            JsonValue,
            np.quantile(np.asarray(segment_values, dtype=np.float64), [0.025, 0.975]).tolist(),
        ),
        "point_macro_f1_undefined_draws": undefined_point,
        "point_macro_f1_percentile_interval": None
        if undefined_point
        else cast(
            JsonValue,
            np.quantile(np.asarray(point_values, dtype=np.float64), [0.025, 0.975]).tolist(),
        ),
        "boundary_f1_percentile_interval": None
        if undefined_boundary
        else cast(JsonValue, np.quantile(boundary_values, [0.025, 0.975]).tolist()),
        "typed_boundary_f1_percentile_interval": None
        if undefined_typed
        else cast(JsonValue, np.quantile(typed_values, [0.025, 0.975]).tolist()),
        "boundary_f1_undefined_draws": undefined_boundary,
        "typed_boundary_f1_undefined_draws": undefined_typed,
        "limitation": (
            "Intervals quantify experiment sampling variation, "
            "not label correctness or simulator transfer."
        ),
    }


def _boundary_score(counts: IntArray, *, undefined: float = 0.0) -> float:
    matched, false_positive, false_negative = (int(value) for value in counts)
    denominator = 2 * matched + false_positive + false_negative
    return 2 * matched / denominator if denominator else undefined


def paired_bootstrap(
    reference: dict[str, ExperimentCounts],
    candidate: dict[str, ExperimentCounts],
    *,
    repetitions: int,
    seed: int = BOOTSTRAP_SEED,
    index_samples: Iterator[list[int]] | None = None,
) -> dict[str, JsonValue]:
    """Compare aggregate-count scores using identical resampled experiments."""
    if not reference or reference.keys() != candidate.keys() or repetitions < 1:
        raise ValueError("paired comparison requires aligned experiments and positive repetitions")
    identifiers = sorted(reference)
    for identifier in identifiers:
        if len(reference[identifier]) != 4 or len(candidate[identifier]) != 4:
            raise ValueError("paired comparison requires four count arrays per experiment")
        for index, (left, right) in enumerate(
            zip(reference[identifier], candidate[identifier], strict=True)
        ):
            shape = (3, 3) if index < 2 else (3,)
            if any(
                values.dtype != np.int64 or values.shape != shape or (values < 0).any()
                for values in (left, right)
            ):
                raise ValueError("paired comparison requires nonnegative integer counts")
            if not np.array_equal(left[..., 0] + left[..., 2], right[..., 0] + right[..., 2]):
                raise ValueError("paired comparison reference populations differ")
    left_arrays = [np.stack([reference[key][i] for key in identifiers]) for i in range(4)]
    right_arrays = [np.stack([candidate[key][i] for key in identifiers]) for i in range(4)]

    def score(counts: IntArray, index: int) -> float:
        if index == 0:
            return cast(float, _metrics(counts)["macro_f1"])
        if index == 1:
            return point_macro_f1_from_counts(counts)
        return _boundary_score(counts, undefined=math.nan)

    rng = np.random.Generator(np.random.PCG64(seed))
    draws = repetitions if len(identifiers) > 1 else 0
    samples = np.empty((draws, 4), dtype=np.float64)
    for repeat in range(draws):
        chosen = (
            rng.integers(0, len(identifiers), size=len(identifiers))
            if index_samples is None
            else np.asarray(next(index_samples), dtype=np.int64)
        )
        for index, (left, right) in enumerate(zip(left_arrays, right_arrays, strict=True)):
            samples[repeat, index] = score(right[chosen].sum(axis=0), index) - score(
                left[chosen].sum(axis=0), index
            )
    metrics: dict[str, JsonValue] = {}
    for index, (name, left, right) in enumerate(
        zip(
            ("segment_macro_f1", "point_macro_f1", "boundary_f1", "typed_boundary_f1"),
            left_arrays,
            right_arrays,
            strict=True,
        )
    ):
        changes = [score(b, index) - score(a, index) for a, b in zip(left, right, strict=True)]
        values = samples[:, index]
        # A reflected lower quantile makes swapping models exactly negate the interval.
        interval = (
            [float(np.quantile(values, 0.025)), -float(np.quantile(-values, 0.025))]
            if draws and np.isfinite(values).all()
            else None
        )
        delta = score(right.sum(axis=0), index) - score(left.sum(axis=0), index)
        metrics[name] = {
            "delta": delta if math.isfinite(delta) else None,
            "undefined_draws": int(np.isnan(values).sum()),
            "undefined_experiments": sum(map(math.isnan, changes)),
            "percentile_interval": cast(JsonValue, interval),
            "improved_experiments": sum(value > 0 for value in changes),
            "tied_experiments": sum(value == 0 for value in changes),
            "worse_experiments": sum(value < 0 for value in changes),
        }
    return {
        "method": (
            "Paired whole-experiment bootstrap; all instruments stay together and both models "
            "use the same sampled identities."
        ),
        "seed": seed,
        "repetitions": repetitions,
        "confidence_level": 0.95,
        "experiment_count": len(identifiers),
        "metrics": metrics,
        "limitation": "A sampling interval requires at least two independent experiments."
        if not draws
        else (
            "Sampling uncertainty conditional on these models and references; "
            "not independent label accuracy, training-seed variation "
            "or model-selection uncertainty. "
            "Reusing checkpoint-selection data limits inference about unseen experiments. "
            "Deltas aggregate counts, not experiment-level F1 averages."
        ),
    }


def _reference_grid_sha256(sequences: IntervalEvaluationData, indices: list[int]) -> str:
    grid = []
    for index in sorted(indices, key=lambda i: sequences.info[i].instrument_id):
        info = sequences.info[index]
        grid.append(
            {
                "instrument_id": info.instrument_id,
                "step_count": info.step_count,
                "intervals": [
                    {
                        "start": item.start,
                        "stop": item.stop,
                        "direction": item.direction.value if item.direction is not None else None,
                        "status": item.status,
                    }
                    for item in sequences.references[index]
                ],
            }
        )
    return hashlib.sha256(json_bytes(grid)).hexdigest()


def report(
    sequences: IntervalEvaluationData,
    predictions: tuple[tuple[DirectionSpan, ...], ...],
    reference_kind: Literal["automatic", "human"] = "automatic",
    bootstrap_repetitions: int = 200,
    *,
    keep_segment_draws: bool = False,
) -> dict[str, JsonValue]:
    """Score complete intervals; model scores or reference boundaries are not inputs."""
    if reference_kind not in {"automatic", "human"}:
        raise ValueError("unknown reference kind")
    if (
        len(predictions) != len(sequences.info)
        or len(sequences.references) != len(sequences.info)
        or sequences.offsets.shape != (len(sequences.info) + 1,)
        or sequences.offsets[0] != 0
        or sequences.offsets[-1] != len(sequences.labels)
        or not len(sequences.info)
    ):
        raise ValueError("evaluation requires matching complete sequences and references")
    if len({(info.experiment_id, info.instrument_id) for info in sequences.info}) != len(
        sequences.info
    ):
        raise ValueError("evaluation cannot repeat an experiment/instrument sequence")
    grouped: dict[str, ExperimentCounts] = {}
    grouped_lags: dict[str, list[int]] = {}
    sequence_indices: dict[str, list[int]] = {}
    segment_total = np.zeros((len(DIRECTION_NAMES), 3), dtype=np.int64)
    point_total = np.zeros((len(DIRECTION_NAMES) + 1, len(DIRECTION_NAMES) + 1), dtype=np.int64)
    reference_lengths: list[int] = []
    predicted_lengths: list[int] = []
    boundary_correct = boundary_reference = boundary_predicted = 0
    for index, (info, reference, predicted) in enumerate(
        zip(sequences.info, sequences.references, predictions, strict=True)
    ):
        start, stop = int(sequences.offsets[index]), int(sequences.offsets[index + 1])
        if stop - start != info.step_count:
            raise ValueError("evaluation sequence lengths disagree with metadata")
        validate_interval_grid(reference, info.instrument_id, info.step_count)
        validate_interval_grid(predicted, info.instrument_id, info.step_count)
        labels = dense_labels(reference, info.step_count)
        if not np.array_equal(labels, sequences.labels[start:stop]):
            raise ValueError("evaluation references disagree with dense targets")
        predicted_labels = dense_labels(predicted, info.step_count)
        segment_counts = _segment_counts(reference, predicted)
        point_table = _point_table(labels, predicted_labels)
        point_total += point_table
        point_counts = _point_direction_counts(point_table)
        expected_boundaries = _boundaries(reference)
        actual_boundaries = _boundaries(predicted)
        matched = _boundary_matches(expected_boundaries, actual_boundaries)
        boundary_counts = np.asarray(
            [matched, len(actual_boundaries) - matched, len(expected_boundaries) - matched],
            dtype=np.int64,
        )
        typed_counts, lags = _typed_boundary_counts(reference, predicted, labels, predicted_labels)
        grouped_lags.setdefault(info.experiment_id, []).extend(lags)
        sequence_indices.setdefault(info.experiment_id, []).append(index)
        segment_total += segment_counts
        if info.experiment_id in grouped:
            previous_segments, previous_points, previous_boundaries, previous_typed = grouped[
                info.experiment_id
            ]
            grouped[info.experiment_id] = (
                previous_segments + segment_counts,
                previous_points + point_counts,
                previous_boundaries + boundary_counts,
                previous_typed + typed_counts,
            )
        else:
            grouped[info.experiment_id] = (
                segment_counts,
                point_counts,
                boundary_counts,
                typed_counts,
            )
        reference_lengths.extend(
            item.stop - item.start for item in reference if item.direction is not None
        )
        predicted_lengths.extend(
            item.stop - item.start for item in predicted if item.direction is not None
        )
        boundary_correct += matched
        boundary_reference += len(expected_boundaries)
        boundary_predicted += len(actual_boundaries)
    point_metrics = _point_metrics(point_total)
    segment_metrics = _metrics(segment_total)
    segment_metrics["iou_threshold"] = IOU_THRESHOLD
    segment_metrics["matching"] = (
        "Maximum-cardinality one-to-one matching within experiment, instrument and direction."
    )
    eligible_count = int(point_total[1:, :].sum())
    predicted_count = int(point_total[:, 1:].sum())
    covered_count = int(point_total[1:, 1:].sum())
    boundary_denominator = boundary_reference + boundary_predicted
    return {
        "evaluation_version": "2.5",
        "reference_kind": reference_kind,
        "reference_limitation": (
            "Agreement with automatic observation rules; "
            "not independent human accuracy or external simulator transfer."
            if reference_kind == "automatic"
            else "Agreement with explicitly attributed human references; "
            "reviewer disagreement remains possible."
        ),
        "segment_macro_f1": segment_metrics["macro_f1"],
        "point_macro_f1": point_metrics["macro_f1"],
        "primary_metric": "segment_macro_f1",
        "segment_metrics": segment_metrics,
        "point_metrics": point_metrics,
        "boundary_metrics": {
            "tolerance_steps": BOUNDARY_TOLERANCE,
            "definition": (
                "Internal boundaries between adjacent classified intervals; "
                "availability edges excluded."
            ),
            "precision": boundary_correct / boundary_predicted if boundary_predicted else 0.0,
            "recall": boundary_correct / boundary_reference if boundary_reference else 0.0,
            "f1": 2 * boundary_correct / boundary_denominator if boundary_denominator else 0.0,
            "matched": boundary_correct,
            "reference_count": boundary_reference,
            "predicted_count": boundary_predicted,
            "defined": boundary_denominator > 0,
        },
        "typed_boundary_metrics": _typed_summary(
            np.stack([value[3] for value in grouped.values()]).sum(axis=0),
            [lag for identifier in sorted(grouped_lags) for lag in grouped_lags[identifier]],
        ),
        "fragmentation": {
            "reference": _length_summary(reference_lengths),
            "predicted": _length_summary(predicted_lengths),
            "predicted_to_reference_interval_ratio": len(predicted_lengths) / len(reference_lengths)
            if reference_lengths
            else None,
        },
        "coverage": {
            "observed_step_count": len(sequences.labels),
            "reference_classified_step_count": eligible_count,
            "predicted_classified_step_count": predicted_count,
            "reference_classified_fraction": eligible_count / len(sequences.labels),
            "predicted_classified_fraction": predicted_count / len(sequences.labels),
            "eligible_prediction_coverage": covered_count / eligible_count
            if eligible_count
            else None,
        },
        "experiment_count": len(grouped),
        "instrument_sequence_count": len(sequences.info),
        "observed_step_count": len(sequences.labels),
        "classified_step_count": predicted_count,
        "sample_count": eligible_count,
        "confidence_intervals": _bootstrap(
            grouped, bootstrap_repetitions, keep_segment_draws=keep_segment_draws
        ),
        "experiment_statistics": [
            {
                "experiment_id": identifier,
                "reference_grid_sha256": _reference_grid_sha256(
                    sequences, sequence_indices[identifier]
                ),
                "segment_counts": cast(JsonValue, grouped[identifier][0].tolist()),
                "point_counts": cast(JsonValue, grouped[identifier][1].tolist()),
                "boundary_counts": cast(JsonValue, grouped[identifier][2].tolist()),
                "typed_boundary_counts": cast(JsonValue, grouped[identifier][3].tolist()),
                "typed_boundary_lags": cast(JsonValue, sorted(grouped_lags[identifier])),
                "segment_macro_f1": _metrics(grouped[identifier][0])["macro_f1"],
                "point_macro_f1": _metrics(grouped[identifier][1])["macro_f1"],
                "boundary_f1": _boundary_score(grouped[identifier][2]),
                "typed_boundary_f1": _boundary_score(grouped[identifier][3]),
            }
            for identifier in sorted(grouped)
        ],
        "experiment_statistics_layout": (
            "Each direction row and the boundary vector contain true positives, "
            "false positives, false negatives."
        ),
    }
