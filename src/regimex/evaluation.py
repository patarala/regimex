"""Direction interval matching, point diagnostics, and experiment-level uncertainty."""

import hashlib
from itertools import pairwise
from typing import Literal, Protocol, cast

import numpy as np
from pydantic import JsonValue

from regimex.directions import DIRECTION_NAMES, DirectionInterval
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
    def references(self) -> tuple[tuple[DirectionInterval, ...], ...]: ...


def dense_labels(intervals: tuple[DirectionInterval, ...], length: int) -> IntArray:
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


def _point_counts(labels: IntArray, predicted: IntArray) -> tuple[IntArray, IntArray, IntArray]:
    if labels.shape != predicted.shape or labels.ndim != 1:
        raise ValueError("metrics require matching one-dimensional target arrays")
    if ((labels < -1) | (labels >= len(DIRECTION_NAMES))).any() or (
        (predicted < -1) | (predicted >= len(DIRECTION_NAMES))
    ).any():
        raise ValueError("metric targets must be direction indices or -1 for unavailable")
    eligible = labels >= 0
    covered = eligible & (predicted >= 0)
    confusion = np.zeros((len(DIRECTION_NAMES), len(DIRECTION_NAMES)), dtype=np.int64)
    np.add.at(confusion, (labels[covered], predicted[covered]), 1)
    abstentions = np.bincount(labels[eligible & ~covered], minlength=len(DIRECTION_NAMES)).astype(
        np.int64
    )
    correct = np.diag(confusion)
    counts = np.column_stack(
        (correct, confusion.sum(axis=0) - correct, confusion.sum(axis=1) + abstentions - correct)
    ).astype(np.int64)
    return counts, confusion, abstentions


def scores(labels: IntArray, predicted: IntArray) -> dict[str, JsonValue]:
    """Point diagnostics exclude unavailable references and penalize abstention."""
    counts, confusion, abstentions = _point_counts(labels, predicted)
    result = _metrics(counts)
    result.update(
        {
            "confusion_matrix": cast(JsonValue, confusion.tolist()),
            "abstentions_by_class": cast(JsonValue, abstentions.tolist()),
            "sample_count": int((labels >= 0).sum()),
            "unavailable_reference_count": int((labels < 0).sum()),
            "defined": bool((labels >= 0).any()),
        }
    )
    return result


def _matched_intervals(
    reference: list[DirectionInterval], predicted: list[DirectionInterval]
) -> int:
    """Maximum-cardinality bipartite matching, rather than order-dependent greed."""
    edges: list[list[int]] = []
    for expected in reference:
        candidates = []
        for index, actual in enumerate(predicted):
            overlap = max(0, min(expected.stop, actual.stop) - max(expected.start, actual.start))
            union = expected.stop - expected.start + actual.stop - actual.start - overlap
            if overlap / union >= IOU_THRESHOLD:
                candidates.append(index)
        edges.append(candidates)
    assigned: dict[int, int] = {}

    def augment(reference_index: int, visited: set[int]) -> bool:
        for predicted_index in edges[reference_index]:
            if predicted_index in visited:
                continue
            visited.add(predicted_index)
            if predicted_index not in assigned or augment(assigned[predicted_index], visited):
                assigned[predicted_index] = reference_index
                return True
        return False

    return sum(augment(index, set()) for index in range(len(reference)))


def _segment_counts(
    reference: tuple[DirectionInterval, ...], predicted: tuple[DirectionInterval, ...]
) -> IntArray:
    result = np.zeros((len(DIRECTION_NAMES), 3), dtype=np.int64)
    for index, name in enumerate(DIRECTION_NAMES):
        expected = [item for item in reference if item.direction == name]
        actual = [item for item in predicted if item.direction == name]
        matched = _matched_intervals(expected, actual)
        result[index] = matched, len(actual) - matched, len(expected) - matched
    return result


def _boundaries(intervals: tuple[DirectionInterval, ...]) -> list[int]:
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


def _aligned_lags(reference: list[int], predicted: list[int], tolerance: int | None) -> list[int]:
    """Match ordered transitions by maximum count, then minimum absolute lag.

    Transitions supplied here have the same type and common available block.
    An optimal absolute-distance assignment can be ordered without crossings.
    Two score rows bound working memory; byte choices permit deterministic replay.
    """
    if not reference or not predicted:
        return []
    choices = np.zeros((len(reference) + 1, len(predicted) + 1), dtype=np.uint8)
    previous_count = np.zeros(len(predicted) + 1, dtype=np.int64)
    previous_cost = np.zeros(len(predicted) + 1, dtype=np.int64)
    for i, expected in enumerate(reference, 1):
        counts = np.zeros_like(previous_count)
        costs = np.zeros_like(previous_cost)
        for j, actual in enumerate(predicted, 1):
            # On equal count/cost, prefer a pair, then skip the current prediction.
            options = [
                (-int(previous_count[j]), int(previous_cost[j]), 2),
                (-int(counts[j - 1]), int(costs[j - 1]), 1),
            ]
            lag = actual - expected
            if tolerance is None or abs(lag) <= tolerance:
                options.append(
                    (-int(previous_count[j - 1]) - 1, int(previous_cost[j - 1]) + abs(lag), 0)
                )
            count, cost, choice = min(options)
            counts[j], costs[j], choices[i, j] = -count, cost, choice
        previous_count, previous_cost = counts, costs
    lags = []
    i, j = len(reference), len(predicted)
    while i and j:
        choice = choices[i, j]
        if choice == 0:
            lags.append(predicted[j - 1] - reference[i - 1])
            i, j = i - 1, j - 1
        elif choice == 1:
            j -= 1
        else:
            i -= 1
    return list(reversed(lags))


def _typed_boundary_counts(
    reference: tuple[DirectionInterval, ...],
    predicted: tuple[DirectionInterval, ...],
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
        matched += len(_aligned_lags(expected, actual, BOUNDARY_TOLERANCE))
        lags.extend(_aligned_lags(expected, actual, None))
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


def _validate_intervals(
    intervals: tuple[DirectionInterval, ...], instrument_id: str, length: int
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


def _bootstrap(
    grouped: dict[str, ExperimentCounts], repetitions: int
) -> dict[str, JsonValue] | None:
    if repetitions == 0:
        return None
    if repetitions < 0:
        raise ValueError("bootstrap repetitions must be nonnegative")
    groups = [grouped[key] for key in sorted(grouped)]
    if not groups:
        raise ValueError("bootstrap requires at least one experiment")
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
        point_values.append(_metrics(point_counts[chosen].sum(axis=0))["macro_f1"])
        boundary_values.append(_boundary_score(boundary_counts[chosen].sum(axis=0)))
        typed_values.append(_boundary_score(typed_counts[chosen].sum(axis=0)))
    return {
        "method": "Whole experiments sampled with replacement, including all their instruments.",
        "seed": BOOTSTRAP_SEED,
        "repetitions": repetitions,
        "confidence_level": 0.95,
        "experiment_count": len(groups),
        "segment_macro_f1_percentile_interval": cast(
            JsonValue,
            np.quantile(np.asarray(segment_values, dtype=np.float64), [0.025, 0.975]).tolist(),
        ),
        "point_macro_f1_percentile_interval": cast(
            JsonValue,
            np.quantile(np.asarray(point_values, dtype=np.float64), [0.025, 0.975]).tolist(),
        ),
        "boundary_f1_percentile_interval": cast(
            JsonValue,
            np.quantile(np.asarray(boundary_values, dtype=np.float64), [0.025, 0.975]).tolist(),
        ),
        "typed_boundary_f1_percentile_interval": cast(
            JsonValue, np.quantile(np.asarray(typed_values), [0.025, 0.975]).tolist()
        ),
        "limitation": (
            "Intervals quantify experiment sampling variation, "
            "not label correctness or simulator transfer."
        ),
    }


def _boundary_score(counts: IntArray) -> float:
    matched, false_positive, false_negative = (int(value) for value in counts)
    denominator = 2 * matched + false_positive + false_negative
    return 2 * matched / denominator if denominator else 0.0


def paired_bootstrap(
    reference: dict[str, ExperimentCounts],
    candidate: dict[str, ExperimentCounts],
    *,
    repetitions: int,
    seed: int = BOOTSTRAP_SEED,
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
        return cast(float, _metrics(counts)["macro_f1"]) if index < 2 else _boundary_score(counts)

    rng = np.random.Generator(np.random.PCG64(seed))
    samples = np.empty((repetitions, 4), dtype=np.float64)
    for repeat in range(repetitions):
        chosen = rng.integers(0, len(identifiers), size=len(identifiers))
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
        interval = [float(np.quantile(values, 0.025)), -float(np.quantile(-values, 0.025))]
        metrics[name] = {
            "delta": score(right.sum(axis=0), index) - score(left.sum(axis=0), index),
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
        "limitation": (
            "Sampling uncertainty conditional on these models and references; "
            "not independent label accuracy, training-seed variation "
            "or model-selection uncertainty. "
            "Reusing checkpoint-selection data limits inference about unseen experiments. "
            "Deltas aggregate counts, not experiment-level F1 averages."
        ),
    }


def report(
    sequences: IntervalEvaluationData,
    predictions: tuple[tuple[DirectionInterval, ...], ...],
    reference_kind: Literal["automatic", "human"] = "automatic",
    bootstrap_repetitions: int = 200,
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
    reference_grids: dict[str, list[dict[str, JsonValue]]] = {}
    segment_total = np.zeros((len(DIRECTION_NAMES), 3), dtype=np.int64)
    all_predicted = np.full(len(sequences.labels), -1, dtype=np.int64)
    reference_lengths: list[int] = []
    predicted_lengths: list[int] = []
    boundary_correct = boundary_reference = boundary_predicted = 0
    for index, (info, reference, predicted) in enumerate(
        zip(sequences.info, sequences.references, predictions, strict=True)
    ):
        start, stop = int(sequences.offsets[index]), int(sequences.offsets[index + 1])
        if stop - start != info.step_count:
            raise ValueError("evaluation sequence lengths disagree with metadata")
        _validate_intervals(reference, info.instrument_id, info.step_count)
        _validate_intervals(predicted, info.instrument_id, info.step_count)
        labels = dense_labels(reference, info.step_count)
        if not np.array_equal(labels, sequences.labels[start:stop]):
            raise ValueError("evaluation references disagree with dense targets")
        predicted_labels = dense_labels(predicted, info.step_count)
        all_predicted[start:stop] = predicted_labels
        segment_counts = _segment_counts(reference, predicted)
        point_counts, _, _ = _point_counts(labels, predicted_labels)
        expected_boundaries = _boundaries(reference)
        actual_boundaries = _boundaries(predicted)
        matched = _boundary_matches(expected_boundaries, actual_boundaries)
        boundary_counts = np.asarray(
            [matched, len(actual_boundaries) - matched, len(expected_boundaries) - matched],
            dtype=np.int64,
        )
        typed_counts, lags = _typed_boundary_counts(reference, predicted, labels, predicted_labels)
        grouped_lags.setdefault(info.experiment_id, []).extend(lags)
        reference_grids.setdefault(info.experiment_id, []).append(
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
                    for item in reference
                ],
            }
        )
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
    point_metrics = scores(sequences.labels, all_predicted)
    segment_metrics = _metrics(segment_total)
    segment_metrics["iou_threshold"] = IOU_THRESHOLD
    segment_metrics["matching"] = (
        "Maximum-cardinality one-to-one matching within experiment, instrument and direction."
    )
    eligible = sequences.labels >= 0
    covered = all_predicted >= 0
    eligible_count = int(eligible.sum())
    boundary_denominator = boundary_reference + boundary_predicted
    return {
        "evaluation_version": "2.3",
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
            "predicted_classified_step_count": int(covered.sum()),
            "reference_classified_fraction": eligible_count / len(sequences.labels),
            "predicted_classified_fraction": float(covered.mean()),
            "eligible_prediction_coverage": int((eligible & covered).sum()) / eligible_count
            if eligible_count
            else None,
        },
        "experiment_count": len(grouped),
        "instrument_sequence_count": len(sequences.info),
        "observed_step_count": len(sequences.labels),
        "classified_step_count": int(covered.sum()),
        "sample_count": eligible_count,
        "confidence_intervals": _bootstrap(grouped, bootstrap_repetitions),
        "experiment_statistics": [
            {
                "experiment_id": identifier,
                "reference_grid_sha256": hashlib.sha256(
                    json_bytes(
                        sorted(
                            reference_grids[identifier], key=lambda item: str(item["instrument_id"])
                        )
                    )
                ).hexdigest(),
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
