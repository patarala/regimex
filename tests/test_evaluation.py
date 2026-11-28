"""Interval matching cannot be replaced by high per-step agreement."""

from dataclasses import dataclass
from typing import Literal

import pytest

from regimex.contracts import Observation
from regimex.directions import Direction, DirectionInterval, summarize_interval

np = pytest.importorskip("numpy")
from regimex.evaluation import ExperimentCounts, dense_labels, interval_iou
from regimex.features import IntArray
from regimex.prediction_inputs import PredictionSequenceInfo

evaluation = pytest.importorskip("regimex.evaluation")


def rows(length: int, instrument_id: str = "A/Q") -> tuple[Observation, ...]:
    return tuple(
        Observation(
            instrument_id=instrument_id,
            step=step,
            last_price=100.0,
            trade_count=1,
            trade_volume=1.0,
            buy_order_count=None,
            buy_order_volume=None,
            sell_order_count=None,
            sell_order_volume=None,
            best_bid=None,
            best_ask=None,
        )
        for step in range(length)
    )


def intervals(
    observations: tuple[Observation, ...], pieces: list[tuple[int, int, Direction | None]]
) -> tuple[DirectionInterval, ...]:
    return tuple(
        DirectionInterval(
            instrument_id=observations[0].instrument_id,
            start=start,
            stop=stop,
            direction=direction,
            status="observed" if direction is not None else "insufficient_data",
            statistics=summarize_interval(observations, start, stop),
        )
        for start, stop, direction in pieces
    )


@dataclass(frozen=True)
class EvaluationData:
    labels: IntArray
    offsets: IntArray
    info: tuple[PredictionSequenceInfo, ...]
    references: tuple[tuple[DirectionInterval, ...], ...]


def sequences(
    references: tuple[tuple[DirectionInterval, ...], ...],
    experiment_ids: tuple[str, ...] | None = None,
) -> EvaluationData:
    ids = experiment_ids or tuple(f"experiment-{i}" for i in range(len(references)))
    lengths = [items[-1].stop for items in references]
    offsets = np.concatenate(([0], np.cumsum(lengths))).astype(np.int64)
    labels = np.concatenate(
        [dense_labels(items, length) for items, length in zip(references, lengths, strict=True)]
    )
    result = EvaluationData(
        labels=labels,
        offsets=offsets,
        info=tuple(
            PredictionSequenceInfo(
                experiment_id=identifier,
                instrument_id=items[0].instrument_id,
                step_count=length,
            )
            for index, (identifier, items, length) in enumerate(
                zip(ids, references, lengths, strict=True)
            )
        ),
        references=references,
    )
    return result


def test_reference_grid_digest_preserves_paired_evaluation_across_input_order() -> None:
    references = (
        intervals(rows(4, "Z/Q"), [(0, 4, Direction.GROWTH)]),
        intervals(rows(3, "A/Q"), [(0, 3, Direction.DECLINE)]),
        intervals(rows(4, "A/Q"), [(0, 2, Direction.SIDEWAYS), (2, 4, None)]),
    )
    data = sequences(references, ("first", "second", "first"))
    result = evaluation.report(data, references, bootstrap_repetitions=0)
    assert {
        row["experiment_id"]: row["reference_grid_sha256"]
        for row in result["experiment_statistics"]
    } == {
        "first": "ae47f133dd2aec5e35ae8afdd4a529f6bc8deffd3f0d7f186e2424d7892ea204",
        "second": "b636042f679df45f23fafe0d91da4ca806f05e6d6a822d4b96e3e50951be145c",
    }


@pytest.mark.parametrize("reference_kind", ["automatic", "human"])
def test_perfect_three_direction_intervals_have_both_metrics_one(
    reference_kind: Literal["automatic", "human"],
) -> None:
    reference = intervals(
        rows(96),
        [(0, 32, Direction.SIDEWAYS), (32, 64, Direction.GROWTH), (64, 96, Direction.DECLINE)],
    )
    result = evaluation.report(
        sequences((reference,)),
        (reference,),
        reference_kind=reference_kind,
        bootstrap_repetitions=0,
    )
    assert result["segment_macro_f1"] == 1
    assert result["point_macro_f1"] == 1
    assert result["boundary_metrics"]["f1"] == 1
    assert result["primary_metric"] == "segment_macro_f1"
    assert result["reference_kind"] == reference_kind
    assert result["evaluation_version"] == "2.5"
    assert not any("acceptance" in key for key in result)
    assert result["confidence_intervals"] is None


def test_oversegmentation_is_penalized_despite_perfect_point_agreement() -> None:
    observations = rows(288)
    reference = intervals(
        observations,
        [(0, 96, Direction.SIDEWAYS), (96, 192, Direction.GROWTH), (192, 288, Direction.DECLINE)],
    )
    predicted = intervals(
        observations,
        [
            (start, start + 32, direction)
            for left, direction in [
                (0, Direction.SIDEWAYS),
                (96, Direction.GROWTH),
                (192, Direction.DECLINE),
            ]
            for start in range(left, left + 96, 32)
        ],
    )
    result = evaluation.report(sequences((reference,)), (predicted,), bootstrap_repetitions=0)
    assert result["point_macro_f1"] == 1
    assert result["segment_macro_f1"] == 0
    assert result["fragmentation"]["predicted_to_reference_interval_ratio"] == 3
    assert result["boundary_metrics"]["reference_count"] == 2
    assert result["boundary_metrics"]["predicted_count"] == 8


def test_one_reference_cannot_match_two_predicted_halves() -> None:
    observations = rows(64)
    reference = intervals(observations, [(0, 64, Direction.GROWTH)])
    predicted = intervals(observations, [(0, 32, Direction.GROWTH), (32, 64, Direction.GROWTH)])
    result = evaluation.report(sequences((reference,)), (predicted,), bootstrap_repetitions=0)
    metrics = result["segment_metrics"]["per_class"]["growth"]
    assert metrics["matched"] == 1
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 1
    assert metrics["f1"] == pytest.approx(2 / 3)
    assert result["segment_macro_f1"] == pytest.approx(2 / 9)


def test_strict_interval_score_uses_the_same_predictions_and_includes_exact_threshold() -> None:
    observations = rows(32)
    ref = intervals(observations, [(0, 16, Direction.GROWTH), (16, 32, Direction.DECLINE)])
    pred = intervals(
        observations,
        [(0, 12, Direction.GROWTH), (12, 22, Direction.SIDEWAYS), (22, 32, Direction.DECLINE)],
    )
    result = evaluation.report(sequences((ref,)), (pred,), bootstrap_repetitions=0)
    standard = result["segment_metrics"]
    strict = result["strict_segment_metrics"]
    assert standard["iou_threshold"] == 0.50
    assert strict["iou_threshold"] == 0.75
    assert strict["per_class"]["growth"]["matched"] == 1  # 12/16
    assert strict["per_class"]["decline"]["matched"] == 0  # 10/16
    assert standard["per_class"]["decline"]["matched"] == 1
    assert result["segment_macro_f1"] == evaluation.segment_macro_f1((ref,), (pred,)) == 2 / 3
    assert (
        strict["macro_f1"]
        == evaluation.segment_macro_f1((ref,), (pred,), iou_threshold=0.75)
        == 1 / 3
    )


def test_interval_matching_resolves_half_overlap_ties_without_scanning_all_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observations = rows(320)
    reference: list[DirectionInterval] = []
    predicted: list[DirectionInterval] = []
    expected = {}
    for start in range(0, len(observations), 10):
        index = len(reference)
        # One prediction merges the first two intervals. Two predictions split the third.
        for target, spans in (
            (reference, ((0, 2), (2, 4), (5, 9))),
            (predicted, ((0, 4), (5, 7), (7, 9))),
        ):
            target.extend(
                intervals(
                    observations,
                    [(start + a, start + b, Direction.GROWTH) for a, b in spans],
                )
            )
        expected[index] = index
        expected[index + 2] = index + 1

    calls = 0

    def count_iou(left: DirectionInterval, right: DirectionInterval) -> float:
        nonlocal calls
        calls += 1
        return interval_iou(left, right)

    monkeypatch.setattr(evaluation, "interval_iou", count_iou)
    assert evaluation.matched_interval_indices(reference, predicted) == expected
    assert calls <= len(reference) + len(predicted)


def test_boundary_tolerance_is_inclusive_and_matching_one_to_one() -> None:
    observations = rows(128)
    reference = intervals(observations, [(0, 64, Direction.GROWTH), (64, 128, Direction.DECLINE)])
    within = intervals(observations, [(0, 68, Direction.GROWTH), (68, 128, Direction.DECLINE)])
    outside = intervals(observations, [(0, 69, Direction.GROWTH), (69, 128, Direction.DECLINE)])
    data = sequences((reference,))
    assert (
        evaluation.report(data, (within,), bootstrap_repetitions=0)["boundary_metrics"]["matched"]
        == 1
    )
    assert (
        evaluation.report(data, (outside,), bootstrap_repetitions=0)["boundary_metrics"]["matched"]
        == 0
    )


def test_unavailable_references_are_excluded_but_eligible_abstention_is_false_negative() -> None:
    result = evaluation.scores(np.array([-1, 0, 1, 2]), np.array([0, 0, -1, 2]))
    assert result["sample_count"] == 3
    assert result["unavailable_reference_count"] == 1
    assert result["macro_f1"] == pytest.approx(2 / 3)
    assert result["per_class"]["growth"]["support"] == 1
    assert result["abstentions_by_class"] == [0, 1, 0]


def test_all_unavailable_is_explicitly_undefined() -> None:
    reference = intervals(rows(64), [(0, 64, None)])
    result = evaluation.report(
        sequences((reference,)), (reference,), reference_kind="human", bootstrap_repetitions=0
    )
    assert result["point_metrics"]["defined"] is False
    assert result["coverage"]["reference_classified_fraction"] == 0
    assert result["coverage"]["eligible_prediction_coverage"] is None
    assert result["reference_kind"] == "human"
    assert result["boundary_metrics"]["defined"] is False


def test_matching_is_isolated_by_experiment_and_instrument() -> None:
    first = intervals(rows(64), [(0, 64, Direction.GROWTH)])
    second = intervals(rows(64, "B/Q"), [(0, 64, Direction.DECLINE)])
    first_wrong = intervals(rows(64), [(0, 64, Direction.DECLINE)])
    second_wrong = intervals(rows(64, "B/Q"), [(0, 64, Direction.GROWTH)])
    data = sequences((first, second), ("same-experiment", "same-experiment"))
    result = evaluation.report(data, (first_wrong, second_wrong), bootstrap_repetitions=10)
    assert result["segment_macro_f1"] == 0
    assert result["experiment_count"] == 1
    assert result["instrument_sequence_count"] == 2
    assert result["confidence_intervals"]["experiment_count"] == 1


def test_experiment_bootstrap_is_reproducible_and_does_not_reorder_arrays() -> None:
    first = intervals(rows(64), [(0, 64, Direction.GROWTH)])
    second = intervals(rows(64), [(0, 64, Direction.DECLINE)])
    wrong = intervals(rows(64), [(0, 64, Direction.SIDEWAYS)])
    data = sequences((first, second))
    original = data.labels.copy()
    one = evaluation.report(data, (first, wrong), bootstrap_repetitions=40)
    two = evaluation.report(data, (first, wrong), bootstrap_repetitions=40)
    assert one == two
    assert np.array_equal(data.labels, original)
    limits = one["confidence_intervals"]["segment_macro_f1_percentile_interval"]
    assert limits[0] <= one["segment_macro_f1"] <= limits[1]
    assert one["confidence_intervals"]["experiment_count"] == 2


def test_pooled_coverage_counts_predictions_on_unlabeled_steps() -> None:
    a, b = rows(6), rows(4, "B/Q")
    reference = (
        intervals(a, [(0, 2, None), (2, 4, Direction.GROWTH), (4, 6, Direction.DECLINE)]),
        intervals(b, [(0, 2, Direction.SIDEWAYS), (2, 4, None)]),
    )
    predicted = (
        intervals(
            a,
            [
                (0, 1, None),
                (1, 2, Direction.SIDEWAYS),
                (2, 3, Direction.GROWTH),
                (3, 5, None),
                (5, 6, Direction.SIDEWAYS),
            ],
        ),
        intervals(
            b, [(0, 1, Direction.SIDEWAYS), (1, 2, Direction.DECLINE), (2, 4, Direction.GROWTH)]
        ),
    )
    data = sequences(reference, ("shared", "shared"))
    result = evaluation.report(data, predicted, bootstrap_repetitions=0)
    metrics = result["point_metrics"]
    assert metrics["confusion_matrix"] == [[1, 0, 1], [0, 1, 0], [1, 0, 0]]
    assert metrics["abstentions_by_class"] == [0, 1, 1]
    assert metrics["unavailable_reference_count"] == 4
    assert result["coverage"] == {
        "observed_step_count": 10,
        "reference_classified_step_count": 6,
        "predicted_classified_step_count": 7,
        "reference_classified_fraction": 0.6,
        "predicted_classified_fraction": 0.7,
        "eligible_prediction_coverage": 4 / 6,
    }
    assert result["experiment_statistics"][0]["point_counts"] == [[1, 1, 1], [1, 0, 1], [0, 1, 2]]


def test_inconsistent_evaluation_grids_and_targets_fail_explicitly() -> None:
    reference = intervals(rows(64), [(0, 64, Direction.GROWTH)])
    data = sequences((reference,))
    gap = (reference[0].model_copy(update={"start": 1}),)
    with pytest.raises(ValueError, match="without overlaps or gaps"):
        evaluation.report(data, (gap,))
    with pytest.raises(ValueError, match="matching complete"):
        evaluation.report(data, ())
    data.labels[0] = 0
    with pytest.raises(ValueError, match="disagree"):
        evaluation.report(data, (reference,))


def test_invalid_scores_and_unknown_class_indices_are_rejected() -> None:
    evaluation.probabilities_valid(np.array([[0.2, 0.3, 0.5]]), 1)
    for invalid in [np.zeros((1, 3)), np.ones((1, 5)), np.array([[np.nan, 0.0, 1.0]])]:
        with pytest.raises(ValueError, match="invalid class scores"):
            evaluation.probabilities_valid(invalid, 1)
    with pytest.raises(ValueError, match="direction indices"):
        evaluation.scores(np.array([3]), np.array([0]))


def test_boundary_uncertainty_keeps_instruments_of_one_experiment_together() -> None:
    first = intervals(
        rows(96),
        [(0, 32, Direction.SIDEWAYS), (32, 64, Direction.GROWTH), (64, 96, Direction.DECLINE)],
    )
    second = intervals(
        rows(96, "B/Q"),
        [(0, 32, Direction.SIDEWAYS), (32, 64, Direction.GROWTH), (64, 96, Direction.DECLINE)],
    )
    missed = intervals(rows(96, "B/Q"), [(0, 96, Direction.SIDEWAYS)])
    data = sequences((first, second), ("shared-experiment", "shared-experiment"))
    result = evaluation.report(data, (first, missed), bootstrap_repetitions=100)
    assert result["boundary_metrics"]["f1"] == pytest.approx(2 / 3)
    assert result["confidence_intervals"]["boundary_f1_percentile_interval"] is None
    assert "two independent experiments" in result["confidence_intervals"]["limitation"]
    assert result["experiment_statistics"][0]["boundary_counts"] == [2, 0, 2]


def test_undefined_boundary_draws_do_not_become_zero_scores() -> None:
    observations = rows(96)
    flat = intervals(observations, [(0, 96, Direction.GROWTH)])
    change = intervals(observations, [(0, 48, Direction.GROWTH), (48, 96, Direction.DECLINE)])
    split = intervals(observations, [(0, 48, Direction.GROWTH), (48, 96, Direction.GROWTH)])
    for reference, predicted, expected_empty in (
        ((flat, change), (flat, change), (True, True)),
        ((flat, flat), (split, split), (False, True)),
        ((flat, flat), (flat, flat), (True, True)),
    ):
        report = evaluation.report(sequences(reference), predicted, bootstrap_repetitions=100)
        confidence = report["confidence_intervals"]
        for metric, empty in zip(("boundary_f1", "typed_boundary_f1"), expected_empty, strict=True):
            draws = confidence[f"{metric}_undefined_draws"]
            if empty:
                assert draws > 0
                assert confidence[f"{metric}_percentile_interval"] is None
            else:
                assert draws == 0
                # Predicted boundaries with no reference matches have a defined zero F1.
                assert confidence[f"{metric}_percentile_interval"] == [0.0, 0.0]
    assert report["confidence_intervals"]["typed_boundary_f1_undefined_draws"] == 100


def test_point_uncertainty_distinguishes_no_labels_from_no_correct_predictions() -> None:
    observations = rows(9)
    labeled = intervals(observations, [(0, 9, Direction.GROWTH)])
    empty = intervals(observations, [(0, 9, None)])
    for references, expected_undefined in (
        ((empty, empty), 100),
        ((labeled, labeled), 0),
        ((labeled, empty), None),
    ):
        report = evaluation.report(sequences(references), (empty, empty), bootstrap_repetitions=100)
        confidence = report["confidence_intervals"]
        missing = confidence["point_macro_f1_undefined_draws"]
        if expected_undefined is None:
            assert 0 < missing < 100
        else:
            assert missing == expected_undefined
        assert confidence["point_macro_f1_percentile_interval"] == (None if missing else [0, 0])
        assert report["point_metrics"]["defined"] is (expected_undefined != 100)


def test_duplicate_experiment_instrument_cannot_inflate_evaluation_support() -> None:
    reference = intervals(rows(64), [(0, 64, Direction.GROWTH)])
    data = sequences((reference, reference), ("same", "same"))
    with pytest.raises(ValueError, match="repeat an experiment/instrument"):
        evaluation.report(data, (reference, reference))


def test_typed_boundaries_require_both_directions_and_preserve_official_score() -> None:
    observations = rows(40)
    reference = intervals(observations, [(0, 20, Direction.SIDEWAYS), (20, 40, Direction.GROWTH)])
    wrong = intervals(observations, [(0, 20, Direction.DECLINE), (20, 40, Direction.GROWTH)])
    result = evaluation.report(sequences((reference,)), (wrong,), bootstrap_repetitions=0)
    assert result["boundary_metrics"]["f1"] == 1
    typed = result["typed_boundary_metrics"]
    assert typed["matched"] == 0 and typed["f1"] == 0
    assert typed["alignment"]["unmatched_reference_count"] == 1
    assert typed["alignment"]["signed_lag_steps"]["mean_steps"] is None


def test_typed_alignment_distinguishes_displacement_from_omitted_transitions() -> None:
    observations = rows(80)
    reference = intervals(observations, [(0, 20, Direction.SIDEWAYS), (20, 80, Direction.GROWTH)])
    later = intervals(observations, [(0, 30, Direction.SIDEWAYS), (30, 80, Direction.GROWTH)])
    result = evaluation.report(sequences((reference,)), (later,), bootstrap_repetitions=20)
    typed = result["typed_boundary_metrics"]
    assert typed["matched"] == 0
    alignment = typed["alignment"]
    assert alignment["matched"] == 1 and alignment["unmatched_reference_count"] == 0
    assert alignment["signed_lag_steps"]["mean_steps"] == 10
    assert alignment["signed_lag_steps"]["mean_absolute_steps"] == 10
    assert alignment["signed_lag_steps"]["late_count"] == 1
    assert result["experiment_statistics"][0]["typed_boundary_lags"] == [10]


def test_typed_boundary_matching_cannot_cross_reference_or_prediction_gaps() -> None:
    observations = rows(30)
    reference = intervals(
        observations,
        [
            (0, 10, Direction.SIDEWAYS),
            (10, 11, Direction.GROWTH),
            (11, 12, None),
            (12, 30, Direction.GROWTH),
        ],
    )
    predicted = intervals(
        observations,
        [
            (0, 11, Direction.SIDEWAYS),
            (11, 12, None),
            (12, 13, Direction.SIDEWAYS),
            (13, 30, Direction.GROWTH),
        ],
    )
    for expected, actual in ((reference, predicted), (predicted, reference)):
        result = evaluation.report(sequences((expected,)), (actual,), bootstrap_repetitions=0)
        assert result["boundary_metrics"]["matched"] == 1
        assert result["typed_boundary_metrics"]["matched"] == 0
        assert result["typed_boundary_metrics"]["alignment"]["matched"] == 0


def test_typed_alignment_maximizes_count_before_minimizing_absolute_lag() -> None:
    assert evaluation._aligned_lags([10, 14], [13, 18]) == [3, 4]
    assert evaluation._aligned_lags([10, 14], [13]) == [-1]
    assert evaluation._aligned_lags([13], [10, 14]) == [1]
    assert evaluation._aligned_lags([10, 20], [1, 11]) == [-9, -9]
    assert evaluation._boundary_matches([10, 20], [1, 11]) == 1


def test_alignment_preserves_global_cost_and_tie_order_when_skipping_transitions() -> None:
    assert evaluation._aligned_lags([5, 6], [1, 5, 20]) == [-4, -1]
    shorter, longer = [4, 12, 20], [0, 2, 6, 10, 14, 18, 22, 30]
    assert evaluation._aligned_lags(shorter, longer) == [2, 2, 2]
    assert evaluation._aligned_lags(longer, shorter) == [-2, -2, -2]


def test_redundant_same_direction_intervals_are_not_typed_transitions() -> None:
    observations = rows(40)
    reference = intervals(observations, [(0, 40, Direction.GROWTH)])
    predicted = intervals(observations, [(0, 20, Direction.GROWTH), (20, 40, Direction.GROWTH)])
    result = evaluation.report(sequences((reference,)), (predicted,), bootstrap_repetitions=0)
    assert result["boundary_metrics"]["predicted_count"] == 1
    assert result["typed_boundary_metrics"]["predicted_count"] == 0
    assert not result["typed_boundary_metrics"]["defined"]


def test_paired_bootstrap_identity_sign_swap_and_group_order() -> None:
    def counts(correct: int, missed: int) -> ExperimentCounts:
        directions = np.array([[correct, missed, missed]] * 3, dtype=np.int64)
        boundary = np.array([correct, missed, missed], dtype=np.int64)
        return directions, directions.copy(), boundary, boundary.copy()

    first = {"a": counts(2, 1), "b": counts(1, 3), "c": counts(2, 0)}
    second = {"c": counts(2, 0), "a": counts(3, 0), "b": counts(0, 4)}
    identity = evaluation.paired_bootstrap(first, first, repetitions=101)
    forward = evaluation.paired_bootstrap(first, second, repetitions=101)
    backward = evaluation.paired_bootstrap(second, first, repetitions=101)
    assert forward == evaluation.paired_bootstrap(
        dict(reversed(list(first.items()))), second, repetitions=101
    )
    for name, item in forward["metrics"].items():
        same = identity["metrics"][name]
        assert same["delta"] == 0 and same["percentile_interval"] == [0, 0]
        swapped = backward["metrics"][name]
        assert swapped["delta"] == -item["delta"]
        assert swapped["percentile_interval"] == [
            -item["percentile_interval"][1],
            -item["percentile_interval"][0],
        ]
        assert [
            item[key] for key in ("improved_experiments", "tied_experiments", "worse_experiments")
        ] == [1, 1, 1]
        assert item["delta"] == 0  # Equal pooled counts despite different experiment scores.
    with pytest.raises(ValueError, match="aligned"):
        evaluation.paired_bootstrap(first, {"other": counts(2, 1)}, repetitions=20)
    with pytest.raises(ValueError, match="populations"):
        evaluation.paired_bootstrap(first, {**second, "a": counts(3, 1)}, repetitions=20)


def test_sideways_interval_outcomes_follow_the_evaluation_matcher() -> None:
    from regimex.interval_errors import Outcome, sideways_intervals

    observations = rows(24)
    up, side, down = Direction.GROWTH, Direction.SIDEWAYS, Direction.DECLINE
    reference = intervals(observations, [(0, 8, up), (8, 20, side), (20, 24, down)])
    cases: list[tuple[list[tuple[int, int, Direction | None]], Outcome]] = [
        (
            [(0, 8, up), (8, 14, side), (14, 18, up), (18, 20, side), (20, 24, down)],
            Outcome.MATCHED,
        ),
        ([(0, 8, up), (8, 12, side), (12, 16, up), (16, 20, side), (20, 24, down)], Outcome.SPLIT),
        ([(0, 18, up), (18, 22, side), (22, 24, down)], Outcome.LOW_OVERLAP),
        ([(0, 20, up), (20, 24, down)], Outcome.ABSORBED),
        ([(0, 8, up), (8, 14, down), (14, 20, up), (20, 24, down)], Outcome.WRONG_DIRECTION),
        ([(0, 8, side), (8, 20, None), (20, 24, down)], Outcome.UNAVAILABLE),
    ]
    for bounds, outcome in cases:
        actual = intervals(observations, bounds)
        (item,) = sideways_intervals(reference, actual)
        assert item.outcome == outcome
        assert item.reference == reference[1]
        assert all(p.stop > 8 and p.start < 20 for p in item.overlapping)
        assert (
            item.best_sideways_iou == 0.5
            if outcome == Outcome.MATCHED
            else item.best_sideways_iou < 0.5
        )


def test_spanning_prediction_can_merge_two_sideways_references() -> None:
    from regimex.interval_errors import Outcome, sideways_intervals

    observations = rows(12)
    up, side, down = Direction.GROWTH, Direction.SIDEWAYS, Direction.DECLINE
    reference = intervals(
        observations, [(0, 3, up), (3, 6, side), (6, 8, down), (8, 11, side), (11, 12, down)]
    )
    actual = intervals(observations, [(0, 3, up), (3, 11, side), (11, 12, down)])
    records = sideways_intervals(reference, actual)
    assert [(record.outcome, record.best_sideways_iou) for record in records] == [
        (Outcome.MERGED, 3 / 8)
    ] * 2
    with pytest.raises(ValueError, match="cover the full sequence"):
        sideways_intervals(reference, actual[:-1])


def test_sideways_evidence_pairs_steps_and_keeps_tied_scores_separate() -> None:
    from regimex.interval_errors import sideways_evidence

    probabilities = np.array(
        [
            [0.6, 0.3, 0.1],
            [0.7, 0.2, 0.1],
            [0.1, 0.2, 0.7],
            [0.1, 0.8, 0.1],
            [0.5, 0.5, 0.0],
        ]
    )
    evidence = sideways_evidence(probabilities, np.array([0, 1, 0, -1, 1]))
    assert evidence.raw_leaders == {"sideways": 2, "growth": 1, "decline": 1, "tied": 1}
    assert evidence.decoded_sideways_steps == 2
    assert evidence.retained_sideways_steps == 1
    with pytest.raises(ValueError, match="invalid class scores"):
        sideways_evidence(probabilities, np.array([0]))
