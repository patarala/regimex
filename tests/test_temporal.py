"""Price representation isolation, missingness and padded temporal inference."""

from dataclasses import replace

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")

from torch import Tensor

from regimex.contracts import Observation
from regimex.features import DoubleArray, FloatArray, compute_features
from regimex.model_registry import TCNParameters, TrainingPolicy, TrainingRuntime
from regimex.prediction_inputs import (
    PredictionInputs,
    PredictionSequenceInfo,
    PreparedPredictionInputs,
)
from regimex.temporal import (
    TEMPORAL_NAMES,
    TEMPORAL_NUMERIC_NAMES,
    TCNClassifier,
    TemporalPreprocessor,
    _boundary_weights,
    _supervised_loss,
    fit_tcn,
    predict_tcn,
    temporal_features,
)


def sequence(prices: list[float | None], name: str = "example") -> PredictionInputs:
    rows = tuple(
        Observation(
            instrument_id="A/Q",
            step=step,
            last_price=price,
            trade_count=None,
            trade_volume=None,
            buy_order_count=None,
            buy_order_volume=None,
            sell_order_count=None,
            sell_order_volume=None,
            best_bid=None,
            best_ask=None,
        )
        for step, price in enumerate(prices)
    )
    return PredictionInputs(
        compute_features(rows),
        np.array([0, len(rows)], dtype=np.int64),
        (PredictionSequenceInfo(experiment_id=name, instrument_id="A/Q", step_count=len(rows)),),
    )


def combine(left: PredictionInputs, right: PredictionInputs) -> PredictionInputs:
    return PredictionInputs(
        np.concatenate((left.values, right.values)),
        np.array([0, len(left.values), len(left.values) + len(right.values)], dtype=np.int64),
        (*left.info, *right.info),
    )


def test_temporal_features_ignore_activity_and_other_experiments() -> None:
    first = sequence([float(100 * np.exp(step * 0.001)) for step in range(64)])
    expected = temporal_features(first)
    changed_values = first.values.copy()
    changed_values[:, 2:11] = 1000
    changed_values[:, 14:23] = 0
    changed = replace(first, values=changed_values)
    np.testing.assert_array_equal(temporal_features(changed), expected)
    second = sequence([float(50 * np.exp(-step * 0.02)) for step in range(80)], "other")
    np.testing.assert_array_equal(temporal_features(combine(first, second))[:64], expected)
    assert expected.shape == (64, len(TEMPORAL_NAMES))


def test_price_context_uses_actual_time_and_preserves_missing_observations() -> None:
    prices = [float(100 * np.exp(step * 0.001)) if step % 3 else None for step in range(64)]
    values = temporal_features(sequence(prices))
    slope_index = TEMPORAL_NAMES.index("context_32_slope_change_asinh")
    fit_index = TEMPORAL_NAMES.index("context_32_linear_fit")
    coverage_index = TEMPORAL_NAMES.index("context_32_coverage")
    assert values[32, slope_index] == pytest.approx(np.arcsinh(0.031 / 0.01), rel=1e-6)
    assert values[32, fit_index] == pytest.approx(1.0, abs=1e-6)
    assert 0.5 < values[32, coverage_index] < 1
    assert np.isnan(values[0, 0])
    assert values[0, len(TEMPORAL_NUMERIC_NAMES)] == 1
    flat = temporal_features(sequence([100.0] * 64))
    assert flat[32, slope_index] == 0
    assert flat[32, TEMPORAL_NAMES.index("context_32_slope_change_asinh_missing")] == 0
    missing = temporal_features(sequence([None] * 64))
    assert np.isnan(missing[32, slope_index])
    assert missing[32, coverage_index] == 0
    assert missing[32, TEMPORAL_NAMES.index("context_32_slope_change_asinh_missing")] == 1


def test_temporal_preprocessing_restores_training_state_and_masks() -> None:
    values = temporal_features(sequence([100.0, None, 102.0, 103.0]))
    preprocessing = TemporalPreprocessor.fit(values)
    original_mean = preprocessing.mean.copy()
    changed: FloatArray = values.copy()
    changed[:, 0] = 100
    restored = TemporalPreprocessor(
        preprocessing.fill.copy(), preprocessing.mean.copy(), preprocessing.scale.copy()
    )
    np.testing.assert_array_equal(preprocessing.transform(changed), restored.transform(changed))
    np.testing.assert_array_equal(preprocessing.mean, original_mean)
    transformed = preprocessing.transform(values)
    assert np.isfinite(transformed).all()
    np.testing.assert_array_equal(
        transformed[:, len(TEMPORAL_NUMERIC_NAMES) :], values[:, len(TEMPORAL_NUMERIC_NAMES) :]
    )


def test_tcn_padding_cannot_influence_observed_sequence() -> None:
    torch.set_num_threads(1)
    torch.manual_seed(11)
    model = TCNClassifier(5, 8, 0.15).eval()
    real = torch.randn(1, 47, 5)
    padded = torch.cat((real, torch.randn(1, 41, 5) * 100), dim=1)
    with torch.inference_mode():
        expected = model(real, torch.tensor([47]))
        actual = model(padded, torch.tensor([47]))[:, :47]
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


def test_tcn_predictions_are_independent_of_other_batch_members() -> None:
    torch.set_num_threads(1)
    torch.manual_seed(11)
    first = sequence([100.0 + step / 20 for step in range(37)])
    second = sequence([200.0 - step / 5 for step in range(81)], "other")
    raw = combine(first, second)
    values = temporal_features(raw)
    prepared = PreparedPredictionInputs(
        TemporalPreprocessor.fit(values).transform(values), raw.offsets, raw.info
    )
    model = TCNClassifier(len(TEMPORAL_NAMES), 8, 0.15)
    single = predict_tcn(model, prepared, batch_size=1)
    batched = predict_tcn(model, prepared, batch_size=2)
    np.testing.assert_allclose(single, batched, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(batched.sum(axis=1), 1, rtol=1e-12)


def test_tcn_reproduces_selected_validation_checkpoint() -> None:
    parts = [
        sequence([float(100 * np.exp(slope * step)) for step in range(32)], str(label))
        for label, slope in enumerate((0, 0.002, -0.002))
    ]
    raw = PredictionInputs(
        np.concatenate([part.values for part in parts]),
        np.arange(0, 97, 32, dtype=np.int64),
        tuple(part.info[0] for part in parts),
    )
    targets = np.repeat(np.arange(3), 32).astype(np.int64)
    values = temporal_features(raw)
    data = PreparedPredictionInputs(
        TemporalPreprocessor.fit(values).transform(values), raw.offsets, raw.info
    )
    parameters = TCNParameters(channels=4)
    policy = TrainingPolicy(
        optimizer="adamw", seeds=(11,), batch_size=3, epochs=3, validation_interval=1
    )
    runtime = TrainingRuntime(threads=4)
    observed = []

    def scorer(probabilities: DoubleArray) -> float:
        observed.append(probabilities.copy())
        return 1 / len(observed)

    first, history, seed = fit_tcn(
        data, targets, data, parameters=parameters, policy=policy, runtime=runtime, scorer=scorer
    )
    np.testing.assert_array_equal(predict_tcn(first, data), observed[0])
    observed.clear()
    second, repeated_history, repeated_seed = fit_tcn(
        data, targets, data, parameters=parameters, policy=policy, runtime=runtime, scorer=scorer
    )
    np.testing.assert_array_equal(
        predict_tcn(second, data),
        predict_tcn(first, data),
    )
    assert (history, seed) == (repeated_history, repeated_seed)


def loss_and_gradient(
    logits: list[list[list[float]]],
    targets: list[list[int]],
    smoothing: float,
    weights: list[list[float]] | None = None,
) -> tuple[Tensor, Tensor]:
    values = torch.tensor(logits, dtype=torch.float64, requires_grad=True)
    labels = torch.tensor(targets, dtype=torch.int64)
    sample_weights = (
        torch.ones_like(labels, dtype=torch.float64)
        if weights is None
        else torch.tensor(weights, dtype=torch.float64)
    )
    loss = _supervised_loss(
        values,
        labels,
        torch.tensor([0.75, 1.25, 1.5], dtype=torch.float64),
        sample_weights,
        smoothing,
    )
    gradient = torch.autograd.grad(loss, values)[0]
    return loss.detach(), gradient


def test_smoothing_does_not_pull_back_a_known_direction_switch() -> None:
    logits = [[[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]]
    unsmoothed = loss_and_gradient(logits, [[0, 1]], 0.0)
    smoothed = loss_and_gradient(logits, [[0, 1]], 0.05)
    torch.testing.assert_close(smoothed[0], unsmoothed[0], rtol=0, atol=0)
    torch.testing.assert_close(smoothed[1], unsmoothed[1], rtol=0, atol=0)


def test_smoothing_still_penalizes_variation_inside_one_direction() -> None:
    logits = [[[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]]
    unsmoothed = loss_and_gradient(logits, [[0, 0]], 0.0)
    smoothed = loss_and_gradient(logits, [[0, 0]], 0.05)
    assert smoothed[0] > unsmoothed[0]
    smoothing_gradient = smoothed[1] - unsmoothed[1]
    torch.testing.assert_close(smoothing_gradient[0, 0], torch.zeros(3, dtype=torch.float64))
    assert bool(smoothing_gradient[0, 1].abs().sum() > 0)


def test_unknown_and_padded_points_have_no_loss_or_cross_gap_smoothing() -> None:
    targets = [[0, -1, 1, -1]]
    logits = [[[2.0, 0.0, 0.0], [8.0, -5.0, 4.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]]]
    loss, gradient = loss_and_gradient(logits, targets, 0.05)
    reference, _ = loss_and_gradient(logits, targets, 0.0)
    torch.testing.assert_close(loss, reference, rtol=0, atol=0)
    torch.testing.assert_close(gradient[:, [1, 3]], torch.zeros((1, 2, 3), dtype=torch.float64))
    logits[0][1] = [-2.0, 10.0, -30.0]
    changed, _ = loss_and_gradient(logits, targets, 0.05)
    torch.testing.assert_close(changed, loss, rtol=0, atol=0)


def test_smoothing_does_not_connect_separate_sequences() -> None:
    logits = [[[2.0, 0.0, 0.0]], [[0.0, 2.0, 0.0]]]
    unsmoothed = loss_and_gradient(logits, [[0], [1]], 0.0)
    smoothed = loss_and_gradient(logits, [[0], [1]], 0.05)
    torch.testing.assert_close(smoothed[0], unsmoothed[0], rtol=0, atol=0)
    torch.testing.assert_close(smoothed[1], unsmoothed[1], rtol=0, atol=0)


def test_sample_weights_normalize_effective_class_weight() -> None:
    logits = [[[2.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 2.0]]]
    targets = [[0, 1, -1]]
    loss, gradient = loss_and_gradient(logits, targets, 0.0, [[2.0, 1.0, 100.0]])
    scaled_loss, scaled_gradient = loss_and_gradient(logits, targets, 0.0, [[20.0, 10.0, 1000.0]])
    torch.testing.assert_close(loss, scaled_loss)
    torch.testing.assert_close(gradient, scaled_gradient)
    class0, _ = loss_and_gradient([[logits[0][0]]], [[0]], 0.0)
    class1, _ = loss_and_gradient([[logits[0][1]]], [[1]], 0.0)
    expected = (2 * 0.75 * class0 + 1.25 * class1) / (2 * 0.75 + 1.25)
    torch.testing.assert_close(loss, expected)


def test_no_eligible_weight_fails_explicitly() -> None:
    with pytest.raises(ValueError, match="positive eligible target weight"):
        loss_and_gradient([[[0.0, 0.0, 0.0]]], [[-1]], 0.05)


@pytest.mark.parametrize(
    ("targets", "offsets", "expected"),
    [
        ([0] * 8 + [1] * 8, [0, 16], [1] * 4 + [2] * 8 + [1] * 4),
        ([0] * 8, [0, 8], [1] * 8),
        ([0] * 8 + [1] * 8, [0, 8, 16], [1] * 16),
        ([0, 0, -1, 1, 1], [0, 5], [1] * 5),
        ([0, 1, -1, 2, 2], [0, 5], [2, 2, 1, 1, 1]),
        ([0, 1, 0], [0, 3], [2, 2, 2]),
        ([0, 0, 0, 0, 0, 1], [0, 6], [1, 2, 2, 2, 2, 2]),
    ],
)
def test_boundary_weights_respect_exact_neighborhoods_and_available_blocks(
    targets: list[int], offsets: list[int], expected: list[int]
) -> None:
    weights = _boundary_weights(
        np.array(targets, dtype=np.int64), np.array(offsets, dtype=np.int64)
    )
    np.testing.assert_array_equal(weights, np.array(expected, dtype=np.float32))
