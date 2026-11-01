"""Observation features, missingness, preprocessing, and aggregate context isolation."""

import pytest

np = pytest.importorskip("numpy")

from regimex.contracts import Observation
from regimex.features import FEATURE_NAMES, NUMERIC_NAMES, Preprocessor, compute_features
from regimex.prediction_inputs import PredictionSequenceInfo, PreparedPredictionInputs


def row(step: int, price: float | None) -> Observation:
    return Observation(
        instrument_id="A/Q",
        step=step,
        last_price=price,
        trade_count=0,
        trade_volume=0.0,
        buy_order_count=None,
        buy_order_volume=None,
        sell_order_count=0,
        sell_order_volume=0.0,
        best_bid=None,
        best_ask=None,
    )


def test_known_returns_and_missing_masks() -> None:
    values = compute_features((row(0, 100.0), row(1, 110.0), row(2, 121.0)))
    price = FEATURE_NAMES.index("relative_log_price")
    returns = FEATURE_NAMES.index("log_return")
    trades = FEATURE_NAMES.index("log_trade_count")
    buys = FEATURE_NAMES.index("log_buy_count")
    np.testing.assert_allclose(values[:, price], [0.0, np.log(1.1), np.log(1.21)], rtol=1e-6)
    np.testing.assert_allclose(values[1:, returns], np.log(1.1), rtol=1e-6)
    assert np.isnan(values[0, returns])
    assert values[0, len(NUMERIC_NAMES) + returns] == 1
    assert values[0, trades] == 0
    assert values[0, len(NUMERIC_NAMES) + trades] == 0
    assert np.isnan(values[0, buys])
    assert values[0, len(NUMERIC_NAMES) + buys] == 1


def test_returns_do_not_bridge_missing_observations() -> None:
    values = compute_features((row(0, 100.0), row(1, None), row(2, 121.0), row(3, 133.1)))
    returns = FEATURE_NAMES.index("log_return")
    assert np.isnan(values[:3, returns]).all()
    assert values[3, returns] == pytest.approx(np.log(1.1))
    assert values[2, FEATURE_NAMES.index("relative_log_price")] == pytest.approx(np.log(1.21))
    with pytest.raises(ValueError, match="one ordered contiguous"):
        compute_features((row(0, 100.0), row(2, 110.0)))
    with pytest.raises(ValueError, match="one ordered contiguous"):
        compute_features((row(0, 100.0), row(1, 110.0).model_copy(update={"instrument_id": "B/Q"})))


def test_large_finite_observations_do_not_overflow_ratios() -> None:
    observation = Observation(
        instrument_id="A/Q",
        step=0,
        last_price=1e308,
        trade_count=1,
        trade_volume=1e308,
        buy_order_count=1,
        buy_order_volume=1e308,
        sell_order_count=1,
        sell_order_volume=9e307,
        best_bid=9e307,
        best_ask=1e308,
    )
    values = compute_features((observation,))
    assert values[0, FEATURE_NAMES.index("volume_imbalance")] == pytest.approx(1 / 19)
    assert values[0, FEATURE_NAMES.index("relative_spread")] == pytest.approx(2 / 19)


def test_preprocessing_retains_missingness_and_uses_training_only() -> None:
    training = compute_features(tuple(row(i, 100.0 + i) for i in range(20)))
    first = Preprocessor.fit(training)
    validation = compute_features(tuple(row(i, 1000.0 + 100 * i) for i in range(20)))
    initial = (first.fill.copy(), first.mean.copy(), first.scale.copy())
    first.transform(validation)
    for original, current in zip(initial, (first.fill, first.mean, first.scale), strict=True):
        np.testing.assert_array_equal(original, current)
    transformed = first.transform(training)
    assert np.isfinite(transformed).all()
    np.testing.assert_array_equal(
        transformed[:, len(NUMERIC_NAMES) :], training[:, len(NUMERIC_NAMES) :]
    )


def test_aggregate_contexts_never_cross_instruments_or_experiments() -> None:
    from regimex.baseline import aggregate

    values = np.vstack(
        (
            np.full((5, len(FEATURE_NAMES)), 7, dtype=np.float32),
            np.full((5, len(FEATURE_NAMES)), 100, dtype=np.float32),
        )
    )
    infos = tuple(
        PredictionSequenceInfo(experiment_id=f"exp{i}", instrument_id="A/Q", step_count=5)
        for i in range(2)
    )
    sequence = PreparedPredictionInputs(values, np.array([0, 5, 10], dtype=np.int64), infos)
    aggregates = aggregate(sequence, (3, 16))
    for start, stop, constant in ((0, 5, 7), (5, 10, 100)):
        separate = PreparedPredictionInputs(
            values[start:stop], np.array([0, 5], dtype=np.int64), (infos[start // 5],)
        )
        np.testing.assert_array_equal(aggregates[start:stop], aggregate(separate, (3, 16)))
        for offset in range(0, aggregates.shape[1], 4 * len(FEATURE_NAMES)):
            assert (aggregates[start:stop, offset : offset + len(FEATURE_NAMES)] == constant).all()


def test_baseline_full_context_is_order_invariant_and_has_no_target_input() -> None:
    from regimex.baseline import aggregate

    values = np.arange(8 * len(FEATURE_NAMES), dtype=np.float32).reshape(8, -1)
    info = (PredictionSequenceInfo(experiment_id="exp", instrument_id="A/Q", step_count=8),)
    sequence = PreparedPredictionInputs(values, np.array([0, 8], dtype=np.int64), info)
    shuffled = PreparedPredictionInputs(values[[7, 0, 4, 2, 5, 1, 6, 3]], sequence.offsets, info)
    original = aggregate(sequence, ())
    np.testing.assert_allclose(original, aggregate(shuffled, ()), rtol=1e-6)
    np.testing.assert_array_equal(original, np.repeat(original[:1], 8, axis=0))


def test_batched_baseline_matches_full_fit_without_changing_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("sklearn")
    from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
    from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
    from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

    from regimex import baseline
    from regimex.features import IntArray

    rng = np.random.default_rng(17)

    def sequences(lengths: tuple[int, ...]) -> tuple[PreparedPredictionInputs, IntArray]:
        labels = np.concatenate(
            [np.full(length, index % 3, dtype=np.int64) for index, length in enumerate(lengths)]
        )
        offsets = np.r_[0, np.cumsum(lengths)].astype(np.int64)
        values = rng.normal(size=(len(labels), 4)).astype(np.float32)
        values[:, 0] += labels
        labels[offsets[:-1]] = -1
        inputs = PreparedPredictionInputs(
            values,
            offsets,
            tuple(
                PredictionSequenceInfo(
                    experiment_id=f"exp-{index}", instrument_id="A/Q", step_count=length
                )
                for index, length in enumerate(lengths)
            ),
        )

        return inputs, labels

    training, targets = sequences((17, 29, 41, 23, 31, 37))
    validation, _ = sequences((19, 27, 35))
    original_training = training.values.copy()
    original_validation = validation.values.copy()
    contexts = (4, 16)
    x = baseline.aggregate(training, contexts)[targets >= 0].astype(np.float64)
    valid = baseline.aggregate(validation, contexts).astype(np.float64)
    expected = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=1.0, class_weight="balanced", solver="lbfgs", max_iter=2000, random_state=11
                ),
            ),
        ]
    )
    expected.fit(x, targets[targets >= 0])
    monkeypatch.setattr(baseline, "_BASELINE_BATCH_ROWS", 13)
    fitted, _, selected = baseline.fit_baseline(
        training,
        targets,
        validation,
        regularization_candidates=(1.0,),
        seed=11,
        threads=1,
        context_sizes=contexts,
        scorer=lambda _: 0.5,
    )
    assert selected == 1.0
    np.testing.assert_allclose(fitted.mean, expected.named_steps["scale"].mean_, atol=1e-12)
    np.testing.assert_allclose(fitted.scale, expected.named_steps["scale"].scale_, atol=1e-12)
    probabilities = expected.predict_proba(valid)
    np.testing.assert_allclose(
        fitted.predict(validation),
        probabilities,
        atol=1e-10,
        rtol=1e-10,
    )
    for batch_size in (1, 7, 8192):
        monkeypatch.setattr(baseline, "_BASELINE_BATCH_ROWS", batch_size)
        np.testing.assert_allclose(
            fitted.predict_aggregates(valid), probabilities, atol=1e-10, rtol=1e-10
        )
    np.testing.assert_array_equal(training.values, original_training)
    np.testing.assert_array_equal(validation.values, original_validation)
    np.testing.assert_array_equal(
        valid,
        baseline.aggregate(validation, contexts).astype(np.float64),
    )
