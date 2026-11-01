"""Preserved aggregate logistic calculations, independent of Torch and service I/O."""

import logging
import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from pydantic import JsonValue

from regimex.directions import DIRECTION_NAMES
from regimex.features import DoubleArray, FloatArray, IntArray
from regimex.prediction_inputs import PreparedPredictionInputs, require_prepared_inputs

logger = logging.getLogger(__name__)
Scorer = Callable[[DoubleArray], float]
_BASELINE_BATCH_ROWS = 8192


def _sequence_aggregates(values: FloatArray, context_sizes: tuple[int, ...]) -> FloatArray:
    width = values.shape[1] * 4
    result = np.empty((len(values), width * (len(context_sizes) + 1)), dtype=np.float32)
    for index, size in enumerate(context_sizes):
        for step in range(len(values)):
            left = max(0, step - size // 2)
            right = min(len(values), step + (size + 1) // 2)
            block = values[left:right]
            result[step, index * width : (index + 1) * width] = np.concatenate(
                (block.mean(axis=0), block.std(axis=0), block.min(axis=0), block.max(axis=0))
            )
    result[:, -width:] = np.concatenate(
        (values.mean(axis=0), values.std(axis=0), values.min(axis=0), values.max(axis=0))
    )
    return result


def aggregate(sequences: PreparedPredictionInputs, context_sizes: tuple[int, ...]) -> FloatArray:
    """Mean, std, min and max in centered contexts and the complete sequence.

    Each instrument is summarized independently. Windows stop at sequence edges;
    labels and reference boundaries do not enter the aggregates.
    """
    require_prepared_inputs(sequences)
    width = sequences.values.shape[1] * 4
    result = np.empty((len(sequences.values), width * (len(context_sizes) + 1)), dtype=np.float32)
    for start, stop in zip(sequences.offsets[:-1], sequences.offsets[1:], strict=True):
        result[start:stop] = _sequence_aggregates(sequences.values[start:stop], context_sizes)
    return result


def _training_aggregates(
    sequences: PreparedPredictionInputs, targets: IntArray, context_sizes: tuple[int, ...]
) -> DoubleArray:
    eligible = targets >= 0
    width = sequences.values.shape[1] * 4 * (len(context_sizes) + 1)
    result = np.empty((int(eligible.sum()), width), dtype=np.float64)
    position = 0
    for start, stop in zip(sequences.offsets[:-1], sequences.offsets[1:], strict=True):
        selected = eligible[start:stop]
        count = int(selected.sum())
        if count:
            result[position : position + count] = _sequence_aggregates(
                sequences.values[start:stop], context_sizes
            )[selected]
            position += count
    return result


@dataclass(frozen=True)
class LinearModel:
    coefficients: DoubleArray
    intercept: DoubleArray
    mean: DoubleArray
    scale: DoubleArray
    context_sizes: tuple[int, ...]

    def predict(self, sequences: PreparedPredictionInputs) -> DoubleArray:
        return self.predict_aggregates(aggregate(sequences, self.context_sizes))

    def predict_aggregates(self, values: FloatArray | DoubleArray) -> DoubleArray:
        result = np.empty((len(values), len(DIRECTION_NAMES)), dtype=np.float64)
        for start in range(0, len(values), _BASELINE_BATCH_ROWS):
            stop = start + _BASELINE_BATCH_ROWS
            normalized = (values[start:stop] - self.mean) / self.scale
            logits = normalized @ self.coefficients.T + self.intercept
            logits -= logits.max(axis=1, keepdims=True)
            exp = np.exp(logits)
            result[start:stop] = exp / exp.sum(axis=1, keepdims=True)
        return result


def fit_baseline(
    training: PreparedPredictionInputs,
    targets: IntArray,
    validation: PreparedPredictionInputs,
    *,
    regularization_candidates: tuple[float, ...],
    seed: int,
    threads: int,
    context_sizes: tuple[int, ...],
    scorer: Scorer,
    check_budget: Callable[[], None] = lambda: None,
) -> tuple[LinearModel, list[JsonValue], float]:
    # Training-only dependencies remain absent from restored inference processes.
    from sklearn.exceptions import ConvergenceWarning  # type: ignore[import-untyped]
    from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
    from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]
    from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]

    require_prepared_inputs(training)
    require_prepared_inputs(validation)
    eligible = targets >= 0
    y = targets[eligible]
    if not np.array_equal(np.unique(y), np.arange(len(DIRECTION_NAMES))):
        raise ValueError("baseline requires all three training directions")
    logger.info("Building baseline aggregates for %s eligible training rows", len(y))
    x = _training_aggregates(training, targets, context_sizes)
    # All eligible rows contribute. Batching bounds scaling memory; only the
    # floating-point reduction order differs from a single full-matrix fit.
    scaler = StandardScaler(copy=False)
    for start in range(0, len(x), _BASELINE_BATCH_ROWS):
        scaler.partial_fit(x[start : start + _BASELINE_BATCH_ROWS])
    for start in range(0, len(x), _BASELINE_BATCH_ROWS):
        scaler.transform(x[start : start + _BASELINE_BATCH_ROWS])
    valid = aggregate(validation, context_sizes).astype(np.float64)
    best: LinearModel | None = None
    best_score = -1.0
    best_c = 0.0
    history: list[JsonValue] = []
    for regularization in regularization_candidates:
        check_budget()
        logger.info("Fitting baseline C=%s on %s rows and %s aggregates", regularization, *x.shape)
        fitted = LogisticRegression(
            C=regularization,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=2000,
            random_state=seed,
        )
        with threadpool_limits(limits=threads), warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            try:
                fitted.fit(x, y)
            except ConvergenceWarning as exc:
                raise ValueError(f"baseline did not converge for C={regularization}") from exc
        model = LinearModel(
            np.asarray(fitted.coef_, dtype=np.float64),
            np.asarray(fitted.intercept_, dtype=np.float64),
            np.asarray(scaler.mean_, dtype=np.float64),
            np.asarray(scaler.scale_, dtype=np.float64),
            context_sizes,
        )
        probabilities = model.predict_aggregates(valid)
        for start in range(0, len(valid), _BASELINE_BATCH_ROWS):
            stop = start + _BASELINE_BATCH_ROWS
            expected = fitted.predict_proba(scaler.transform(valid[start:stop], copy=True))
            if not np.allclose(expected, probabilities[start:stop], atol=1e-10, rtol=1e-10):
                raise ValueError("restored baseline probabilities differ from scikit-learn")
        check_budget()
        score = scorer(probabilities)
        if not np.isfinite(score):
            raise ValueError("nonfinite baseline validation score")
        history.append({"regularization": regularization, "validation_segment_macro_f1": score})
        logger.info("baseline C=%s validation_segment_macro_f1=%.4f", regularization, score)
        if score > best_score:
            best, best_score, best_c = model, score, regularization
    assert best is not None
    return best, history, best_c
