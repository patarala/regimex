"""Logistic regression on per-instrument window summaries."""

import logging
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from itertools import chain
from typing import cast

import numpy as np
from pydantic import JsonValue

from regimex.directions import DIRECTION_NAMES
from regimex.features import DoubleArray, FloatArray, IntArray
from regimex.model_contracts import HistoryCallback
from regimex.model_registry import SummaryMode
from regimex.prediction_inputs import PreparedPredictionInputs, require_prepared_inputs

logger = logging.getLogger(__name__)
Scorer = Callable[[DoubleArray], float]
AccumulationDtype = type[np.float32] | type[np.float64]
_BASELINE_BATCH_ROWS = 8192
# Bound work arrays for masked reductions and variance to a batch of windows.
_WINDOW_BATCH_ELEMENTS = 65_536


def _statistics(
    block: FloatArray, observed_only: bool, accumulation_dtype: AccumulationDtype
) -> FloatArray:
    """Reduce the time axis of one window or a batch of equal-length windows."""
    if not observed_only:
        mean = cast(
            FloatArray | DoubleArray, block.mean(axis=-2, dtype=accumulation_dtype, keepdims=True)
        )
        return np.concatenate(
            (
                mean.squeeze(axis=-2),
                block.std(axis=-2, dtype=accumulation_dtype, mean=mean),
                block.min(axis=-2),
                block.max(axis=-2),
            ),
            axis=-1,
            dtype=np.float32,
        )

    # Numeric channels precede their missingness indicators. Indicators remain
    # observed even when their corresponding numeric channel is absent.
    channels = block.shape[-1] // 2
    known = np.ones(block.shape, dtype=np.bool_)
    known[..., :channels] = block[..., channels:] == 0
    count = known.sum(axis=-2)
    denominator = np.maximum(1, count)
    observed_mean = block.sum(axis=-2, where=known, dtype=np.float64) / denominator
    residual = block - np.expand_dims(observed_mean, -2)
    np.square(residual, out=residual)
    variance = residual.sum(axis=-2, where=known) / denominator
    minimum = block.min(axis=-2, where=known, initial=np.inf)
    maximum = block.max(axis=-2, where=known, initial=-np.inf)
    # Empty numeric windows use zero in standardized feature space. Their
    # missingness channels retain the distinction from genuine observed zeros.
    return np.concatenate(
        (
            observed_mean,
            np.sqrt(variance),
            np.where(count, minimum, 0),
            np.where(count, maximum, 0),
        ),
        axis=-1,
        dtype=np.float32,
    )


def _fill_sequence_aggregates(
    values: FloatArray,
    context_sizes: tuple[int, ...],
    summaries: SummaryMode,
    steps: range | IntArray,
    output: FloatArray | DoubleArray,
    accumulation_dtype: AccumulationDtype,
    check_budget: Callable[[], None] | None,
) -> None:
    width = values.shape[1] * 4
    centers = np.asarray(steps, dtype=np.int64)
    # Even a window centered on the first or last step covers the full sequence at this size.
    whole_sequence_size = 2 * len(values) - 1
    whole = None
    if summaries.whole_sequence or any(size >= whole_sequence_size for size in context_sizes):
        if check_budget is not None:
            check_budget()
        whole = _statistics(values, summaries.observed_only, accumulation_dtype)
        if summaries.whole_sequence:
            output[:, -width:] = whole
    for index, size in enumerate(context_sizes):
        columns = slice(index * width, (index + 1) * width)
        if whole is not None and size >= whole_sequence_size:
            if check_budget is not None:
                check_budget()
            output[:, columns] = whole
            continue
        left_width, right_width = size // 2, (size + 1) // 2
        # Only fully contained windows have the common shape needed for a batched reduction.
        first, last = centers.searchsorted([left_width, len(values) - right_width + 1])
        last = max(first, last)
        for row in chain(range(first), range(last, len(centers))):
            if check_budget is not None and row % 128 == 0:
                check_budget()
            step = centers[row]
            left = max(0, step - left_width)
            right = min(len(values), step + right_width)
            output[row, columns] = _statistics(
                values[left:right], summaries.observed_only, accumulation_dtype
            )
        if first < last:
            windows = np.lib.stride_tricks.sliding_window_view(values, size, axis=0).swapaxes(1, 2)
            batch_elements = min(values.size, _WINDOW_BATCH_ELEMENTS)
            batch_size = max(1, min(128, batch_elements // (size * values.shape[1])))
            for start in range(first, last, batch_size):
                if check_budget is not None:
                    check_budget()
                stop = min(start + batch_size, last)
                starts = centers[start:stop] - left_width
                if starts[-1] - starts[0] + 1 == len(starts):
                    batch = windows[starts[0] : starts[-1] + 1]
                else:
                    batch = windows[starts]
                output[start:stop, columns] = _statistics(
                    batch, summaries.observed_only, accumulation_dtype
                )


def aggregate(
    sequences: PreparedPredictionInputs,
    context_sizes: tuple[int, ...],
    summaries: SummaryMode = SummaryMode.LOCAL_AND_GLOBAL,
    *,
    accumulation_dtype: AccumulationDtype = np.float64,
    check_budget: Callable[[], None] | None = None,
) -> FloatArray:
    """Mean, std, min and max in the configured per-instrument windows.

    Each instrument is summarized independently. Windows stop at sequence edges;
    labels and reference boundaries do not enter the aggregates.
    """
    require_prepared_inputs(sequences)
    width = summaries.width(sequences.values.shape[1], context_sizes)
    result = np.empty((len(sequences.values), width), dtype=np.float32)
    for start, stop in zip(sequences.offsets[:-1], sequences.offsets[1:], strict=True):
        values = sequences.values[start:stop]
        _fill_sequence_aggregates(
            values,
            context_sizes,
            summaries,
            range(len(values)),
            result[start:stop],
            accumulation_dtype,
            check_budget,
        )
    return result


def _training_aggregates(
    sequences: PreparedPredictionInputs,
    targets: IntArray,
    context_sizes: tuple[int, ...],
    summaries: SummaryMode,
    check_budget: Callable[[], None],
) -> DoubleArray:
    eligible = targets >= 0
    width = summaries.width(sequences.values.shape[1], context_sizes)
    result = np.empty((int(eligible.sum()), width), dtype=np.float64)
    position = 0
    for start, stop in zip(sequences.offsets[:-1], sequences.offsets[1:], strict=True):
        steps = np.flatnonzero(eligible[start:stop])
        if len(steps):
            end = position + len(steps)
            # Unlabeled neighbors still contribute to a labeled step's context.
            _fill_sequence_aggregates(
                sequences.values[start:stop],
                context_sizes,
                summaries,
                steps,
                result[position:end],
                np.float64,
                check_budget,
            )
            position = end
    return result


@dataclass(frozen=True)
class LinearModel:
    coefficients: DoubleArray
    intercept: DoubleArray
    mean: DoubleArray
    scale: DoubleArray
    context_sizes: tuple[int, ...]
    summaries: SummaryMode = SummaryMode.LOCAL_AND_GLOBAL
    accumulation_dtype: AccumulationDtype = np.float64

    def predict(self, sequences: PreparedPredictionInputs) -> DoubleArray:
        return self.predict_aggregates(
            aggregate(
                sequences,
                self.context_sizes,
                self.summaries,
                accumulation_dtype=self.accumulation_dtype,
            )
        )

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
    summaries: SummaryMode = SummaryMode.LOCAL_AND_GLOBAL,
    check_budget: Callable[[], None] = lambda: None,
    on_history: HistoryCallback | None = None,
) -> tuple[LinearModel, list[JsonValue], float]:
    # Restored models predict from numeric weights using NumPy alone.
    from sklearn.exceptions import ConvergenceWarning  # type: ignore[import-untyped]
    from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
    from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]
    from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]

    require_prepared_inputs(training)
    require_prepared_inputs(validation)
    if targets.shape != (len(training.values),):
        raise ValueError("baseline training targets must align with every observation")
    eligible = targets >= 0
    y = targets[eligible]
    if not np.array_equal(np.unique(y), np.arange(len(DIRECTION_NAMES))):
        raise ValueError("baseline requires all three training directions")
    logger.info("Building baseline aggregates for %s eligible training rows", len(y))
    x = _training_aggregates(training, targets, context_sizes, summaries, check_budget)
    # All eligible rows contribute. Batching bounds scaling memory; only the
    # floating-point reduction order differs from a single full-matrix fit.
    scaler = StandardScaler(copy=False)
    for start in range(0, len(x), _BASELINE_BATCH_ROWS):
        check_budget()
        scaler.partial_fit(x[start : start + _BASELINE_BATCH_ROWS])
    for start in range(0, len(x), _BASELINE_BATCH_ROWS):
        check_budget()
        scaler.transform(x[start : start + _BASELINE_BATCH_ROWS])
    valid = aggregate(validation, context_sizes, summaries, check_budget=check_budget)
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
        with threadpool_limits(limits=threads):
            with warnings.catch_warnings():
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
                summaries,
            )
            probabilities = model.predict_aggregates(valid)
            for start in range(0, len(valid), _BASELINE_BATCH_ROWS):
                check_budget()
                stop = start + _BASELINE_BATCH_ROWS
                reference = scaler.transform(valid[start:stop].astype(np.float64), copy=False)
                expected = fitted.predict_proba(reference)
                if not np.allclose(expected, probabilities[start:stop], atol=1e-10, rtol=1e-10):
                    raise ValueError("restored baseline probabilities differ from scikit-learn")
            check_budget()
            score = scorer(probabilities)
        if not np.isfinite(score):
            raise ValueError("nonfinite baseline validation score")
        entry: dict[str, JsonValue] = {
            "regularization": regularization,
            "validation_segment_macro_f1": score,
        }
        history.append(entry)
        if on_history is not None:
            on_history(entry.copy())
        logger.info("baseline C=%s validation_segment_macro_f1=%.4f", regularization, score)
        if score > best_score:
            best, best_score, best_c = model, score, regularization
    assert best is not None
    return best, history, best_c
