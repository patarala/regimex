"""Price-shape representation and a retrospective temporal convolutional classifier."""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise
from typing import Literal, cast

import numpy as np
import torch
from pydantic import JsonValue
from torch import Tensor, nn
from torch.nn import functional as functional

from regimex.directions import DIRECTION_NAMES
from regimex.features import FEATURE_NAMES, DoubleArray, FloatArray, IntArray
from regimex.model_registry import TCNParameters, TrainingPolicy, TrainingRuntime
from regimex.prediction_inputs import (
    PredictionInputs,
    PreparedPredictionInputs,
    require_prepared_inputs,
    require_raw_inputs,
)

Scorer = Callable[[DoubleArray], float]

logger = logging.getLogger(__name__)
TEMPORAL_VERSION = "1.0"
CONTEXT_SIZES = (32, 64, 128, 256, 512)
DEFAULT_DILATIONS = (1, 2, 4, 8, 16, 32, 64, 128)
SELECTION_BATCH_SIZE: Literal[16] = 16
TEMPORAL_NUMERIC_NAMES = (
    "relative_log_price_asinh",
    "log_return_asinh",
    "absolute_log_return_asinh",
    "sequence_standardized_log_price",
    "sequence_log_price_scale",
    *(
        f"context_{size}_{statistic}"
        for size in CONTEXT_SIZES
        for statistic in ("slope_change_asinh", "linear_fit", "endpoint_change_asinh", "coverage")
    ),
)
TEMPORAL_NAMES = (
    *TEMPORAL_NUMERIC_NAMES,
    *(f"{name}_missing" for name in TEMPORAL_NUMERIC_NAMES),
)


def _price_context(price: DoubleArray, size: int) -> DoubleArray:
    """Fit observed prices against their actual timestamps, without filling gaps."""
    length = len(price)
    time = np.arange(length, dtype=np.float64)
    observed = np.isfinite(price)
    value = np.where(observed, price, 0.0)
    statistics = np.column_stack(
        (observed, time * observed, time * time * observed, value, value * value, time * value)
    )
    prefix = np.vstack((np.zeros((1, 6)), statistics.cumsum(axis=0)))
    positions = np.arange(length)
    left = np.maximum(0, positions - size // 2)
    right = np.minimum(length, positions + (size + 1) // 2)
    count, sx, sxx, sy, syy, sxy = (prefix[right] - prefix[left]).T
    denominator = np.maximum(count, 1)
    xx = np.maximum(0, sxx - sx * sx / denominator)
    yy = np.maximum(0, syy - sy * sy / denominator)
    xy = sxy - sx * sy / denominator
    valid = (count >= 2) & (xx > 0)
    slope = np.divide(xy, xx, out=np.zeros(length), where=valid)
    fit = np.clip(
        np.divide(xy * xy, xx * yy, out=np.zeros(length), where=valid & (yy > 1e-16)),
        0,
        1,
    )
    known = np.flatnonzero(observed)
    change = np.full(length, np.nan)
    if len(known):
        first = np.searchsorted(known, left)
        last = np.searchsorted(known, right) - 1
        good = (first < len(known)) & (last >= first)
        change[good] = price[known[last[good]]] - price[known[first[good]]]
    result = np.column_stack(
        (
            np.arcsinh(slope * (right - left - 1) / 0.01),
            fit,
            np.arcsinh(change / 0.01),
            count / (right - left),
        )
    )
    result[~valid, :3] = np.nan
    return result


def temporal_features(sequences: PredictionInputs) -> FloatArray:
    """Derive price-only inputs independently within each completed sequence.

    Centered contexts and complete-sequence normalization are retrospective.
    Activity, labels, reference boundaries and simulator settings are unused.
    """
    require_raw_inputs(sequences)
    if sequences.values.ndim != 2 or sequences.values.shape[1] != len(FEATURE_NAMES):
        raise ValueError("temporal representation requires raw observation features")
    result = np.full((len(sequences.values), len(TEMPORAL_NUMERIC_NAMES)), np.nan)
    for start, stop in zip(sequences.offsets[:-1], sequences.offsets[1:], strict=True):
        price = sequences.values[start:stop, 0].astype(np.float64)
        returns = sequences.values[start:stop, 1].astype(np.float64)
        observed = np.isfinite(price)
        standardized = np.full(len(price), np.nan)
        scale = np.full(len(price), np.nan)
        if observed.any():
            deviation = float(price[observed].std())
            standardized[observed] = (price[observed] - price[observed].mean()) / max(
                deviation, 0.001
            )
            scale[:] = np.log1p(deviation / 0.001)
        result[start:stop] = np.column_stack(
            (
                np.arcsinh(price / 0.01),
                np.arcsinh(returns / 0.001),
                np.arcsinh(np.abs(returns) / 0.001),
                standardized,
                scale,
                *(_price_context(price, size) for size in CONTEXT_SIZES),
            )
        )
    if np.isinf(result).any():
        raise ValueError("temporal representation produced infinite values")
    return np.concatenate((result, np.isnan(result)), axis=1).astype(np.float32)


@dataclass(frozen=True)
class TemporalPreprocessor:
    fill: DoubleArray
    mean: DoubleArray
    scale: DoubleArray

    @classmethod
    def fit(cls, training: FloatArray) -> TemporalPreprocessor:
        if (
            training.ndim != 2
            or training.shape[1] != len(TEMPORAL_NAMES)
            or not len(training)
            or np.isinf(training).any()
        ):
            raise ValueError("invalid temporal training features")
        fill = np.zeros(len(TEMPORAL_NAMES), dtype=np.float64)
        for column in range(len(TEMPORAL_NUMERIC_NAMES)):
            known = training[np.isfinite(training[:, column]), column]
            if len(known):
                fill[column] = np.median(known)
        complete = np.where(np.isnan(training), fill, training)
        mean = complete.mean(axis=0)
        scale = complete.std(axis=0)
        scale[scale < 1e-12] = 1.0
        mean[len(TEMPORAL_NUMERIC_NAMES) :] = 0.0
        scale[len(TEMPORAL_NUMERIC_NAMES) :] = 1.0
        return cls(fill, mean, scale)

    def transform(self, values: FloatArray) -> FloatArray:
        if values.ndim != 2 or values.shape[1] != len(self.fill) or np.isinf(values).any():
            raise ValueError("incompatible temporal features")
        result = (np.where(np.isnan(values), self.fill, values) - self.mean) / self.scale
        if not np.isfinite(result).all():
            raise ValueError("temporal preprocessing produced nonfinite values")
        return result.astype(np.float32)


class _ResidualBlock(nn.Module):
    def __init__(self, hidden: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.conv = nn.Conv1d(hidden, hidden, 3, padding=dilation, dilation=dilation)
        self.mix = nn.Conv1d(hidden, hidden, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: Tensor, mask: Tensor) -> Tensor:
        return cast(Tensor, (values + self.dropout(self.mix(self.conv(values).relu()))) * mask)


class TCNClassifier(nn.Module):
    def __init__(
        self,
        features: int,
        hidden: int,
        dropout: float,
        dilations: tuple[int, ...] = DEFAULT_DILATIONS,
    ) -> None:
        super().__init__()
        self.input = nn.Conv1d(features, hidden, 1)
        self.blocks = nn.ModuleList(
            _ResidualBlock(hidden, dilation, dropout) for dilation in dilations
        )
        self.head = nn.Conv1d(hidden, len(DIRECTION_NAMES), 1)

    def forward(self, batch: Tensor, lengths: Tensor) -> Tensor:
        mask = (
            torch.arange(batch.shape[1], device=batch.device)[None, :] < lengths[:, None]
        ).unsqueeze(1)
        # Re-mask every residual block so padded positions cannot feed back into valid ones.
        values = self.input(batch.transpose(1, 2) * mask) * mask
        for block in self.blocks:
            values = block(values, mask)
        return cast(Tensor, (self.head(values) * mask).transpose(1, 2))


def _prediction_batch(
    sequences: PreparedPredictionInputs, selected: IntArray
) -> tuple[Tensor, Tensor]:
    require_prepared_inputs(sequences)
    lengths = np.diff(sequences.offsets)[selected]
    values = np.zeros(
        (len(selected), int(lengths.max()), sequences.values.shape[1]), dtype=np.float32
    )
    for row, index in enumerate(selected):
        start, stop = sequences.offsets[index : index + 2]
        values[row, : stop - start] = sequences.values[start:stop]
    return torch.from_numpy(values), torch.from_numpy(lengths)


def predict_tcn(
    model: TCNClassifier,
    sequences: PreparedPredictionInputs,
    batch_size: int = SELECTION_BATCH_SIZE,
) -> DoubleArray:
    require_prepared_inputs(sequences)
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("prediction batch size must be a positive integer")
    model.eval()
    result = np.empty((len(sequences.values), len(DIRECTION_NAMES)), dtype=np.float64)
    with torch.inference_mode():
        for first in range(0, len(sequences.info), batch_size):
            selected = np.arange(
                first, min(first + batch_size, len(sequences.info)), dtype=np.int64
            )
            batch, lengths = _prediction_batch(sequences, selected)
            probabilities = model(batch, lengths).softmax(dim=2).numpy()
            for row, index in enumerate(selected):
                start, stop = sequences.offsets[index : index + 2]
                result[start:stop] = probabilities[row, : stop - start]
    result /= result.sum(axis=1, keepdims=True)
    return result


def _interval_weights(targets: IntArray, offsets: IntArray, power: float) -> FloatArray:
    weights = np.ones(len(targets), dtype=np.float32)
    if power:
        for start, stop in pairwise(offsets):
            labels = targets[start:stop]
            boundaries = np.concatenate(([0], np.flatnonzero(np.diff(labels)) + 1, [len(labels)]))
            for left, right in pairwise(boundaries):
                weights[start + left : start + right] = (len(labels) / (right - left)) ** power
        weights /= weights[targets >= 0].mean()
    return weights


def _supervised_batch(
    training: PreparedPredictionInputs, targets: IntArray, selected: IntArray
) -> tuple[Tensor, Tensor, Tensor]:
    inputs, lengths = _prediction_batch(training, selected)
    labels = np.full(inputs.shape[:2], -1, dtype=np.int64)
    for row, index in enumerate(selected):
        start, stop = training.offsets[index : index + 2]
        labels[row, : stop - start] = targets[start:stop]
    return inputs, torch.from_numpy(labels), lengths


def _boundary_weights(targets: IntArray, offsets: IntArray) -> FloatArray:
    """Emphasize four training steps on each side of an internal direction change.

    Availability edges are not direction changes. Neighborhoods stay inside their
    own sequence and contiguous eligible block; overlapping neighborhoods use the
    same multiplier rather than compounding it.
    """
    weights = np.ones(len(targets), dtype=np.float32)
    for start, stop in pairwise(offsets):
        labels = targets[start:stop]
        eligible = labels >= 0
        edges = np.diff(np.r_[False, eligible, False].astype(np.int8))
        for left, right in zip(
            np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True
        ):
            changes = np.flatnonzero(np.diff(labels[left:right])) + left + 1
            for boundary in changes:
                weights[start + max(left, boundary - 4) : start + min(right, boundary + 4)] = 2.0
    return weights


def _supervised_loss(
    logits: Tensor,
    targets: Tensor,
    class_weights: Tensor,
    sample_weights: Tensor,
    smoothing: float,
) -> Tensor:
    """Fit directions while smoothing only within a known training direction."""
    eligible = targets >= 0
    effective = class_weights[targets.clamp_min(0)] * eligible * sample_weights
    if not bool(effective.sum() > 0):
        raise ValueError("supervised batch requires positive eligible target weight")
    losses = functional.cross_entropy(
        logits.transpose(1, 2), targets, weight=class_weights, ignore_index=-1, reduction="none"
    )
    loss = (losses * sample_weights).sum() / effective.sum()
    within_direction = eligible[:, 1:] & eligible[:, :-1] & (targets[:, 1:] == targets[:, :-1])
    if smoothing and bool(within_direction.any()):
        log_probabilities = logits.log_softmax(dim=2)
        changes = (
            (log_probabilities[:, 1:] - log_probabilities[:, :-1].detach())
            .square()
            .clamp_max(16)
            .mean(dim=2)
        )
        loss = loss + smoothing * changes[within_direction].mean()
    return loss


def fit_tcn(
    training: PreparedPredictionInputs,
    targets: IntArray,
    validation: PreparedPredictionInputs,
    *,
    parameters: TCNParameters,
    policy: TrainingPolicy,
    runtime: TrainingRuntime,
    scorer: Scorer,
    check_budget: Callable[[], None] | None = None,
) -> tuple[TCNClassifier, list[dict[str, JsonValue]], int]:
    """Fit saved numeric inputs with supervision kept outside prediction data."""
    require_prepared_inputs(training)
    require_prepared_inputs(validation)
    if (
        targets.dtype != np.int64
        or targets.shape != (len(training.values),)
        or not np.isin(targets, [-1, 0, 1, 2]).all()
        or training.values.shape[1] != validation.values.shape[1]
        or policy.optimizer != "adamw"
    ):
        raise ValueError("TCN requires aligned direction targets, features and AdamW policy")
    torch.set_num_threads(runtime.threads)
    torch.use_deterministic_algorithms(True)
    labels = targets[targets >= 0]
    counts = np.bincount(labels, minlength=len(DIRECTION_NAMES))
    if (counts == 0).any():
        raise ValueError("TCN requires all three training directions")
    class_weights = torch.tensor(len(labels) / (len(DIRECTION_NAMES) * counts), dtype=torch.float32)
    interval_weights = _interval_weights(
        targets, training.offsets, parameters.interval_weight_power
    )
    interval_weights *= _boundary_weights(targets, training.offsets)
    best_state: dict[str, Tensor] | None = None
    best_score = -1.0
    best_seed = policy.seeds[0]
    history: list[dict[str, JsonValue]] = []
    for seed in policy.seeds:
        if check_budget is not None:
            check_budget()
        torch.manual_seed(seed)
        rng = np.random.Generator(np.random.PCG64(seed))
        model = TCNClassifier(
            training.values.shape[1], parameters.channels, parameters.dropout, parameters.dilations
        )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=policy.learning_rate, weight_decay=policy.weight_decay
        )
        seed_best = -1.0
        stale = 0
        for epoch in range(1, policy.epochs + 1):
            model.train()
            ordering = rng.permutation(len(training.info)).astype(np.int64)
            total_loss = 0.0
            total_labeled = 0
            for first in range(0, len(ordering), policy.batch_size):
                if check_budget is not None:
                    check_budget()
                selected: IntArray = ordering[first : first + policy.batch_size]
                inputs, batch_targets, lengths = _supervised_batch(training, targets, selected)
                count = int((batch_targets >= 0).sum())
                if not count:
                    continue
                sample_weights = torch.zeros_like(batch_targets, dtype=torch.float32)
                for row, index in enumerate(selected):
                    start, stop = training.offsets[index : index + 2]
                    sample_weights[row, : stop - start] = torch.from_numpy(
                        interval_weights[start:stop]
                    )
                optimizer.zero_grad(set_to_none=True)
                logits = model(inputs, lengths)
                loss = _supervised_loss(
                    logits, batch_targets, class_weights, sample_weights, parameters.smoothing
                )
                if not bool(torch.isfinite(loss)):
                    raise ValueError("nonfinite TCN training loss")
                torch.autograd.backward(loss)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                total_loss += float(loss.detach()) * count
                total_labeled += count
            entry: dict[str, JsonValue] = {
                "seed": seed,
                "epoch": epoch,
                "training_loss": total_loss / total_labeled,
            }
            validate = (
                epoch == 1 or epoch % policy.validation_interval == 0 or epoch == policy.epochs
            )
            if validate:
                if check_budget is not None:
                    check_budget()
                score = scorer(predict_tcn(model, validation, batch_size=SELECTION_BATCH_SIZE))
                if not np.isfinite(score):
                    raise ValueError("nonfinite TCN validation score")
                entry["validation_segment_macro_f1"] = score
                logger.info(
                    "TCN seed=%s epoch=%s loss=%.4f validation_segment_macro_f1=%.4f",
                    seed,
                    epoch,
                    total_loss / total_labeled,
                    score,
                )
                if score > best_score:
                    best_score, best_seed = score, seed
                    best_state = {
                        key: tensor.detach().clone() for key, tensor in model.state_dict().items()
                    }
                if score > seed_best:
                    seed_best, stale = score, 0
                else:
                    stale += 1
            history.append(entry)
            if stale >= policy.patience:
                break
    assert best_state is not None
    best = TCNClassifier(
        training.values.shape[1], parameters.channels, parameters.dropout, parameters.dilations
    )
    best.load_state_dict(best_state)
    return best, history, best_seed
