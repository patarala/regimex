"""Observation features for retrospective analysis and training-only numeric preprocessing."""

from dataclasses import dataclass
from math import isfinite
from sys import float_info

import numpy as np
from numpy.typing import NDArray

from regimex.backends.baseline_layout import FEATURE_NAMES as FEATURE_NAMES
from regimex.backends.baseline_layout import FEATURE_VERSION as FEATURE_VERSION
from regimex.backends.baseline_layout import NUMERIC_NAMES as NUMERIC_NAMES
from regimex.contracts import Observation

FloatArray = NDArray[np.float32]
DoubleArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _imbalance(buy: float | None, sell: float | None) -> float:
    if buy is None or sell is None:
        return float("nan")
    total = buy + sell
    if total == 0:
        return 0.0
    if isfinite(total):
        return (buy - sell) / total
    maximum = max(buy, sell)
    buy_scaled, sell_scaled = buy / maximum, sell / maximum
    return (buy_scaled - sell_scaled) / (buy_scaled + sell_scaled)


def log_price_channels(rows: tuple[Observation, ...]) -> FloatArray:
    """Relative log price and adjacent log return on one instrument's original grid."""
    if not rows or any(
        row.instrument_id != rows[0].instrument_id or row.step != i for i, row in enumerate(rows)
    ):
        raise ValueError("features require one ordered contiguous instrument sequence")
    prices = np.fromiter(
        (np.log(row.last_price) if row.last_price is not None else np.nan for row in rows),
        dtype=np.float64,
        count=len(rows),
    )
    known = np.flatnonzero(np.isfinite(prices))
    reference = prices[known[0]] if len(known) else np.nan
    return np.column_stack((prices - reference, np.diff(prices, prepend=np.nan))).astype(np.float32)


def compute_features(rows: tuple[Observation, ...]) -> FloatArray:
    prices = log_price_channels(rows)
    values = np.full((len(rows), len(NUMERIC_NAMES)), np.nan, dtype=np.float64)
    values[:, :2] = prices
    values[:, 11] = np.abs(prices[:, 1])
    for step, row in enumerate(rows):
        activity = (
            row.trade_count,
            row.trade_volume,
            row.buy_order_count,
            row.buy_order_volume,
            row.sell_order_count,
            row.sell_order_volume,
        )
        for column, value in enumerate(activity, start=2):
            if value is not None:
                values[step, column] = np.log1p(value)
        values[step, 8] = _imbalance(row.buy_order_count, row.sell_order_count)
        values[step, 9] = _imbalance(row.buy_order_volume, row.sell_order_volume)
        if row.best_bid is not None and row.best_ask is not None:
            total = row.best_ask + row.best_bid
            if isfinite(total) and total >= 2 * float_info.min:
                values[step, 10] = (row.best_ask - row.best_bid) / (total / 2)
            else:
                maximum = max(row.best_ask, row.best_bid)
                ask, bid = row.best_ask / maximum, row.best_bid / maximum
                values[step, 10] = (ask - bid) / ((ask + bid) / 2)
    masks = np.isnan(values).astype(np.float64)
    return np.concatenate((values, masks), axis=1).astype(np.float32)


@dataclass(frozen=True)
class Preprocessor:
    fill: DoubleArray
    mean: DoubleArray
    scale: DoubleArray

    @classmethod
    def fit(cls, training: FloatArray) -> Preprocessor:
        if training.ndim != 2 or training.shape[1] != len(FEATURE_NAMES) or not len(training):
            raise ValueError("invalid training feature matrix")
        if np.isinf(training).any():
            raise ValueError("features contain infinite values")
        fill = np.zeros(training.shape[1], dtype=np.float64)
        for column in range(len(NUMERIC_NAMES)):
            known = training[np.isfinite(training[:, column]), column]
            if len(known):
                fill[column] = np.median(known)
        complete = np.where(np.isnan(training), fill, training)
        mean = complete.mean(axis=0)
        scale = complete.std(axis=0)
        scale[scale < 1e-12] = 1.0
        mean[len(NUMERIC_NAMES) :] = 0.0
        scale[len(NUMERIC_NAMES) :] = 1.0
        return cls(fill, mean, scale)

    def transform(self, values: FloatArray) -> FloatArray:
        if values.ndim != 2 or values.shape[1] != len(self.fill) or np.isinf(values).any():
            raise ValueError("incompatible features")
        result = (np.where(np.isnan(values), self.fill, values) - self.mean) / self.scale
        if not np.isfinite(result).all():
            raise ValueError("preprocessing produced nonfinite values")
        return result.astype(np.float32)
