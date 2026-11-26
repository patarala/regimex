"""Observation features for retrospective analysis and training-only numeric preprocessing."""

from dataclasses import dataclass
from math import isfinite
from sys import float_info
from typing import Self

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


def log_price_changes(prices: DoubleArray, reference: float | DoubleArray) -> DoubleArray:
    """Compute log price ratios, retaining small changes and extreme finite endpoints."""
    # Separate logs cover ratios outside the normal float range. For other prices
    # take the log after division, preserving relative precision across price levels.
    result = np.log(prices) - np.log(reference)
    with np.errstate(over="ignore", under="ignore"):
        ratio = prices / reference
        change = (prices - reference) / reference
    np.log(ratio, out=result, where=np.isfinite(ratio) & (ratio >= np.finfo(np.float64).tiny))
    np.log1p(change, out=result, where=np.abs(change) <= 0.5)
    return result


def log_price_channels(rows: tuple[Observation, ...], *, legacy_prices: bool = False) -> FloatArray:
    """Relative log price and adjacent log return on one instrument's original grid."""
    if not rows or any(
        row.instrument_id != rows[0].instrument_id or row.step != i for i, row in enumerate(rows)
    ):
        raise ValueError("features require one ordered contiguous instrument sequence")
    prices = np.fromiter((row.last_price for row in rows), dtype=np.float64, count=len(rows))
    if legacy_prices:
        prices = np.fromiter((np.log(price) for price in prices), dtype=np.float64, count=len(rows))
    known = np.flatnonzero(np.isfinite(prices))
    reference = float(prices[known[0]]) if len(known) else np.nan
    if legacy_prices:
        relative, returns = prices - reference, np.diff(prices, prepend=np.nan)
    else:
        relative = log_price_changes(prices, reference)
        returns = np.r_[np.nan, log_price_changes(prices[1:], prices[:-1])]
    return np.column_stack((relative, returns)).astype(np.float32)


def compute_features(rows: tuple[Observation, ...], *, legacy_prices: bool = False) -> FloatArray:
    prices = log_price_channels(rows, legacy_prices=legacy_prices)
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
    def fit(cls, training: FloatArray) -> Self:
        return cls._fit_channels(training, len(NUMERIC_NAMES))

    @classmethod
    def _fit_channels(cls, training: FloatArray, numeric_width: int) -> Self:
        if training.ndim != 2 or training.shape[1] != 2 * numeric_width or not len(training):
            raise ValueError("invalid training feature matrix")
        if np.isinf(training).any():
            raise ValueError("features contain infinite values")
        fill = np.zeros(training.shape[1], dtype=np.float64)
        for column in range(numeric_width):
            known = training[np.isfinite(training[:, column]), column]
            if len(known):
                # Keep the input precision without overflowing the sum of the middle pair.
                median = np.median(known.astype(np.float64), overwrite_input=True)
                fill[column] = median.astype(known.dtype)
        complete = np.where(np.isnan(training), fill, training)
        mean = complete.mean(axis=0)
        scale = complete.std(axis=0)
        scale[scale < 1e-12] = 1.0
        mean[numeric_width:] = 0.0
        scale[numeric_width:] = 1.0
        return cls(fill, mean, scale)

    def transform(self, values: FloatArray) -> FloatArray:
        if values.ndim != 2 or values.shape[1] != len(self.fill) or np.isinf(values).any():
            raise ValueError("incompatible features")
        with np.errstate(over="ignore"):
            result = (
                (np.where(np.isnan(values), self.fill, values) - self.mean) / self.scale
            ).astype(np.float32)
        if not np.isfinite(result).all():
            raise ValueError("preprocessing produced nonfinite values")
        return result
