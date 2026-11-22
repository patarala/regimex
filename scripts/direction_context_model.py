"""Learn direction from price movement across centered and offset windows."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from regimex.contracts import Observation
from regimex.directions import DIRECTION_NAMES, AnalysisConfig, observation_blocks
from regimex.trend_pauses import _sums

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
SCALES = (1, 2, 4, 8)
_SIGNED = ("movement", "within_level_movement", "within_level_correlation")
_MAGNITUDE = ("absolute_movement", "absolute_within_level_movement", "absolute_correlation")
_MEASURES = (*_SIGNED, *_MAGNITUDE, "residual", "coverage")
FEATURE_NAMES = tuple(
    f"{scale}_{position}_{name}"
    for scale in SCALES
    for position in ("center", "best_fit")
    for name in _MEASURES
)
_SIGN_COLUMNS = np.tile([name in _SIGNED for name in _MEASURES], 2 * len(SCALES))


def _line(xx: FloatArray, xy: FloatArray, yy: FloatArray) -> tuple[FloatArray, FloatArray]:
    slope = np.divide(xy, xx, out=np.zeros_like(xy), where=xx > 1e-10)
    return slope, np.maximum(0.0, yy - slope * xy)


def _window_features(
    prefix: FloatArray, prices: FloatArray, width: int, unit: float, minimum_coverage: float
) -> tuple[FloatArray, FloatArray]:
    """Fit every window at this width; allow one intercept change in its interior."""
    first = np.arange(len(prices) - width + 1)
    stop = first + width
    window = _sums(prefix, first, stop, first)
    xx, xy, yy = window.centered()
    slope, residual = _line(xx, xy, yy)
    within_slope, within_yy = slope.copy(), yy.copy()

    if width >= 4:
        changes = np.nan_to_num(np.abs(np.diff(prices)), nan=-1)
        largest = np.argmax(np.lib.stride_tricks.sliding_window_view(changes, width - 1), axis=1)
        for split in (
            first + largest + 1,
            first + width // 4,
            first + width // 2,
            first + 3 * width // 4,
        ):
            left = _sums(prefix, first, split, first)
            right = _sums(prefix, split, stop, split)
            lxx, lxy, lyy = left.centered()
            rxx, rxy, ryy = right.centered()
            split_yy = lyy + ryy
            adjusted, error = _line(lxx + rxx, lxy + rxy, split_yy)
            # A single point on one side cannot distinguish a level change from a trend.
            improves = (left.n >= 2) & (right.n >= 2) & (error < residual - 1e-12 * yy - 1e-17)
            within_slope = np.where(improves, adjusted, within_slope)
            within_yy = np.where(improves, split_yy, within_yy)
            residual = np.where(improves, error, residual)

    signal = np.maximum(0.0, within_yy - residual)
    r2 = np.divide(signal, within_yy, out=np.zeros_like(signal), where=within_yy > 1e-14)
    signed = np.column_stack(
        (
            np.arcsinh(slope * (width - 1) / unit),
            np.arcsinh(within_slope * (width - 1) / unit),
            np.sign(within_slope) * np.sqrt(np.minimum(1.0, r2)),
        )
    )
    rmse = np.sqrt(residual / np.maximum(window.n, 1))
    values = np.column_stack((signed, np.abs(signed), np.arcsinh(rmse / unit), window.n / width))
    return values, np.where(window.n >= max(2, minimum_coverage * width), rmse, np.inf)


def features(rows: tuple[Observation, ...], config: AnalysisConfig) -> FloatArray:
    """Compute local features independently in each available block of one instrument."""
    result = np.zeros((len(rows), len(FEATURE_NAMES)))
    unit = np.log1p(config.minimum_relative_change)
    for start, stop, status in observation_blocks(rows, config):
        if status != "observed":
            continue
        prices = np.array([r.last_price for r in rows[start:stop]], dtype=np.float64)
        known = np.isfinite(prices)
        prices = np.log(prices)
        prices -= prices[known][0]
        y = np.where(known, prices, 0.0)
        step = np.arange(len(prices), dtype=np.float64)
        moments = np.column_stack((known, step * known, y, step**2 * known, step * y, y**2))
        prefix = np.vstack((np.zeros(6), np.cumsum(moments, axis=0)))
        indices = np.arange(len(prices))
        for i, scale in enumerate(SCALES):
            width = min(scale * config.min_duration, len(prices))
            windows, error = _window_features(
                prefix, prices, width, unit, config.minimum_price_coverage
            )
            # Near a bend, a slightly offset window may better describe the current direction.
            # Prefer the centered window when fits tie. Every alternative contains this step.
            starts = np.clip(
                indices[:, None] - np.array([width // 2, width // 4, 3 * width // 4]),
                0,
                len(prices) - width,
            )
            errors = error[starts]
            tied = errors <= errors.min(axis=1, keepdims=True) + 1e-6 * unit
            best = starts[indices, np.argmax(tied, axis=1)]
            cols = len(_MEASURES)
            col = 2 * i * cols
            result[start:stop, col : col + cols] = windows[starts[:, 0]]
            result[start:stop, col + cols : col + 2 * cols] = windows[best]
    return result


@dataclass(frozen=True)
class DirectionContextModel:
    """Separate the evidence for movement from its sign under price reflection."""

    mean: FloatArray
    scale: FloatArray
    trend_weights: FloatArray
    sign_weights: FloatArray
    trend_intercept: float

    def predict(self, x: FloatArray) -> FloatArray:
        _validate_matrix(x)
        normalized = (x - self.mean) / self.scale
        trend_logit = normalized[:, ~_SIGN_COLUMNS] @ self.trend_weights + self.trend_intercept
        growth_logit = normalized[:, _SIGN_COLUMNS] @ self.sign_weights
        trend = np.exp(-np.logaddexp(0, -trend_logit))
        growth = np.exp(-np.logaddexp(0, -growth_logit))
        return np.column_stack((1 - trend, trend * growth, trend * (1 - growth)))

    def save(self, path: Path) -> None:
        with path.open("wb") as destination:
            np.savez(
                destination,
                features=FEATURE_NAMES,
                classes=DIRECTION_NAMES,
                mean=self.mean,
                scale=self.scale,
                trend_weights=self.trend_weights,
                sign_weights=self.sign_weights,
                trend_intercept=self.trend_intercept,
            )

    @classmethod
    def load(cls, path: Path) -> DirectionContextModel:
        with np.load(path, allow_pickle=False) as saved:
            if (
                tuple(saved["features"]) != FEATURE_NAMES
                or tuple(saved["classes"]) != DIRECTION_NAMES
            ):
                raise ValueError("Incompatible direction context features or classes")
            widths = (
                len(FEATURE_NAMES),
                len(FEATURE_NAMES),
                (~_SIGN_COLUMNS).sum(),
                _SIGN_COLUMNS.sum(),
            )
            arrays = [saved[key] for key in ("mean", "scale", "trend_weights", "sign_weights")]
            if any(
                a.dtype != np.float64 or a.shape != (n,) or not np.isfinite(a).all()
                for a, n in zip(arrays, widths, strict=True)
            ):
                raise ValueError("Invalid direction context weights")
            bias = saved["trend_intercept"]
            if (
                bias.shape != ()
                or bias.dtype != np.float64
                or not np.isfinite(bias)
                or np.any(arrays[1] <= 0)
                or np.any(arrays[0][_SIGN_COLUMNS] != 0)
            ):
                raise ValueError("Invalid direction context scaling or intercept")
            return cls(arrays[0], arrays[1], arrays[2], arrays[3], float(bias))


def _validate_matrix(x: FloatArray) -> None:
    if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES) or not np.isfinite(x).all():
        raise ValueError("Expected one finite feature vector per original time step")


def fit(x: FloatArray, y: IntArray) -> DirectionContextModel:
    """Fit both logistic regressions using only the supplied labeled training rows."""
    from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
    from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]

    _validate_matrix(x)
    if y.shape != (len(x),) or y.dtype.kind != "i" or not np.isin(y, [-1, 0, 1, 2]).all():
        raise ValueError("Expected one direction label or -1 for each training row")
    x, y = x[y >= 0], y[y >= 0]
    if len(np.unique(y)) != len(DIRECTION_NAMES):
        raise ValueError("Direction context fitting requires training examples of every class")
    mean = x.mean(axis=0)
    mean[_SIGN_COLUMNS] = 0
    scale = np.sqrt(np.mean((x - mean) ** 2, axis=0))
    scale[scale < 1e-8] = 1
    x = (x - mean) / scale
    trending = y != 0
    with threadpool_limits(limits=1):
        trend = LogisticRegression(C=0.1, max_iter=600).fit(x[:, ~_SIGN_COLUMNS], trending)
        sign = LogisticRegression(C=0.1, max_iter=600, fit_intercept=False).fit(
            x[trending][:, _SIGN_COLUMNS], y[trending] == 1
        )
    return DirectionContextModel(
        mean, scale, trend.coef_[0], sign.coef_[0], float(trend.intercept_[0])
    )
