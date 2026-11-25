"""Price segment features, accumulation expert and block gate for Segment MoE."""

from itertools import pairwise
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from regimex.backends.adaptive_forest import Floats, price_blocks
from regimex.backends.multiscale_segments import _cuts
from regimex.contracts import Observation
from regimex.directions import AnalysisConfig

_RADIUS = 4
_WINDOW = 2 * _RADIUS + 1
_LAGS = (1, 4, 16, 32)
_BLOCK_FEATURES = 7 * len(_LAGS)
type Float32Array = NDArray[np.float32]


def _piece_evidence(values: Floats) -> tuple[float, float, float, float]:
    """Fit a common slope on either side of a possible price-level change."""
    finite = np.isfinite(values)
    x = np.flatnonzero(finite).astype(float)
    y = values[finite]
    if len(y) < 2:
        return 0.0, 0.0, 0.0, 0.0
    x = (x - x.mean()) / (len(values) - 1)
    y = y - y.mean()
    xx, xy, yy = x @ x, x @ y, y @ y

    # Compare the unsplit fit with every cut having two observations on each side.
    cuts = np.arange(2, len(y) - 1)
    sum_x, sum_y = x.cumsum()[cuts - 1], y.cumsum()[cuts - 1]
    sizes = 1 / cuts + 1 / (len(y) - cuts)
    within_xx = np.maximum(np.r_[xx, xx - sum_x * sum_x * sizes], 1e-24)
    within_xy = np.r_[xy, xy - sum_x * sum_y * sizes]
    within_yy = np.maximum(np.r_[yy, yy - sum_y * sum_y * sizes], 0.0)
    error = np.maximum(within_yy - within_xy * within_xy / within_xx, 0.0)
    tolerance = 32 * np.finfo(float).eps * max(float(yy), 1e-18)
    best = np.flatnonzero(error <= error.min() + tolerance)[0]
    movement = np.clip(100 * within_xy / within_xx, -300, 300)
    fit = np.clip(within_xy * within_xy / within_xx / np.maximum(within_yy, 1e-20), 0, 1)
    return float(movement[best]), float(fit[best]), float(movement[0]), float(fit[0])


def _block_evidence(values: Floats) -> Float32Array:
    """Describe the size and consistency of changes, independent of their direction."""
    prices = values[np.isfinite(values)]
    features: list[float] = []
    for lag in _LAGS:
        changes = prices[lag:] - prices[:-lag]
        if len(changes) < 2:
            features.extend([0.0] * 7)
            continue
        magnitudes = np.abs(changes)
        nonzero = changes[magnitudes > 1e-12]
        rms = np.sqrt(np.mean(changes * changes))
        median, upper = np.log1p(1000 * np.quantile(magnitudes, [0.5, 0.9]))
        signs = np.sign(nonzero)
        features.extend(
            [
                np.log1p(1000 * rms),
                median,
                upper,
                np.mean(magnitudes) / (rms + 1e-12),
                np.abs(np.mean(changes)) / (rms + 1e-12),
                np.mean(signs[1:] * signs[:-1]) if len(signs) > 1 else 0.0,
                np.max(magnitudes) / (np.sqrt(sum(changes * changes)) + 1e-12),
            ]
        )
    return np.asarray(features, dtype=np.float32)


def geometry(
    rows: tuple[Observation, ...], config: AnalysisConfig
) -> tuple[NDArray[np.int64], Float32Array, Float32Array]:
    """Return original-grid spans, adjacent-piece evidence and block evidence per span."""
    spans = []
    contexts = []
    blocks = []
    for start, prices, prefix, noise in price_blocks(rows, config):
        cuts = _cuts(prefix, start, config, noise)[1]
        pieces = np.array([_piece_evidence(prices[a:b]) for a, b in pairwise(cuts)], np.float32)
        # Reset the window at each availability boundary.
        neighbors = np.pad(pieces, ((_RADIUS, _RADIUS), (0, 0)))
        block = _block_evidence(prices)
        for i, (a, b) in enumerate(pairwise(cuts)):
            spans.append((start + a, start + b))
            contexts.append(neighbors[i : i + _WINDOW])
            blocks.append(block)
    return (
        np.asarray(spans, dtype=np.int64).reshape(-1, 2),
        np.asarray(contexts, dtype=np.float32).reshape(-1, _WINDOW, 4),
        np.asarray(blocks, dtype=np.float32).reshape(-1, _BLOCK_FEATURES),
    )


class Accumulation(nn.Module):
    """Estimate direction using consecutive pieces with the same sign of slope."""

    def __init__(self) -> None:
        super().__init__()
        self.fit_threshold = nn.Parameter(torch.tensor(0.0))
        self.move_threshold = nn.Parameter(torch.tensor(0.0))
        self.log_gain = nn.Parameter(torch.tensor([2.0, 2.0]))
        hidden = nn.Linear(6, 16)
        correction = nn.Linear(16, 1)
        nn.init.zeros_(correction.weight)
        nn.init.zeros_(correction.bias)
        self.adjustment = nn.Sequential(hidden, nn.Tanh(), correction)

    def forward(self, context: Tensor) -> Tensor:
        slope, fit, simple_slope, simple_fit = context.unbind(-1)
        sign = slope[:, _RADIUS : _RADIUS + 1].sign()
        gain = 4 + self.log_gain.clamp(-1, 5).exp()
        fit_threshold = self.fit_threshold.sigmoid()
        reliable = torch.sigmoid(gain[0] * (fit - fit_threshold)) * (sign * slope > 0).float()
        # A neighbor contributes only through the intervening pieces.
        left = reliable[:, :_RADIUS].flip((1,)).cumprod(1).flip((1,))
        right = reliable[:, _RADIUS + 1 :].cumprod(1)
        current = slope[:, _RADIUS]
        supported = (
            (slope[:, :_RADIUS].abs() * left).sum(1)
            + (slope[:, _RADIUS + 1 :].abs() * right).sum(1)
            + current.abs()
        ).clamp(max=100)
        sign = sign[:, 0]
        summary = torch.stack(
            (
                supported.asinh(),
                (current * sign).asinh(),
                fit[:, _RADIUS],
                (simple_slope[:, _RADIUS] * sign).asinh(),
                simple_fit[:, _RADIUS],
                (left.sum(1) + right.sum(1)) / (2 * _RADIUS),
            ),
            dim=1,
        )
        residual = self.adjustment(summary).squeeze(1)
        move_threshold = 2 * self.move_threshold.sigmoid()
        direction = torch.sigmoid(gain[0] * (fit[:, _RADIUS] - fit_threshold)) * torch.sigmoid(
            gain[1] * (supported - move_threshold) + residual
        )
        direction = direction * (sign != 0)
        return torch.stack(
            (
                1 - direction,
                torch.where(sign > 0, direction, 0.0),
                torch.where(sign < 0, direction, 0.0),
            ),
            dim=1,
        )


class Gate(nn.Module):
    """Return mixture log odds from the observed price block."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(_BLOCK_FEATURES, 24), nn.Tanh(), nn.Linear(24, 1))

    def forward(self, features: Tensor) -> Tensor:
        return cast(Tensor, self.layers(features))
