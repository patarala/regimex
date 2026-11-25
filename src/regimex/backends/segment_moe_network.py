"""Tensor-only modules for mixing two expert predictions on a price segment."""

from typing import cast

import torch
from torch import Tensor, nn

from regimex.backends.segment_moe_layout import (
    ACCUMULATION_INPUTS,
    ACCUMULATION_WIDTH,
    BLOCK_FEATURE_COUNT,
    CLASS_COUNT,
    CONTEXT_LENGTH,
    GATE_WIDTH,
    NEIGHBOR_RADIUS,
    SEGMENT_FEATURE_COUNT,
)


class AccumulationExpert(nn.Module):
    """Estimate direction using consecutive pieces with the same sign of slope."""

    def __init__(self) -> None:
        super().__init__()
        self.fit_threshold = nn.Parameter(torch.tensor(0.0))
        self.move_threshold = nn.Parameter(torch.tensor(0.0))
        self.log_gain = nn.Parameter(torch.tensor([2.0, 2.0]))
        hidden = nn.Linear(ACCUMULATION_INPUTS, ACCUMULATION_WIDTH)
        correction = nn.Linear(ACCUMULATION_WIDTH, 1)
        nn.init.zeros_(correction.weight)
        nn.init.zeros_(correction.bias)
        self.adjustment = nn.Sequential(hidden, nn.Tanh(), correction)

    def forward(self, context: Tensor) -> Tensor:
        slope, fit, simple_slope, simple_fit = context.unbind(-1)
        current = slope[:, NEIGHBOR_RADIUS]
        sign = current.sign()
        gain = 4 + self.log_gain.clamp(-1, 5).exp()
        fit_threshold = self.fit_threshold.sigmoid()
        reliable = torch.sigmoid(gain[0] * (fit - fit_threshold)) * (sign[:, None] * slope > 0)
        # A neighbor contributes only through the intervening pieces.
        left = reliable[:, :NEIGHBOR_RADIUS].flip((1,)).cumprod(1).flip((1,))
        right = reliable[:, NEIGHBOR_RADIUS + 1 :].cumprod(1)
        supported = (
            (slope[:, :NEIGHBOR_RADIUS].abs() * left).sum(1)
            + (slope[:, NEIGHBOR_RADIUS + 1 :].abs() * right).sum(1)
            + current.abs()
        ).clamp(max=100)
        summary = torch.stack(
            (
                supported.asinh(),
                current.abs().asinh(),
                fit[:, NEIGHBOR_RADIUS],
                (simple_slope[:, NEIGHBOR_RADIUS] * sign).asinh(),
                simple_fit[:, NEIGHBOR_RADIUS],
                (left.sum(1) + right.sum(1)) / (2 * NEIGHBOR_RADIUS),
            ),
            dim=1,
        )
        residual = self.adjustment(summary).squeeze(1)
        move_threshold = 2 * self.move_threshold.sigmoid()
        direction = torch.sigmoid(
            gain[0] * (fit[:, NEIGHBOR_RADIUS] - fit_threshold)
        ) * torch.sigmoid(gain[1] * (supported - move_threshold) + residual)
        direction = direction * (sign != 0)
        return torch.stack(
            (
                1 - direction,
                torch.where(sign > 0, direction, 0.0),
                torch.where(sign < 0, direction, 0.0),
            ),
            dim=1,
        )


class BlockGate(nn.Module):
    """Return mixture log odds from the observed price block."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(BLOCK_FEATURE_COUNT, GATE_WIDTH), nn.Tanh(), nn.Linear(GATE_WIDTH, 1)
        )

    def forward(self, features: Tensor) -> Tensor:
        return cast(Tensor, self.layers(features))


class ContextExpert(nn.Module):
    """Average the logits of a window and its direction-reflected counterpart."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(CONTEXT_LENGTH * SEGMENT_FEATURE_COUNT, width),
            nn.SiLU(),
            nn.Linear(width, CLASS_COUNT),
        )

    def forward(self, segments: Tensor) -> Tensor:
        adjusted, adjusted_fit, simple, simple_fit = segments.unbind(-1)
        adjusted = adjusted.asinh()
        simple = simple.asinh()
        direct = torch.stack((adjusted, adjusted_fit, simple, simple_fit), dim=-1)
        reflected = torch.stack((-adjusted, adjusted_fit, -simple, simple_fit), dim=-1)
        logits: Tensor = self.layers(direct.flatten(1))
        # Restore the original growth/decline order before averaging the two views.
        reflected_logits: Tensor = self.layers(reflected.flatten(1))[:, (0, 2, 1)]
        return ((logits + reflected_logits) * 0.5).softmax(-1)


class SegmentMoE(nn.Module):
    """Mix two segment-level direction distributions with a block-level gate."""

    def __init__(self, width: int) -> None:
        super().__init__()
        # These names determine the tensor order in version 1 of the weight codec.
        self.accumulation = AccumulationExpert()
        self.context = ContextExpert(width)
        self.gate = BlockGate()

    def forward(self, segments: Tensor, block: Tensor) -> Tensor:
        accumulation_weight = self.gate(block).sigmoid()
        return cast(
            Tensor,
            accumulation_weight * self.accumulation(segments)
            + (1 - accumulation_weight) * self.context(segments),
        )
