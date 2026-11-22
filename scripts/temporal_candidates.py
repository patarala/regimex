"""Networks compared on the original instrument grid by compare_temporal_architectures."""

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from regimex.backends.tcn_layout import DEFAULT_DILATIONS
from regimex.directions import DIRECTION_NAMES
from regimex.temporal import TCNClassifier


class PatchEncoder(nn.Module):
    """Attend across eight-step patches, keeping a local path to each step's logits."""

    patch_size = 8

    def __init__(self, features: int, width: int, dropout: float) -> None:
        super().__init__()
        self.local = nn.Linear(features, width)
        self.patch = nn.Linear(features * self.patch_size, width)
        self.layers = nn.ModuleList(
            nn.TransformerEncoderLayer(
                width,
                nhead=4,
                dim_feedforward=width * 3,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(2)
        )
        frequencies = torch.exp(torch.arange(0, width, 2) * (-math.log(10000) / width))
        self.register_buffer("frequencies", frequencies)
        self.readout = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, len(DIRECTION_NAMES)))

    def forward(self, batch: Tensor, lengths: Tensor) -> Tensor:
        steps = batch.shape[1]
        valid = torch.arange(steps, device=batch.device)[None, :] < lengths[:, None]
        batch = batch * valid.unsqueeze(-1)
        padded = F.pad(batch, (0, 0, 0, -steps % self.patch_size))
        patches = self.patch(padded.reshape(batch.shape[0], -1, batch.shape[2] * self.patch_size))
        indices = torch.arange(patches.shape[1], device=batch.device)
        phases = indices[:, None] * self.frequencies
        patches = patches + torch.stack((phases.sin(), phases.cos()), dim=-1).flatten(1)
        padding = indices[None, :] * self.patch_size >= lengths[:, None]
        for layer in self.layers:
            patches = layer(patches, src_key_padding_mask=padding)
        context = patches.repeat_interleave(self.patch_size, dim=1)[:, :steps]
        return self.readout(context + self.local(batch)) * valid.unsqueeze(-1)


@dataclass(frozen=True)
class Candidate:
    name: str
    channels: int = 32
    dilations: tuple[int, ...] = DEFAULT_DILATIONS
    interval_power: float = 0.0
    attention: bool = False

    def build(self, features: int) -> nn.Module:
        if self.attention:
            return PatchEncoder(features, self.channels, 0.15)
        return TCNClassifier(features, self.channels, 0.15, self.dilations)


CANDIDATES = (
    Candidate("tcn-32"),
    Candidate("tcn-48", channels=48),
    Candidate("tcn-compact", dilations=tuple(range(1, 9))),
    Candidate("tcn-weight-0.5", interval_power=0.5),
    Candidate("tcn-weight-1", interval_power=1.0),
    Candidate("patch-transformer", channels=32, attention=True),
)
