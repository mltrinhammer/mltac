from __future__ import annotations

import torch
from torch import nn


def _match_time(x: torch.Tensor, target_len: int) -> torch.Tensor:
    """Crop/pad a convolution output so residual additions keep the same length."""

    if x.shape[-1] == target_len:
        return x
    if x.shape[-1] > target_len:
        extra = x.shape[-1] - target_len
        left = extra // 2
        return x[..., left : left + target_len]
    pad_total = target_len - x.shape[-1]
    left = pad_total // 2
    right = pad_total - left
    return nn.functional.pad(x, (left, right))


class TemporalBlock(nn.Module):
    """Residual dilated Conv1d block for offline sequence regression."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2

        # The convolutions operate over time; feature dimensions enter as
        # channels. GroupNorm keeps training stable for small batches.
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            nn.GroupNorm(1, out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            nn.GroupNorm(1, out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = _match_time(self.net(x), x.shape[-1])
        return y + self.residual(x)


class TCNRegressor(nn.Module):
    """Small TCN that predicts one engagement value per frame."""

    def __init__(
        self,
        input_dim: int,
        hidden_channels: int = 64,
        levels: int = 4,
        kernel_size: int = 5,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        channels = [input_dim] + [hidden_channels] * levels
        blocks = []
        for idx in range(levels):
            # Dilations expand temporal context without pooling, so predictions
            # remain frame-aligned with the target sequence.
            blocks.append(
                TemporalBlock(
                    in_channels=channels[idx],
                    out_channels=channels[idx + 1],
                    kernel_size=kernel_size,
                    dilation=2**idx,
                    dropout=dropout,
                )
            )
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Conv1d(hidden_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input is [batch, features, time]; output is [batch, time].
        return self.head(self.tcn(x)).squeeze(1)

