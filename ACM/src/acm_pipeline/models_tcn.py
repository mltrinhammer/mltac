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
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
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
        self.head = nn.Conv1d(hidden_channels, output_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input is [batch, features, time]. Role-level output stays [batch,
        # time]; dyadic output becomes [batch, time, 2].
        out = self.head(self.tcn(x))
        if self.output_dim == 1:
            return out.squeeze(1)
        return out.transpose(1, 2)


class DyadicTCNRegressor(nn.Module):
    """TCN for dyadic engagement with configurable shared or role-specific heads."""

    def __init__(
        self,
        input_dim: int,
        hidden_channels: int = 64,
        levels: int = 4,
        kernel_size: int = 5,
        dropout: float = 0.2,
        head_type: str = "shared",
    ) -> None:
        super().__init__()
        if head_type not in {"shared", "role_specific"}:
            raise ValueError(f"Unsupported dyadic head_type: {head_type}")
        self.head_type = head_type

        # The encoder is identical for both variants. This keeps the
        # comparison focused on the prediction head rather than temporal
        # capacity or receptive field.
        channels = [input_dim] + [hidden_channels] * levels
        blocks = []
        for idx in range(levels):
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

        if head_type == "shared":
            # One head predicts both target channels jointly. The two output
            # filters can still differ, but they live in the same module.
            self.head = nn.Conv1d(hidden_channels, 2, kernel_size=1)
            self.role_heads = None
        else:
            # Separate lightweight heads test whether novice and expert need
            # different mappings from the shared dyadic temporal encoding.
            self.head = None
            self.role_heads = nn.ModuleList(
                [
                    nn.Conv1d(hidden_channels, 1, kernel_size=1),
                    nn.Conv1d(hidden_channels, 1, kernel_size=1),
                ]
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input is [batch, 2 * features, time]. Output is always
        # [batch, time, 2] with channels [novice, expert].
        encoded = self.tcn(x)
        if self.head_type == "shared":
            out = self.head(encoded)
        else:
            out = torch.cat([head(encoded) for head in self.role_heads], dim=1)
        return out.transpose(1, 2)
