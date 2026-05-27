from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal position encoding for frame-order information."""

    def __init__(self, d_model: int, max_len: int = 4096, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Transformer attention is permutation-invariant by itself. These fixed
        # sin/cos vectors give the encoder an explicit notion of frame order and
        # relative temporal distance within each window.
        positions = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(positions * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(positions * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is [batch, time, d_model]. Slice positions to the current window
        # length, then apply dropout as regularization.
        if x.shape[1] > self.pe.shape[1]:
            raise ValueError(f"Sequence length {x.shape[1]} exceeds max_len {self.pe.shape[1]}.")
        return self.dropout(x + self.pe[:, : x.shape[1]])


class TransformerRegressor(nn.Module):
    """Encoder-only Transformer for frame-level engagement regression."""

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.2,
        max_len: int = 4096,
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")

        # Project arbitrary feature dimensions into the Transformer model space.
        # This lets the same architecture consume raw/PCA/RP manifests with
        # different n_features values.
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = SinusoidalPositionalEncoding(d_model=d_model, max_len=max_len, dropout=dropout)

        # Encoder-only architecture: each frame attends to the rest of the
        # window, then a regression head maps encoded frames to scalar targets.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, output_dim),
        )

    def forward(self, x: torch.Tensor, frame_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Predict one engagement value per frame.

        Args:
            x: [batch, features, time], matching the TCN data-loader format.
            frame_mask: [batch, time], 1 for real frames and 0 for padding.
        """

        # Reorder to [batch, time, features] because PyTorch's batch-first
        # Transformer expects sequence length in the middle dimension.
        h = x.transpose(1, 2)
        h = self.input_projection(h)
        h = self.positional_encoding(h)

        # PyTorch uses True to mark positions that should be ignored. The data
        # loader uses 1 for real frames, so invert the mask for attention.
        key_padding_mask = None
        if frame_mask is not None:
            key_padding_mask = frame_mask <= 0

        h = self.encoder(h, src_key_padding_mask=key_padding_mask)
        out = self.head(h)
        if out.shape[-1] == 1:
            return out.squeeze(-1)
        return out
