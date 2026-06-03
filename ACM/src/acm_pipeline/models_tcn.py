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


class IndependentDyadicTCNRegressor(nn.Module):
    """Shared person-level TCN applied independently to novice and expert."""

    def __init__(
        self,
        n_features_per_role: int,
        hidden_channels: int = 64,
        levels: int = 4,
        kernel_size: int = 5,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.n_features_per_role = n_features_per_role
        self.person_tcn = TCNRegressor(
            input_dim=n_features_per_role,
            hidden_channels=hidden_channels,
            levels=levels,
            kernel_size=kernel_size,
            dropout=dropout,
            output_dim=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        novice_x = x[:, : self.n_features_per_role]
        expert_x = x[:, self.n_features_per_role : 2 * self.n_features_per_role]
        novice_pred = self.person_tcn(novice_x)
        expert_pred = self.person_tcn(expert_x)
        return torch.stack([novice_pred, expert_pred], dim=2)


class DyadicTCNRegressor(nn.Module):
    """TCN for dyadic engagement with one shared two-channel prediction head."""

    def __init__(
        self,
        input_dim: int,
        hidden_channels: int = 64,
        levels: int = 4,
        kernel_size: int = 5,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        # The shared dyadic baseline uses one encoder over the paired input and
        # one lightweight head that predicts both roles jointly.
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
        self.head = nn.Conv1d(hidden_channels, 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input is [batch, 2 * features, time]. Output is always
        # [batch, time, 2] with channels [novice, expert].
        return self.head(self.tcn(x)).transpose(1, 2)


class RoleAttentionTCNRegressor(nn.Module):
    """Separate role TCNs with role-specific self/partner/joint attention heads."""

    def __init__(
        self,
        n_features_per_role: int,
        hidden_channels: int = 64,
        levels: int = 4,
        kernel_size: int = 5,
        dropout: float = 0.2,
        attention_context: str = "joint",
        attention_heads: int = 4,
        attention_past_frames: int | None = 1500,
        exclude_current_frame: bool = False,
    ) -> None:
        super().__init__()
        if attention_context not in {"self", "partner", "joint"}:
            raise ValueError(f"Unsupported attention_context: {attention_context}")
        if hidden_channels % attention_heads != 0:
            raise ValueError("hidden_channels must be divisible by attention_heads.")
        self.n_features_per_role = n_features_per_role
        self.attention_context = attention_context
        self.attention_past_frames = attention_past_frames
        self.exclude_current_frame = exclude_current_frame

        def make_encoder() -> nn.Sequential:
            # Encoders are role-specific: novice and expert have the same TCN
            # structure, but separate weights.
            channels = [n_features_per_role] + [hidden_channels] * levels
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
            return nn.Sequential(*blocks)

        self.novice_encoder = make_encoder()
        self.expert_encoder = make_encoder()

        # Attention modules are also role-specific. This lets novice and expert
        # learn different temporal/context-selection patterns.
        self.novice_attention = nn.MultiheadAttention(hidden_channels, attention_heads, dropout=dropout, batch_first=True)
        self.expert_attention = nn.MultiheadAttention(hidden_channels, attention_heads, dropout=dropout, batch_first=True)

        # Each head receives the target hidden state and the attended context.
        # Keeping target_hidden in the head preserves a direct role-specific
        # pathway even if attention learns to down-weight context.
        self.novice_head = nn.Conv1d(hidden_channels * 2, 1, kernel_size=1)
        self.expert_head = nn.Conv1d(hidden_channels * 2, 1, kernel_size=1)

    def _attention_mask(self, time_len: int, n_sources: int, device: torch.device) -> torch.Tensor:
        """Build a bool mask where True entries are hidden from attention."""

        query_t = torch.arange(time_len, device=device).unsqueeze(1)
        source_t = torch.arange(time_len, device=device).repeat(n_sources).unsqueeze(0)
        lag = query_t - source_t
        allowed = lag >= 0
        if self.exclude_current_frame:
            allowed &= lag > 0
        if self.attention_past_frames is not None:
            allowed &= lag <= self.attention_past_frames

        # The first frame has no past when current-frame context is excluded.
        # Unmask one position to avoid all-masked attention rows and resulting
        # NaNs; downstream diagnostics will still show the lag/source used.
        empty_rows = ~torch.any(allowed, dim=1)
        if torch.any(empty_rows):
            allowed[empty_rows, 0] = True
        return ~allowed

    def _context_sequence(self, target_hidden: torch.Tensor, partner_hidden: torch.Tensor) -> tuple[torch.Tensor, list[str]]:
        """Select self, partner, or joint keys/values for one role's attention."""

        if self.attention_context == "self":
            return target_hidden, ["self"]
        if self.attention_context == "partner":
            return partner_hidden, ["partner"]
        return torch.cat([target_hidden, partner_hidden], dim=1), ["self", "partner"]

    def _attend(
        self,
        attention: nn.MultiheadAttention,
        target_hidden_bt: torch.Tensor,
        partner_hidden_bt: torch.Tensor,
        need_weights: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[str]]:
        """Run one role-specific attention head over the selected context."""

        context, source_blocks = self._context_sequence(target_hidden_bt, partner_hidden_bt)
        mask = self._attention_mask(target_hidden_bt.shape[1], len(source_blocks), target_hidden_bt.device)
        attended, weights = attention(
            query=target_hidden_bt,
            key=context,
            value=context,
            attn_mask=mask,
            need_weights=need_weights,
            average_attn_weights=True,
        )
        return attended, weights if need_weights else None, source_blocks

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> torch.Tensor | tuple[torch.Tensor, dict[str, object]]:
        # Input is [batch, 2 * features_per_role, time] in role order
        # [novice_features, expert_features].
        novice_x = x[:, : self.n_features_per_role]
        expert_x = x[:, self.n_features_per_role : 2 * self.n_features_per_role]

        novice_hidden = self.novice_encoder(novice_x)
        expert_hidden = self.expert_encoder(expert_x)
        novice_bt = novice_hidden.transpose(1, 2)
        expert_bt = expert_hidden.transpose(1, 2)

        novice_attended, novice_weights, novice_sources = self._attend(self.novice_attention, novice_bt, expert_bt, return_attention)
        expert_attended, expert_weights, expert_sources = self._attend(self.expert_attention, expert_bt, novice_bt, return_attention)

        novice_head_in = torch.cat([novice_hidden, novice_attended.transpose(1, 2)], dim=1)
        expert_head_in = torch.cat([expert_hidden, expert_attended.transpose(1, 2)], dim=1)
        novice_pred = self.novice_head(novice_head_in)
        expert_pred = self.expert_head(expert_head_in)
        pred = torch.cat([novice_pred, expert_pred], dim=1).transpose(1, 2)

        if not return_attention:
            return pred
        return pred, {
            "novice_weights": novice_weights,
            "expert_weights": expert_weights,
            "novice_sources": novice_sources,
            "expert_sources": expert_sources,
        }
