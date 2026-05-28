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


def _shift_hidden_sequence(x: torch.Tensor, lag: int) -> torch.Tensor:
    """Return partner hidden states shifted so output at t can use partner at t + lag."""

    if lag == 0:
        return x

    shifted = torch.zeros_like(x)
    time_len = x.shape[-1]
    if abs(lag) >= time_len:
        return shifted

    if lag < 0:
        # Negative lag means past partner context. For lag=-25, prediction at
        # time t receives partner hidden state from t-25, with zeros at the
        # first 25 frames where that context does not exist.
        offset = abs(lag)
        shifted[..., offset:] = x[..., : time_len - offset]
    else:
        # Positive lag means future partner context. This is only appropriate
        # for offline experiments, where the full interaction is available.
        shifted[..., : time_len - lag] = x[..., lag:]
    return shifted


class PartnerLagTCNRegressor(nn.Module):
    """Two role-specific TCN encoders with separate heads using lagged partner states."""

    def __init__(
        self,
        n_features_per_role: int,
        hidden_channels: int = 64,
        levels: int = 4,
        kernel_size: int = 5,
        dropout: float = 0.2,
        partner_lags: tuple[int, ...] = (0,),
    ) -> None:
        super().__init__()
        self.n_features_per_role = n_features_per_role
        self.partner_lags = tuple(partner_lags)

        def make_encoder() -> nn.Sequential:
            # Each role gets its own temporal encoder. The structure is the
            # same, but the weights are not shared, so novice and expert can
            # learn different temporal representations.
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

        # Each head sees target-role hidden state plus one shifted partner
        # hidden sequence per requested lag. Heads are separate because the
        # novice and expert engagement mappings may differ.
        head_channels = hidden_channels * (1 + len(self.partner_lags))
        self.novice_head = nn.Conv1d(head_channels, 1, kernel_size=1)
        self.expert_head = nn.Conv1d(head_channels, 1, kernel_size=1)

    def _head_input(self, target_hidden: torch.Tensor, partner_hidden: torch.Tensor) -> torch.Tensor:
        """Concatenate target hidden state with lagged partner hidden states."""

        partner_context = [_shift_hidden_sequence(partner_hidden, lag) for lag in self.partner_lags]
        return torch.cat([target_hidden] + partner_context, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input is [batch, 2 * features_per_role, time] in role order
        # [novice_features, expert_features]. Output is [batch, time, 2].
        novice_x = x[:, : self.n_features_per_role]
        expert_x = x[:, self.n_features_per_role : 2 * self.n_features_per_role]

        novice_hidden = self.novice_encoder(novice_x)
        expert_hidden = self.expert_encoder(expert_x)

        novice_in = self._head_input(novice_hidden, expert_hidden)
        expert_in = self._head_input(expert_hidden, novice_hidden)
        novice_pred = self.novice_head(novice_in)
        expert_pred = self.expert_head(expert_in)
        return torch.cat([novice_pred, expert_pred], dim=1).transpose(1, 2)


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


def _pooled_past_context(x: torch.Tensor, pool_frames: int, exclude_current_frame: bool = True) -> torch.Tensor:
    """Mean-pool a hidden sequence over the previous pool_frames for every t."""

    if pool_frames <= 0:
        raise ValueError("pool_frames must be positive.")

    batch, channels, time_len = x.shape
    if exclude_current_frame:
        # Shift right by one frame so context at t summarizes frames before t.
        # The first frame has no past and therefore receives a zero context.
        shifted = torch.zeros_like(x)
        shifted[..., 1:] = x[..., :-1]
    else:
        shifted = x

    # Cumulative sums make causal window means efficient. Padding one zero
    # column at the front lets us subtract arbitrary window starts cleanly.
    prefix = torch.cat([torch.zeros(batch, channels, 1, device=x.device, dtype=x.dtype), torch.cumsum(shifted, dim=-1)], dim=-1)
    end = torch.arange(time_len, device=x.device) + 1
    start = torch.clamp(end - pool_frames, min=0)
    pooled_sum = prefix.index_select(-1, end) - prefix.index_select(-1, start)
    if exclude_current_frame:
        # At query frame t there are t real past frames available. Clamp at one
        # so the first all-zero context remains numerically well-defined.
        counts_1d = torch.clamp(torch.minimum(torch.arange(time_len, device=x.device), torch.tensor(pool_frames, device=x.device)), min=1)
    else:
        counts_1d = torch.clamp(end - start, min=1)
    counts = counts_1d.to(dtype=x.dtype).view(1, 1, time_len)
    return pooled_sum / counts


class GatedPooledTCNRegressor(nn.Module):
    """Separate role TCNs with learned gates over pooled past partner context."""

    def __init__(
        self,
        n_features_per_role: int,
        hidden_channels: int = 64,
        levels: int = 4,
        kernel_size: int = 5,
        dropout: float = 0.2,
        partner_pool_frames: int = 750,
        exclude_current_frame: bool = True,
        gate_type: str = "scalar",
    ) -> None:
        super().__init__()
        if gate_type not in {"scalar", "channel"}:
            raise ValueError(f"Unsupported gate_type: {gate_type}")
        self.n_features_per_role = n_features_per_role
        self.partner_pool_frames = partner_pool_frames
        self.exclude_current_frame = exclude_current_frame
        self.gate_type = gate_type

        def make_encoder() -> nn.Sequential:
            # Role-specific encoders keep beginner/novice and expert temporal
            # representations separate before interaction is introduced.
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

        gate_out_channels = 1 if gate_type == "scalar" else hidden_channels
        # Gates are learned from target hidden state and pooled partner context.
        # A scalar gate is easiest to interpret; a channel gate is more flexible.
        self.novice_gate = nn.Conv1d(hidden_channels * 2, gate_out_channels, kernel_size=1)
        self.expert_gate = nn.Conv1d(hidden_channels * 2, gate_out_channels, kernel_size=1)

        # Separate heads keep the role-specific interpretation clear: partner
        # context may affect novice and expert engagement differently.
        self.novice_head = nn.Conv1d(hidden_channels, 1, kernel_size=1)
        self.expert_head = nn.Conv1d(hidden_channels, 1, kernel_size=1)

    def _fuse(self, target_hidden: torch.Tensor, partner_hidden: torch.Tensor, gate_layer: nn.Conv1d) -> tuple[torch.Tensor, torch.Tensor]:
        """Pool partner history, learn a gate, and fuse it with target hidden state."""

        partner_context = _pooled_past_context(
            partner_hidden,
            pool_frames=self.partner_pool_frames,
            exclude_current_frame=self.exclude_current_frame,
        )
        gate = torch.sigmoid(gate_layer(torch.cat([target_hidden, partner_context], dim=1)))
        fused = target_hidden + gate * partner_context
        return fused, gate

    def forward(self, x: torch.Tensor, return_gates: bool = False) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # Input is [batch, 2 * features_per_role, time] with role order
        # [novice_features, expert_features]. Output is [batch, time, 2].
        novice_x = x[:, : self.n_features_per_role]
        expert_x = x[:, self.n_features_per_role : 2 * self.n_features_per_role]

        novice_hidden = self.novice_encoder(novice_x)
        expert_hidden = self.expert_encoder(expert_x)

        novice_fused, novice_gate = self._fuse(novice_hidden, expert_hidden, self.novice_gate)
        expert_fused, expert_gate = self._fuse(expert_hidden, novice_hidden, self.expert_gate)

        novice_pred = self.novice_head(novice_fused)
        expert_pred = self.expert_head(expert_fused)
        pred = torch.cat([novice_pred, expert_pred], dim=1).transpose(1, 2)
        if not return_gates:
            return pred
        return pred, {
            "novice_gate": novice_gate,
            "expert_gate": expert_gate,
        }
