"""PinSoRo adaptations of the three active TCN architecture variants."""

from __future__ import annotations

import torch
from torch import nn

from src.acm_pipeline.models_tcn import TemporalBlock


TASK_CLASSES = 4
SOCIAL_CLASSES = 5


def _make_encoder(
    input_dim: int,
    hidden_channels: int,
    levels: int,
    kernel_size: int,
    dropout: float,
    causal: bool,
) -> nn.Sequential:
    channels = [input_dim] + [hidden_channels] * levels
    return nn.Sequential(
        *[
            TemporalBlock(channels[idx], channels[idx + 1], kernel_size, 2**idx, dropout, causal=causal)
            for idx in range(levels)
        ]
    )


class _SharedClassificationHeads(nn.Module):
    def __init__(self, hidden_channels: int) -> None:
        super().__init__()
        self.task_head = nn.Conv1d(hidden_channels, TASK_CLASSES, kernel_size=1)
        self.social_head = nn.Conv1d(hidden_channels, SOCIAL_CLASSES, kernel_size=1)

    def forward(self, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "task": self.task_head(hidden).transpose(1, 2),
            "social": self.social_head(hidden).transpose(1, 2),
        }


class _DomainSocialClassificationHeads(nn.Module):
    def __init__(self, hidden_channels: int) -> None:
        super().__init__()
        self.task_head = nn.Conv1d(hidden_channels, TASK_CLASSES, kernel_size=1)
        self.cc_social_head = nn.Conv1d(hidden_channels, SOCIAL_CLASSES, kernel_size=1)
        self.cr_social_head = nn.Conv1d(hidden_channels, SOCIAL_CLASSES, kernel_size=1)

    def forward(
        self, hidden: torch.Tensor, domain_ids: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if domain_ids.ndim != 1 or domain_ids.shape[0] != hidden.shape[0]:
            raise ValueError(
                "domain_ids must contain one domain ID per flattened role: "
                f"hidden={tuple(hidden.shape)} domain_ids={tuple(domain_ids.shape)}"
            )
        if not torch.all((domain_ids == 0) | (domain_ids == 1)):
            raise ValueError("Domain-specific social heads require CC=0 or CR=1.")
        cc_social = self.cc_social_head(hidden)
        cr_social = self.cr_social_head(hidden)
        social = torch.where(
            (domain_ids == 0).reshape(-1, 1, 1),
            cc_social,
            cr_social,
        )
        return {
            "task": self.task_head(hidden).transpose(1, 2),
            "social": social.transpose(1, 2),
        }


class PinSoRoIndividualTCN(nn.Module):
    """Shared person-level TCN applied without cross-role information."""

    def __init__(self, n_features_per_role: int, hidden_channels: int, levels: int, kernel_size: int, dropout: float, causal_tcn: bool = False) -> None:
        super().__init__()
        self.encoder = _make_encoder(n_features_per_role, hidden_channels, levels, kernel_size, dropout, causal_tcn)
        self.heads = _SharedClassificationHeads(hidden_channels)

    def forward(
        self, x: torch.Tensor, domain_ids: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        batch, roles, features, time = x.shape
        hidden = self.encoder(x.reshape(batch * roles, features, time))
        logits = self.heads(hidden)
        return {
            head: value.reshape(batch, roles, time, value.shape[-1])
            for head, value in logits.items()
        }


class PinSoRoDyadicSharedTCN(nn.Module):
    """One dyadic encoder with one joint role-output head per target."""

    def __init__(self, n_features_per_role: int, hidden_channels: int, levels: int, kernel_size: int, dropout: float, causal_tcn: bool = False) -> None:
        super().__init__()
        self.encoder = _make_encoder(2 * n_features_per_role, hidden_channels, levels, kernel_size, dropout, causal_tcn)
        self.task_head = nn.Conv1d(hidden_channels, 2 * TASK_CLASSES, kernel_size=1)
        self.social_head = nn.Conv1d(hidden_channels, 2 * SOCIAL_CLASSES, kernel_size=1)

    @staticmethod
    def _reshape(logits: torch.Tensor, roles: int, classes: int) -> torch.Tensor:
        batch, _, time = logits.shape
        return logits.reshape(batch, roles, classes, time).permute(0, 1, 3, 2)

    def forward(
        self, x: torch.Tensor, domain_ids: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        if x.shape[1] != 2:
            raise ValueError(f"Dyadic model requires two roles, got shape {tuple(x.shape)}")
        batch, roles, features, time = x.shape
        hidden = self.encoder(x.reshape(batch, roles * features, time))
        return {
            "task": self._reshape(self.task_head(hidden), roles, TASK_CLASSES),
            "social": self._reshape(self.social_head(hidden), roles, SOCIAL_CLASSES),
        }


class PinSoRoAttentionTCN(nn.Module):
    """Shared role encoder and attention over synchronized self/partner history."""

    def __init__(
        self,
        n_features_per_role: int,
        hidden_channels: int,
        levels: int,
        kernel_size: int,
        dropout: float,
        attention_heads: int = 4,
        causal_tcn: bool = False,
        causal_attention: bool = False,
        domain_social_heads: bool = False,
    ) -> None:
        super().__init__()
        if hidden_channels % attention_heads != 0:
            raise ValueError("hidden_channels must be divisible by attention_heads.")
        self.encoder = _make_encoder(n_features_per_role, hidden_channels, levels, kernel_size, dropout, causal_tcn)
        self.attention = nn.MultiheadAttention(hidden_channels, attention_heads, dropout=dropout, batch_first=True)
        self.domain_social_heads = domain_social_heads
        self.heads = (
            _DomainSocialClassificationHeads(2 * hidden_channels)
            if domain_social_heads
            else _SharedClassificationHeads(2 * hidden_channels)
        )
        self.causal_attention = causal_attention

    def _attention_mask(self, time: int, roles: int, device: torch.device) -> torch.Tensor | None:
        if not self.causal_attention:
            return None
        query_time = torch.arange(time, device=device).unsqueeze(1)
        source_time = torch.arange(time, device=device).repeat(roles).unsqueeze(0)
        return source_time > query_time

    def forward(
        self, x: torch.Tensor, domain_ids: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        if x.shape[1] != 2:
            raise ValueError(f"Attention model requires two roles, got shape {tuple(x.shape)}")
        batch, roles, features, time = x.shape
        encoded = self.encoder(x.reshape(batch * roles, features, time))
        encoded = encoded.reshape(batch, roles, -1, time)
        role_hidden = encoded.permute(0, 1, 3, 2)
        context = role_hidden.reshape(batch, roles * time, -1)
        attention_mask = self._attention_mask(time, roles, x.device)

        attended = []
        for role_idx in range(roles):
            role_context, _ = self.attention(
                role_hidden[:, role_idx],
                context,
                context,
                attn_mask=attention_mask,
                need_weights=False,
            )
            attended.append(role_context)
        attended_bt = torch.stack(attended, dim=1)
        head_input = torch.cat([role_hidden, attended_bt], dim=-1)
        flat = head_input.permute(0, 1, 3, 2).reshape(batch * roles, -1, time)
        if self.domain_social_heads:
            if domain_ids is None:
                raise ValueError("domain_ids are required with domain-specific social heads.")
            logits = self.heads(flat, domain_ids.repeat_interleave(roles))
        else:
            logits = self.heads(flat)
        return {
            head: value.reshape(batch, roles, time, value.shape[-1])
            for head, value in logits.items()
        }


def build_pinsoro_tcn(
    model_name: str,
    n_features_per_role: int,
    hidden_channels: int = 64,
    levels: int = 4,
    kernel_size: int = 5,
    dropout: float = 0.2,
    attention_heads: int = 4,
    causal_tcn: bool = False,
    causal_attention: bool = False,
    domain_social_heads: bool = False,
) -> nn.Module:
    if domain_social_heads and model_name != "attention":
        raise ValueError("Domain-specific social heads currently require the attention model.")
    common = {
        "n_features_per_role": n_features_per_role,
        "hidden_channels": hidden_channels,
        "levels": levels,
        "kernel_size": kernel_size,
        "dropout": dropout,
        "causal_tcn": causal_tcn,
    }
    if model_name == "simple":
        return PinSoRoIndividualTCN(**common)
    if model_name == "dyadic_shared":
        return PinSoRoDyadicSharedTCN(**common)
    if model_name == "attention":
        return PinSoRoAttentionTCN(
            **common,
            attention_heads=attention_heads,
            causal_attention=causal_attention,
            domain_social_heads=domain_social_heads,
        )
    raise ValueError(f"Unknown PinSoRo model: {model_name}")
