"""Group-level multimodal engagement models."""

from __future__ import annotations

import math

import torch
from torch import nn

from src.acm_pipeline.models_tcn import TemporalBlock


class GroupRoleWiseMultimodalFusion(nn.Module):
    """Shared per-participant multimodal projection and fusion."""

    def __init__(
        self,
        modality_dims: dict[str, int],
        d_shared: int,
        fusion_mode: str = "gated",
        modality_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not modality_dims:
            raise ValueError("modality_dims must not be empty.")
        if fusion_mode not in {"gated", "concat"}:
            raise ValueError(f"Unsupported fusion_mode: {fusion_mode}")
        if modality_dropout < 0.0 or modality_dropout >= 1.0:
            raise ValueError("modality_dropout must be in [0, 1).")

        self.modality_order = tuple(modality_dims.keys())
        self.modality_dims = dict(modality_dims)
        self.d_shared = d_shared
        self.fusion_mode = fusion_mode
        self.modality_dropout = modality_dropout
        gate_hidden = max(1, d_shared // 2)
        self.projections = nn.ModuleDict({name: nn.Linear(dim, d_shared) for name, dim in self.modality_dims.items()})
        self.gates = nn.ModuleDict(
            {
                name: nn.Sequential(nn.Linear(d_shared, gate_hidden), nn.ReLU(), nn.Linear(gate_hidden, 1))
                for name in self.modality_order
            }
        )

    @property
    def fused_features_per_role(self) -> int:
        if self.fusion_mode == "gated":
            return self.d_shared
        return self.d_shared * len(self.modality_order)

    def _sample_keep_mask(self, device: torch.device) -> torch.Tensor | None:
        if not self.training or self.modality_dropout <= 0.0 or len(self.modality_order) <= 1:
            return None
        keep = torch.rand(len(self.modality_order), device=device) >= self.modality_dropout
        if not torch.any(keep):
            keep[torch.randint(len(self.modality_order), size=(1,), device=device)] = True
        return keep

    def forward(self, x_modalities: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, object]]:
        projected: list[torch.Tensor] = []
        first_device: torch.device | None = None
        for modality_name in self.modality_order:
            x = x_modalities[modality_name]
            if first_device is None:
                first_device = x.device
            expected_dim = self.modality_dims[modality_name]
            if x.ndim != 4:
                raise ValueError(f"Expected modality {modality_name!r} to be [B,N,F,T], got {tuple(x.shape)}")
            if x.shape[2] != expected_dim:
                raise ValueError(f"Modality {modality_name!r} expected {expected_dim} features, got {x.shape[2]}")
            bsz, n_roles, _features, time_len = x.shape
            flat = x.permute(0, 1, 3, 2).reshape(bsz * n_roles, time_len, expected_dim)
            projected.append(self.projections[modality_name](flat).reshape(bsz, n_roles, time_len, self.d_shared))

        assert first_device is not None
        keep_mask = self._sample_keep_mask(first_device)
        if keep_mask is not None:
            projected = [feature if bool(keep_mask[idx].item()) else torch.zeros_like(feature) for idx, feature in enumerate(projected)]

        if self.fusion_mode == "concat":
            fused = torch.cat(projected, dim=-1).permute(0, 1, 3, 2)
            weights = None
        else:
            logits = torch.cat([self.gates[name](feature) for name, feature in zip(self.modality_order, projected)], dim=-1)
            if keep_mask is not None:
                logits = logits.masked_fill(~keep_mask.view(1, 1, 1, -1), -1e9)
            weights = torch.softmax(logits, dim=-1)
            fused_bt = sum(weights[..., idx : idx + 1] * feature for idx, feature in enumerate(projected))
            fused = fused_bt.permute(0, 1, 3, 2)

        return fused, {
            "modality_order": list(self.modality_order),
            "fusion_mode": self.fusion_mode,
            "weights": weights,
        }


class MeanPoolGroupMultimodalTCNRegressor(nn.Module):
    """Encode each participant, then apply linear post-prediction partner interaction."""

    def __init__(
        self,
        modality_dims: dict[str, int],
        fusion_channels: int = 64,
        fusion_mode: str = "gated",
        modality_dropout: float = 0.0,
        hidden_channels: int = 64,
        levels: int = 4,
        kernel_size: int = 5,
        dropout: float = 0.2,
        encoder_sharing: str = "shared",
        max_role_encoders: int = 8,
        prediction_head_sharing: str = "shared",
        prediction_interaction_scale: float = 0.1,
        metadata_dim: int = 0,
        metadata_embedding_dim: int = 16,
        metadata_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if encoder_sharing not in {"shared", "separate"}:
            raise ValueError(f"Unsupported encoder_sharing: {encoder_sharing}")
        if prediction_head_sharing not in {"shared", "role_specific"}:
            raise ValueError(f"Unsupported prediction_head_sharing: {prediction_head_sharing}")
        if max_role_encoders < 1:
            raise ValueError("max_role_encoders must be positive.")
        self.encoder_sharing = encoder_sharing
        self.max_role_encoders = max_role_encoders
        self.prediction_head_sharing = prediction_head_sharing
        self.prediction_interaction_scale = float(prediction_interaction_scale)
        self.metadata_dim = int(metadata_dim)
        self.metadata_embedding_dim = int(metadata_embedding_dim) if self.metadata_dim > 0 else 0
        self.fusion = GroupRoleWiseMultimodalFusion(
            modality_dims=modality_dims,
            d_shared=fusion_channels,
            fusion_mode=fusion_mode,
            modality_dropout=modality_dropout,
        )
        fused_dim = self.fusion.fused_features_per_role

        def make_encoder() -> nn.Sequential:
            channels = [fused_dim] + [hidden_channels] * levels
            return nn.Sequential(
                *[
                    TemporalBlock(
                        in_channels=channels[idx],
                        out_channels=channels[idx + 1],
                        kernel_size=kernel_size,
                        dilation=2**idx,
                        dropout=dropout,
                    )
                    for idx in range(levels)
                ]
            )

        if encoder_sharing == "shared":
            self.person_encoder = make_encoder()
            self.role_encoders = None
        else:
            self.person_encoder = None
            self.role_encoders = nn.ModuleList([make_encoder() for _ in range(max_role_encoders)])
        if self.metadata_dim > 0:
            self.metadata_encoder = nn.Sequential(
                nn.Linear(self.metadata_dim, self.metadata_embedding_dim),
                nn.ReLU(),
                nn.Dropout(metadata_dropout),
            )
        else:
            self.metadata_encoder = None
        self.legacy_mean_partner = (
            self.prediction_interaction_scale == 0.0
            and self.metadata_dim == 0
            and encoder_sharing == "shared"
            and prediction_head_sharing == "shared"
        )
        if self.legacy_mean_partner:
            self.head = nn.Conv1d(hidden_channels * 2, 1, kernel_size=1)
            self.role_heads = None
            self.prediction_interaction = None
        else:
            head_channels = hidden_channels + self.metadata_embedding_dim
            if prediction_head_sharing == "shared":
                self.head = nn.Conv1d(head_channels, 1, kernel_size=1)
                self.role_heads = None
            else:
                self.head = None
                self.role_heads = nn.ModuleList([nn.Conv1d(head_channels, 1, kernel_size=1) for _ in range(max_role_encoders)])
            self.prediction_interaction = nn.Linear(2, 1)

    def forward(
        self,
        x_modalities: dict[str, torch.Tensor],
        role_mask: torch.Tensor | None = None,
        metadata: torch.Tensor | None = None,
        return_gate_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, object]]:
        fused, fusion_info = self.fusion(x_modalities)
        bsz, n_roles, channels, time_len = fused.shape
        if self.encoder_sharing == "shared":
            assert self.person_encoder is not None
            hidden = self.person_encoder(fused.reshape(bsz * n_roles, channels, time_len))
            hidden = hidden.reshape(bsz, n_roles, hidden.shape[1], time_len)
        else:
            assert self.role_encoders is not None
            if n_roles > len(self.role_encoders):
                raise RuntimeError(f"Need {n_roles} role encoders, only configured {len(self.role_encoders)}.")
            hidden_roles = [self.role_encoders[idx](fused[:, idx]) for idx in range(n_roles)]
            hidden = torch.stack(hidden_roles, dim=1)

        if self.metadata_encoder is not None:
            if metadata is None:
                raise ValueError("metadata tensor is required when metadata_dim > 0.")
            if metadata.shape[:2] != (bsz, n_roles) or metadata.shape[2] != self.metadata_dim:
                raise ValueError(f"Expected metadata [B,N,{self.metadata_dim}], got {tuple(metadata.shape)}")
            meta = self.metadata_encoder(metadata.to(device=hidden.device, dtype=hidden.dtype))
            meta = meta.unsqueeze(-1).expand(-1, -1, -1, time_len)
            hidden = torch.cat([hidden, meta], dim=2)

        if role_mask is None:
            role_mask = torch.ones(bsz, n_roles, device=hidden.device, dtype=hidden.dtype)
        role_weights = role_mask.to(device=hidden.device, dtype=hidden.dtype)

        if self.legacy_mean_partner:
            role_weights_4d = role_weights.view(bsz, n_roles, 1, 1)
            group_sum = torch.sum(hidden * role_weights_4d, dim=1, keepdim=True)
            partner_count_hidden = torch.clamp(torch.sum(role_weights_4d, dim=1, keepdim=True) - role_weights_4d, min=1.0)
            partner_hidden = (group_sum - hidden * role_weights_4d) / partner_count_hidden
            partner_hidden = partner_hidden * (role_weights_4d > 0).to(hidden.dtype)
            head_input = torch.cat([hidden, partner_hidden], dim=2).reshape(bsz * n_roles, hidden.shape[2] * 2, time_len)
            pred = self.head(head_input).reshape(bsz, n_roles, time_len).permute(0, 2, 1)
            pred = pred * role_weights.unsqueeze(1)
            if not return_gate_weights:
                return pred
            return pred, fusion_info

        if self.prediction_head_sharing == "shared":
            assert self.head is not None
            base_pred = self.head(hidden.reshape(bsz * n_roles, hidden.shape[2], time_len))
            base_pred = base_pred.reshape(bsz, n_roles, time_len).permute(0, 2, 1)
        else:
            assert self.role_heads is not None
            if n_roles > len(self.role_heads):
                raise RuntimeError(f"Need {n_roles} role heads, only configured {len(self.role_heads)}.")
            role_preds = [self.role_heads[idx](hidden[:, idx]).squeeze(1) for idx in range(n_roles)]
            base_pred = torch.stack(role_preds, dim=2)
        pred_weights = role_weights.unsqueeze(1)
        pred_sum = torch.sum(base_pred * pred_weights, dim=2, keepdim=True)
        partner_count = torch.clamp(torch.sum(pred_weights, dim=2, keepdim=True) - pred_weights, min=1.0)
        partner_pred = (pred_sum - base_pred * pred_weights) / partner_count
        partner_pred = partner_pred * (pred_weights > 0).to(base_pred.dtype)
        interaction_input = torch.stack([base_pred, partner_pred], dim=-1)
        residual = self.prediction_interaction(interaction_input).squeeze(-1)
        pred = base_pred + self.prediction_interaction_scale * residual
        pred = pred * pred_weights
        if not return_gate_weights:
            return pred
        return pred, fusion_info


class DomainPromptModule(nn.Module):
    """Learnable domain-specific vectors prepended to temporal features.

    Each domain (dataset) gets ``n_prompt_tokens`` learnable vectors that are
    prepended along the time axis, conditioning downstream layers on the data's
    cultural/linguistic origin.
    """

    def __init__(
        self,
        n_domains: int,
        prompt_dim: int,
        n_prompt_tokens: int = 4,
    ) -> None:
        super().__init__()
        self.n_domains = n_domains
        self.n_prompt_tokens = n_prompt_tokens
        self.prompts = nn.Parameter(
            torch.randn(n_domains, n_prompt_tokens, prompt_dim) * 0.02
        )

    def forward(
        self,
        x: torch.Tensor,
        domain_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Prepend domain prompt tokens to the time axis.

        Args:
            x: ``[B, N_roles, C, T]`` fused feature tensor.
            domain_ids: ``[B]`` integer domain index per sample.

        Returns:
            ``[B, N_roles, C, T + n_prompt_tokens]`` with prompts prepended.
        """
        bsz, n_roles, channels, time_len = x.shape
        prompt = self.prompts[domain_ids]  # [B, n_prompt_tokens, C]
        prompt = prompt.unsqueeze(1).expand(bsz, n_roles, -1, -1)  # [B, N, P, C]
        x_bt = x.permute(0, 1, 3, 2)  # [B, N, T, C]
        x_prompted = torch.cat([prompt, x_bt], dim=2)  # [B, N, P+T, C]
        return x_prompted.permute(0, 1, 3, 2)  # [B, N, C, P+T]


class ParallelCrossAttention(nn.Module):
    """Parallel reactive/anticipatory cross-attention for multi-participant groups.

    Each participant queries the mean of all other participants' states
    (computed via ``role_mask``), separately for reactive (forward BiLSTM)
    and anticipatory (backward BiLSTM) representations.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.reactive_cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.reactive_norm = nn.LayerNorm(d_model)
        self.anticipatory_cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.anticipatory_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        reactive: torch.Tensor,
        anticipatory: torch.Tensor,
        role_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run parallel cross-attention on reactive and anticipatory states.

        Args:
            reactive: ``[B, N_roles, T, D]`` forward BiLSTM hidden states.
            anticipatory: ``[B, N_roles, T, D]`` backward BiLSTM hidden states.
            role_mask: ``[B, N_roles]`` float mask (1 for present, 0 for padded).

        Returns:
            Tuple of ``(reactive_out, anticipatory_out)`` each ``[B, N, T, D]``.
        """
        bsz, n_roles, time_len, d_model = reactive.shape
        role_weights = role_mask.unsqueeze(-1).unsqueeze(-1)  # [B, N, 1, 1]

        # Mean-of-others for each participant: (sum_all - self) / (count - 1)
        reactive_sum = (reactive * role_weights).sum(dim=1, keepdim=True)
        antic_sum = (anticipatory * role_weights).sum(dim=1, keepdim=True)
        partner_count = role_mask.sum(dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1) - role_weights
        partner_count = partner_count.clamp(min=1.0)
        reactive_partner = (reactive_sum - reactive * role_weights) / partner_count
        antic_partner = (antic_sum - anticipatory * role_weights) / partner_count

        # Merge B*N into batch dim for MultiheadAttention
        r_query = reactive.reshape(bsz * n_roles, time_len, d_model)
        r_kv = reactive_partner.reshape(bsz * n_roles, time_len, d_model)
        a_query = anticipatory.reshape(bsz * n_roles, time_len, d_model)
        a_kv = antic_partner.reshape(bsz * n_roles, time_len, d_model)

        r_out, _ = self.reactive_cross_attn(r_query, r_kv, r_kv, need_weights=False)
        r_out = self.reactive_norm(r_query + r_out)

        a_out, _ = self.anticipatory_cross_attn(a_query, a_kv, a_kv, need_weights=False)
        a_out = self.anticipatory_norm(a_query + a_out)

        return (
            r_out.reshape(bsz, n_roles, time_len, d_model),
            a_out.reshape(bsz, n_roles, time_len, d_model),
        )


class DAPAGroupMultimodalRegressor(nn.Module):
    """Hybrid TCN + BiLSTM + Parallel Cross-Attention group engagement model.

    Inspired by DAPA (Yu et al., MM 2025).  Retains the existing TCN encoder
    for local temporal features, then adds a BiLSTM for full-sequence context
    and parallel cross-attention for inter-participant synchrony modeling.
    """

    def __init__(
        self,
        modality_dims: dict[str, int],
        fusion_channels: int = 64,
        fusion_mode: str = "gated",
        modality_dropout: float = 0.1,
        tcn_hidden_channels: int = 64,
        tcn_levels: int = 4,
        tcn_kernel_size: int = 5,
        tcn_dropout: float = 0.2,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.1,
        cross_attn_heads: int = 4,
        cross_attn_dropout: float = 0.1,
        n_domains: int = 2,
        n_prompt_tokens: int = 4,
        use_domain_prompts: bool = True,
        encoder_sharing: str = "shared",
        max_role_encoders: int = 8,
        prediction_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if encoder_sharing not in {"shared", "separate"}:
            raise ValueError(f"Unsupported encoder_sharing: {encoder_sharing}")

        self.use_domain_prompts = use_domain_prompts
        self.n_prompt_tokens = n_prompt_tokens if use_domain_prompts else 0
        self.encoder_sharing = encoder_sharing
        self.max_role_encoders = max_role_encoders
        self.lstm_hidden = lstm_hidden

        # Stage 1: Multimodal Fusion (reused from existing codebase)
        self.fusion = GroupRoleWiseMultimodalFusion(
            modality_dims=modality_dims,
            d_shared=fusion_channels,
            fusion_mode=fusion_mode,
            modality_dropout=modality_dropout,
        )
        fused_dim = self.fusion.fused_features_per_role

        # Stage 1.5: Domain Prompting
        if use_domain_prompts:
            self.domain_prompt = DomainPromptModule(
                n_domains=n_domains,
                prompt_dim=fused_dim,
                n_prompt_tokens=n_prompt_tokens,
            )
        else:
            self.domain_prompt = None

        # Stage 2: TCN Encoder
        def make_tcn_encoder() -> nn.Sequential:
            channels = [fused_dim] + [tcn_hidden_channels] * tcn_levels
            return nn.Sequential(*[
                TemporalBlock(
                    in_channels=channels[idx],
                    out_channels=channels[idx + 1],
                    kernel_size=tcn_kernel_size,
                    dilation=2 ** idx,
                    dropout=tcn_dropout,
                )
                for idx in range(tcn_levels)
            ])

        if encoder_sharing == "shared":
            self.tcn_encoder = make_tcn_encoder()
            self.role_tcn_encoders = None
        else:
            self.tcn_encoder = None
            self.role_tcn_encoders = nn.ModuleList(
                [make_tcn_encoder() for _ in range(max_role_encoders)]
            )

        # Stage 3: BiLSTM
        self.bilstm = nn.LSTM(
            input_size=tcn_hidden_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
        )

        # Stage 4: Parallel Cross-Attention
        self.cross_attention = ParallelCrossAttention(
            d_model=lstm_hidden,
            n_heads=cross_attn_heads,
            dropout=cross_attn_dropout,
        )

        # Stage 5: Prediction Head
        head_input_dim = lstm_hidden * 2  # reactive + anticipatory concatenated
        self.prediction_head = nn.Sequential(
            nn.Linear(head_input_dim, head_input_dim // 2),
            nn.ReLU(),
            nn.Dropout(prediction_dropout),
            nn.Linear(head_input_dim // 2, 1),
        )

    def forward(
        self,
        x_modalities: dict[str, torch.Tensor],
        role_mask: torch.Tensor | None = None,
        domain_ids: torch.Tensor | None = None,
        return_gate_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, object]]:
        # Stage 1: Multimodal fusion
        fused, fusion_info = self.fusion(x_modalities)
        bsz, n_roles, channels, time_len = fused.shape

        # Stage 1.5: Domain prompting
        if self.use_domain_prompts and self.domain_prompt is not None:
            if domain_ids is None:
                domain_ids = torch.zeros(bsz, dtype=torch.long, device=fused.device)
            fused = self.domain_prompt(fused, domain_ids)
            prompted_time = fused.shape[-1]
        else:
            prompted_time = time_len

        # Stage 2: TCN encoder
        if self.encoder_sharing == "shared":
            assert self.tcn_encoder is not None
            tcn_out = self.tcn_encoder(
                fused.reshape(bsz * n_roles, channels, prompted_time)
            )
        else:
            assert self.role_tcn_encoders is not None
            if n_roles > len(self.role_tcn_encoders):
                raise RuntimeError(
                    f"Need {n_roles} role TCN encoders, only {len(self.role_tcn_encoders)} configured."
                )
            tcn_parts = [self.role_tcn_encoders[i](fused[:, i]) for i in range(n_roles)]
            tcn_out = torch.stack(tcn_parts, dim=1).reshape(
                bsz * n_roles, tcn_parts[0].shape[1], prompted_time
            )

        # Strip prompt tokens so downstream sees original time length
        if self.n_prompt_tokens > 0:
            tcn_out = tcn_out[:, :, self.n_prompt_tokens:]

        # Stage 3: BiLSTM  [B*N, T, tcn_hidden] -> [B*N, T, 2*lstm_hidden]
        lstm_out, _ = self.bilstm(tcn_out.transpose(1, 2))

        reactive = lstm_out[:, :, :self.lstm_hidden]     # forward states
        anticipatory = lstm_out[:, :, self.lstm_hidden:]  # backward states

        reactive = reactive.reshape(bsz, n_roles, time_len, self.lstm_hidden)
        anticipatory = anticipatory.reshape(bsz, n_roles, time_len, self.lstm_hidden)

        # Stage 4: Parallel cross-attention
        if role_mask is None:
            role_mask = torch.ones(bsz, n_roles, device=reactive.device, dtype=reactive.dtype)
        role_mask_float = role_mask.to(device=reactive.device, dtype=reactive.dtype)

        reactive_out, antic_out = self.cross_attention(
            reactive, anticipatory, role_mask_float,
        )

        # Stage 5: Prediction
        combined = torch.cat([reactive_out, antic_out], dim=-1)  # [B, N, T, 2D]
        pred = self.prediction_head(combined).squeeze(-1)         # [B, N, T]
        pred = pred.permute(0, 2, 1)  # [B, T, N] to match existing output format
        pred = pred * role_mask_float.unsqueeze(1)

        if not return_gate_weights:
            return pred
        return pred, fusion_info
