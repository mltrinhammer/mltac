"""Train PinSoRo shared-person early-fusion models with optional logit interaction.

This keeps the existing early-fusion pipeline untouched. The model first scores
each role with a shared person encoder, then optionally applies a small
post-logit residual interaction module before the usual reconstruction/HMM path.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from src.acm_pipeline.models_tcn import TemporalBlock  # noqa: E402
from src.acm_pipeline.pinsoro_data import (  # noqa: E402
    PinSoRoWindow,
    PinSoRoWindowDataset,
    read_pinsoro_window_manifests,
)
from src.acm_pipeline.pinsoro_train_utils import (  # noqa: E402
    CLASS_COUNTS,
    HEADS,
    masked_multitask_cross_entropy,
    prediction_coverage_rows,
    write_csv,
    write_metric_outputs,
    write_pinsoro_submission_tree,
    write_prediction_scores,
    write_predictions,
    write_test_predictions,
)
from train_gated_fusion import (  # noqa: E402
    DEFAULT_FEATURES,
    RoleProjectedFusion,
    filter_domain,
    modality_dims,
    reconstruct,
)
from train_pinsoro_tcn import (  # noqa: E402
    compute_class_weights,
    compute_cr_social_weights,
    load_checkpoint,
    make_loader,
    resolve_device,
    save_checkpoint,
    serializable_args,
    set_seed,
)


class LogitInteractionResidual(nn.Module):
    """Residual interaction module over per-role logits.

    Modes:
    - linear: same-window residual using self+partner logits for the same head.
    - gated: same-window residual with a learned gate over the residual strength.
    - tcn: causal temporal residual over self+partner logits for the same head.
    - cross_head_linear: same-window residual using task+social logits jointly.
    """

    def __init__(
        self,
        mode: str,
        hidden_channels: int,
        kernel_size: int,
        dropout: float,
        scale: float,
    ) -> None:
        super().__init__()
        if mode not in {"none", "linear", "gated", "tcn", "cross_head_linear", "attention"}:
            raise ValueError(f"Unsupported interaction mode: {mode}")
        self.mode = mode
        self.scale = float(scale)
        modules: dict[str, nn.Module] = {}
        gates: dict[str, nn.Module] = {}
        attentions: dict[str, nn.Module] = {}
        if mode in {"linear", "gated"}:
            modules = {
                head: nn.Conv1d(2 * n_classes, 2 * n_classes, kernel_size=1)
                for head, n_classes in CLASS_COUNTS.items()
            }
            if mode == "gated":
                gates = {
                    head: nn.Conv1d(2 * n_classes, 2 * n_classes, kernel_size=1)
                    for head, n_classes in CLASS_COUNTS.items()
                }
        elif mode == "tcn":
            modules = {
                head: nn.Sequential(
                    TemporalBlock(2 * n_classes, hidden_channels, kernel_size, 1, dropout, causal=True),
                    TemporalBlock(hidden_channels, hidden_channels, kernel_size, 2, dropout, causal=True),
                    nn.Conv1d(hidden_channels, 2 * n_classes, kernel_size=1),
                )
                for head, n_classes in CLASS_COUNTS.items()
            }
        elif mode == "cross_head_linear":
            total_channels = 2 * sum(CLASS_COUNTS.values())
            modules = {
                head: nn.Conv1d(total_channels, 2 * n_classes, kernel_size=1)
                for head, n_classes in CLASS_COUNTS.items()
            }
        elif mode == "attention":
            attentions = {
                head: nn.MultiheadAttention(
                    embed_dim=n_classes,
                    num_heads=1,
                    dropout=dropout,
                    batch_first=True,
                )
                for head, n_classes in CLASS_COUNTS.items()
            }
        self.modules_by_head = nn.ModuleDict(modules)
        self.gates_by_head = nn.ModuleDict(gates)
        self.attentions_by_head = nn.ModuleDict(attentions)

    def _same_head_input(self, value: torch.Tensor) -> torch.Tensor:
        batch, roles, _time, n_classes = value.shape
        return value.permute(0, 1, 3, 2).reshape(batch, roles * n_classes, value.shape[2])

    def _cross_head_input(self, logits: dict[str, torch.Tensor]) -> torch.Tensor:
        parts = []
        for head in CLASS_COUNTS:
            if head not in logits:
                raise KeyError(f"Missing logits for cross-head interaction: {head}")
            parts.append(self._same_head_input(logits[head]))
        return torch.cat(parts, dim=1)

    def forward(self, logits: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.mode == "none":
            return logits
        if self.mode == "attention":
            corrected = {}
            for head, value in logits.items():
                if value.shape[1] != 2:
                    raise RuntimeError(f"Attention interaction expects 2 roles, got {value.shape[1]}.")
                attention = self.attentions_by_head[head]
                role0, _ = attention(value[:, 0], value[:, 1], value[:, 1], need_weights=False)
                role1, _ = attention(value[:, 1], value[:, 0], value[:, 0], need_weights=False)
                residual = torch.stack([role0, role1], dim=1)
                corrected[head] = value + self.scale * residual
            return corrected
        corrected = {}
        cross_head_input = self._cross_head_input(logits) if self.mode == "cross_head_linear" else None
        for head, value in logits.items():
            batch, roles, time, n_classes = value.shape
            interaction_input = cross_head_input if cross_head_input is not None else self._same_head_input(value)
            residual = self.modules_by_head[head](interaction_input)
            if self.mode == "gated":
                gate = torch.sigmoid(self.gates_by_head[head](interaction_input))
                residual = gate * residual
            residual = residual.reshape(batch, roles, n_classes, time).permute(0, 1, 3, 2)
            corrected[head] = value + self.scale * residual
        return corrected


ROLE_METADATA_VECTORS = {"purple": (1.0, 0.0), "yellow": (0.0, 1.0)}


def read_participant_metadata(path: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    table: dict[tuple[str, str, str], dict[str, str]] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            table[(row["domain"], row["session_id"], row["role"])] = {
                "age": row.get("age", "").strip(),
                "gender": row.get("gender", "").strip(),
            }
    return table


def participant_age_stats(windows: list, table: dict[tuple[str, str, str], dict[str, str]]) -> tuple[float, float]:
    seen: set[tuple[str, str, str]] = set()
    ages: list[float] = []
    for window in windows:
        for role in window.roles:
            key = (window.domain, window.session_id, role)
            if key in seen:
                continue
            seen.add(key)
            try:
                ages.append(float(table.get(key, {}).get("age", "")))
            except ValueError:
                pass
    if not ages:
        return 0.0, 1.0
    mean = float(np.mean(ages))
    std = float(np.std(ages))
    return mean, std if std > 1e-6 else 1.0


def metadata_dim_for_mode(mode: str) -> int:
    if mode == "none":
        return 0
    if mode == "role":
        return 2
    if mode == "age_gender":
        return 5
    if mode == "age_gender_role":
        return 7
    raise ValueError(f"Unsupported PinSoRo metadata mode: {mode}")


def encode_participant_metadata(
    row: dict[str, str],
    role: str,
    mode: str,
    age_mean: float,
    age_std: float,
) -> np.ndarray:
    values: list[float] = []
    if mode in {"age_gender", "age_gender_role"}:
        try:
            age_z = (float(row.get("age", "")) - age_mean) / age_std
            age_known = 1.0
        except ValueError:
            age_z = 0.0
            age_known = 0.0
        gender = row.get("gender", "")
        values.extend(
            [
                float(age_z),
                age_known,
                1.0 if gender == "1" else 0.0,
                1.0 if gender == "2" else 0.0,
                1.0 if gender not in {"1", "2"} else 0.0,
            ]
        )
    if mode in {"role", "age_gender_role"}:
        values.extend(ROLE_METADATA_VECTORS.get(role, (0.0, 0.0)))
    return np.asarray(values, dtype=np.float32)


class RoleMetadataPinSoRoWindowDataset(PinSoRoWindowDataset):
    def __init__(
        self,
        *args,
        metadata_mode: str = "role",
        metadata_table: dict[tuple[str, str, str], dict[str, str]] | None = None,
        age_mean: float = 0.0,
        age_std: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if metadata_mode not in {"role", "age_gender", "age_gender_role"}:
            raise ValueError(f"Unsupported PinSoRo metadata mode: {metadata_mode}")
        self.metadata_mode = metadata_mode
        self.metadata_table = metadata_table or {}
        self.age_mean = float(age_mean)
        self.age_std = float(age_std) if abs(float(age_std)) > 1e-6 else 1.0

    def __getitem__(self, idx: int) -> dict[str, object]:
        item = super().__getitem__(idx)
        window = self.windows[idx]
        item["metadata"] = np.asarray(
            [
                encode_participant_metadata(
                    self.metadata_table.get((window.domain, window.session_id, role), {}),
                    role,
                    self.metadata_mode,
                    self.age_mean,
                    self.age_std,
                )
                for role in window.roles
            ],
            dtype=np.float32,
        )
        return item


class SharedPersonFusionInteractionTCN(nn.Module):
    """Shared per-person early-fusion TCN plus optional post-logit interaction."""

    def __init__(
        self,
        modality_dims: dict[str, int],
        fusion_mode: str,
        fusion_channels: int,
        person_hidden_channels: int,
        person_levels: int,
        person_kernel_size: int,
        dropout: float,
        modality_dropout: float,
        causal_tcn: bool,
        encoder_sharing: str,
        interaction_mode: str,
        interaction_hidden_channels: int,
        interaction_kernel_size: int,
        interaction_scale: float,
        head_architecture: str = "shared_tcn",
        head_adapter_levels: int = 1,
        metadata_dim: int = 0,
        metadata_embedding_dim: int = 16,
        metadata_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if encoder_sharing not in {"shared", "separate", "dyadic_shared"}:
            raise ValueError(f"Unsupported encoder_sharing: {encoder_sharing}")
        if head_architecture not in {"shared_tcn", "head_adapters", "separate_tcn"}:
            raise ValueError(f"Unsupported head_architecture: {head_architecture}")
        if head_adapter_levels < 1:
            raise ValueError("head_adapter_levels must be at least 1.")
        self.encoder_sharing = encoder_sharing
        self.head_architecture = head_architecture
        self.metadata_dim = int(metadata_dim)
        self.metadata_embedding_dim = int(metadata_embedding_dim) if self.metadata_dim > 0 else 0
        self.fusion = RoleProjectedFusion(modality_dims, fusion_channels, modality_dropout, fusion_mode)

        def make_encoder() -> nn.Sequential:
            channels = [self.fusion.fused_channels_per_role] + [person_hidden_channels] * person_levels
            return nn.Sequential(
                *[
                    TemporalBlock(channels[idx], channels[idx + 1], person_kernel_size, 2**idx, dropout, causal=causal_tcn)
                    for idx in range(person_levels)
                ]
            )

        self.person_encoder: nn.Sequential | None = None
        self.role_encoders: nn.ModuleList | None = None
        self.dyadic_encoder: nn.Sequential | None = None
        self.head_encoders: nn.ModuleDict | None = None
        self.head_role_encoders: nn.ModuleDict | None = None
        self.head_adapters: nn.ModuleDict | None = None
        if head_architecture in {"shared_tcn", "head_adapters"}:
            if encoder_sharing == "shared":
                self.person_encoder = make_encoder()
            elif encoder_sharing == "separate":
                self.role_encoders = nn.ModuleList([make_encoder(), make_encoder()])
            else:
                dyadic_channels = [2 * self.fusion.fused_channels_per_role] + [person_hidden_channels] * person_levels
                self.dyadic_encoder = nn.Sequential(
                    *[
                        TemporalBlock(
                            dyadic_channels[idx],
                            dyadic_channels[idx + 1],
                            person_kernel_size,
                            2**idx,
                            dropout,
                            causal=causal_tcn,
                        )
                        for idx in range(person_levels)
                    ]
                )
            if head_architecture == "head_adapters":
                def make_adapter() -> nn.Sequential:
                    return nn.Sequential(
                        *[
                            TemporalBlock(
                                person_hidden_channels,
                                person_hidden_channels,
                                person_kernel_size,
                                1,
                                dropout,
                                causal=causal_tcn,
                            )
                            for _ in range(head_adapter_levels)
                        ]
                    )

                self.head_adapters = nn.ModuleDict({head: make_adapter() for head in CLASS_COUNTS})
        else:
            if encoder_sharing == "dyadic_shared":
                raise ValueError("encoder_sharing=dyadic_shared is supported for shared_tcn/head_adapters only.")
            if encoder_sharing == "shared":
                self.head_encoders = nn.ModuleDict({head: make_encoder() for head in CLASS_COUNTS})
            else:
                self.head_role_encoders = nn.ModuleDict(
                    {head: nn.ModuleList([make_encoder(), make_encoder()]) for head in CLASS_COUNTS}
                )
        if self.metadata_dim > 0:
            metadata_input_dim = 2 * self.metadata_dim if encoder_sharing == "dyadic_shared" else self.metadata_dim
            self.metadata_encoder = nn.Sequential(
                nn.Linear(metadata_input_dim, self.metadata_embedding_dim),
                nn.ReLU(),
                nn.Dropout(metadata_dropout),
            )
        else:
            self.metadata_encoder = None
        head_channels = person_hidden_channels + self.metadata_embedding_dim
        self.heads = nn.ModuleDict(
            {
                head: nn.Conv1d(
                    head_channels,
                    2 * n_classes if encoder_sharing == "dyadic_shared" else n_classes,
                    kernel_size=1,
                )
                for head, n_classes in CLASS_COUNTS.items()
            }
        )
        self.interaction = LogitInteractionResidual(
            interaction_mode,
            interaction_hidden_channels,
            interaction_kernel_size,
            dropout,
            interaction_scale,
        )

    def _encode_roles(
        self,
        fused: torch.Tensor,
        shared_encoder: nn.Sequential | None,
        role_encoders: nn.ModuleList | None,
    ) -> torch.Tensor:
        batch, roles, channels, time = fused.shape
        if shared_encoder is not None:
            hidden = shared_encoder(fused.reshape(batch * roles, channels, time))
            return hidden.reshape(batch, roles, hidden.shape[1], time)
        if role_encoders is None:
            raise RuntimeError("Role encoders are required for separate role encoding.")
        if roles != len(role_encoders):
            raise RuntimeError(f"Expected {len(role_encoders)} roles, got {roles}.")
        return torch.stack([role_encoders[idx](fused[:, idx]) for idx in range(roles)], dim=1)

    def _encode_dyad(self, fused: torch.Tensor) -> torch.Tensor:
        if self.dyadic_encoder is None:
            raise RuntimeError("Dyadic encoder is not configured.")
        batch, roles, channels, time = fused.shape
        if roles != 2:
            raise RuntimeError(f"Dyadic shared encoding expects 2 roles, got {roles}.")
        return self.dyadic_encoder(fused.reshape(batch, roles * channels, time))

    def _append_metadata(
        self,
        hidden: torch.Tensor,
        metadata: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.metadata_encoder is None:
            return hidden
        batch, roles, _channels, time = hidden.shape
        if metadata is None:
            raise ValueError("metadata tensor is required when metadata_dim > 0.")
        if metadata.shape[:2] != (batch, roles) or metadata.shape[2] != self.metadata_dim:
            raise ValueError(f"Expected metadata [B,R,{self.metadata_dim}], got {tuple(metadata.shape)}")
        meta = self.metadata_encoder(metadata.to(device=hidden.device, dtype=hidden.dtype))
        meta = meta.unsqueeze(-1).expand(-1, -1, -1, time)
        return torch.cat([hidden, meta], dim=2)

    def _append_dyadic_metadata(
        self,
        hidden: torch.Tensor,
        metadata: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.metadata_encoder is None:
            return hidden
        batch, _channels, time = hidden.shape
        if metadata is None:
            raise ValueError("metadata tensor is required when metadata_dim > 0.")
        if metadata.shape[:2] != (batch, 2) or metadata.shape[2] != self.metadata_dim:
            raise ValueError(f"Expected metadata [B,2,{self.metadata_dim}], got {tuple(metadata.shape)}")
        meta = metadata.to(device=hidden.device, dtype=hidden.dtype).reshape(batch, 2 * self.metadata_dim)
        meta = self.metadata_encoder(meta).unsqueeze(-1).expand(-1, -1, time)
        return torch.cat([hidden, meta], dim=1)

    def forward(
        self,
        x: torch.Tensor,
        domain_ids: torch.Tensor | None = None,
        metadata: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        _ = domain_ids
        fused, _info = self.fusion(x)
        batch, roles, _channels, time = fused.shape
        logits = {}
        if self.head_architecture in {"shared_tcn", "head_adapters"}:
            if self.encoder_sharing == "dyadic_shared":
                base_hidden = self._encode_dyad(fused)
                for head, layer in self.heads.items():
                    hidden = base_hidden
                    if self.head_adapters is not None:
                        hidden = self.head_adapters[head](hidden)
                    hidden = self._append_dyadic_metadata(hidden, metadata)
                    value = layer(hidden).reshape(batch, roles, CLASS_COUNTS[head], time)
                    logits[head] = value.permute(0, 1, 3, 2)
            else:
                base_hidden = self._encode_roles(fused, self.person_encoder, self.role_encoders)
                for head, layer in self.heads.items():
                    hidden = base_hidden
                    if self.head_adapters is not None:
                        adapter = self.head_adapters[head]
                        hidden = adapter(hidden.reshape(batch * roles, hidden.shape[2], time))
                        hidden = hidden.reshape(batch, roles, hidden.shape[1], time)
                    hidden = self._append_metadata(hidden, metadata)
                    value = layer(hidden.reshape(batch * roles, hidden.shape[2], time)).transpose(1, 2)
                    logits[head] = value.reshape(batch, roles, time, CLASS_COUNTS[head])
        else:
            assert self.head_architecture == "separate_tcn"
            for head, layer in self.heads.items():
                if self.head_encoders is not None:
                    hidden = self._encode_roles(fused, self.head_encoders[head], None)
                else:
                    assert self.head_role_encoders is not None
                    hidden = self._encode_roles(fused, None, self.head_role_encoders[head])
                hidden = self._append_metadata(hidden, metadata)
                value = layer(hidden.reshape(batch * roles, hidden.shape[2], time)).transpose(1, 2)
                logits[head] = value.reshape(batch, roles, time, CLASS_COUNTS[head])
        return self.interaction(logits)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PinSoRo person-interaction early-fusion model.")
    parser.add_argument("--manifest", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "MoE" / "experiments" / "pinsoro_person_interaction_early_fusion",
    )
    parser.add_argument("--run-name", default="")
    parser.add_argument("--domain-scope", choices=("both", "CC", "CR"), default="both")
    parser.add_argument("--train-split", default="train_internal")
    parser.add_argument("--val-split", default="val_internal")
    parser.add_argument("--test-split", default="test_internal")
    parser.add_argument("--fusion-mode", choices=("gated", "concat"), default="concat")
    parser.add_argument("--fusion-channels", type=int, default=64)
    parser.add_argument("--person-hidden-channels", type=int, default=64)
    parser.add_argument("--person-levels", type=int, default=5)
    parser.add_argument("--person-kernel-size", type=int, default=11)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--modality-dropout", type=float, default=0.1)
    parser.add_argument("--causal-tcn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--encoder-sharing", choices=("shared", "separate", "dyadic_shared"), default="shared")
    parser.add_argument("--head-architecture", choices=("shared_tcn", "head_adapters", "separate_tcn"), default="shared_tcn")
    parser.add_argument("--head-adapter-levels", type=int, default=1)
    parser.add_argument("--interaction-mode", choices=("none", "linear", "gated", "tcn", "cross_head_linear", "attention"), default="none")
    parser.add_argument("--interaction-hidden-channels", type=int, default=32)
    parser.add_argument("--interaction-kernel-size", type=int, default=5)
    parser.add_argument("--interaction-scale", type=float, default=0.1)
    parser.add_argument("--metadata", type=Path, help="Optional PinSoRo participant metadata CSV.")
    parser.add_argument("--metadata-mode", choices=("none", "role", "age_gender", "age_gender_role"), default="none")
    parser.add_argument("--metadata-embedding-dim", type=int, default=16)
    parser.add_argument("--metadata-dropout", type=float, default=0.2)
    parser.add_argument("--cr-social-weighting", choices=("shared_inverse", "unweighted", "sqrt_inverse", "capped_inverse", "targeted"), default="shared_inverse")
    parser.add_argument("--cr-social-weight-cap", type=float, default=5.0)
    parser.add_argument("--cr-social-target-class2-weight", type=float, default=2.0)
    parser.add_argument("--cr-social-target-class3-weight", type=float, default=0.5)
    parser.add_argument("--cr-social-class3-oversample", type=int, default=1)
    parser.add_argument(
        "--cr-social-class3-aux-weight",
        type=float,
        default=0.0,
        help="Optional binary class-3-vs-rest auxiliary loss on CR social logits.",
    )
    parser.add_argument("--cc-task-weighting", choices=("shared_inverse", "targeted"), default="shared_inverse")
    parser.add_argument("--cc-task-target-class0-weight", type=float, default=1.0)
    parser.add_argument("--cc-task-target-class1-weight", type=float, default=1.0)
    parser.add_argument("--cc-task-target-class2-weight", type=float, default=1.0)
    parser.add_argument("--cc-task-target-class3-weight", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--min-epochs", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=0.001)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--temporal-delta-weight",
        type=float,
        default=0.0,
        help="Weight for MSE between probability deltas and one-hot label deltas.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-cached-tensors", type=int, default=6)
    parser.add_argument("--mmap-cache-root", type=Path)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Load output-root/run-name/model_best.pt and export validation/test predictions without training.",
    )
    parser.add_argument("--soft-label-mode", choices=("none", "soft_uniform", "soft_confidence"), default="none")
    parser.add_argument(
        "--active-heads",
        nargs="+",
        choices=("task", "social"),
        default=["task", "social"],
        help="Train and early-stop on only these heads; useful for task/social specialists.",
    )
    return parser.parse_args()


def masked_probability_delta_loss(
    logits: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    active_heads: tuple[str, ...],
) -> torch.Tensor:
    losses = []
    for head in active_heads:
        target = batch[f"{head}_y"]
        mask = batch[f"{head}_mask"].bool()
        adjacent_mask = mask[..., 1:] & mask[..., :-1]
        if not torch.any(adjacent_mask):
            continue
        probs = torch.softmax(logits[head], dim=-1)
        target_one_hot = nn.functional.one_hot(target.clamp_min(0), CLASS_COUNTS[head]).to(
            dtype=probs.dtype,
            device=probs.device,
        )
        pred_delta = probs[..., 1:, :] - probs[..., :-1, :]
        target_delta = target_one_hot[..., 1:, :] - target_one_hot[..., :-1, :]
        squared_error = (pred_delta - target_delta).pow(2).sum(dim=-1)
        losses.append(squared_error[adjacent_mask].mean())
    if not losses:
        return torch.zeros((), device=next(iter(logits.values())).device)
    return torch.stack(losses).mean()


def cr_social_class3_auxiliary_loss(
    logits: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Binary associative-vs-rest loss from the social multiclass logits."""

    if "social" not in logits:
        return torch.zeros((), device=next(iter(logits.values())).device)
    mask = batch["social_mask"].bool() & (batch["domain_id"].reshape(-1, 1, 1) == 1)
    if not torch.any(mask):
        return torch.zeros((), device=logits["social"].device)
    social_logits = logits["social"][mask]
    target = (batch["social_y"][mask] == 3).to(dtype=social_logits.dtype)
    other_logits = torch.cat([social_logits[:, :3], social_logits[:, 4:]], dim=1)
    binary_logit = social_logits[:, 3] - torch.logsumexp(other_logits, dim=1)
    positive = target.sum()
    negative = target.numel() - positive
    pos_weight = (negative / positive.clamp_min(1.0)).clamp(max=50.0)
    return nn.functional.binary_cross_entropy_with_logits(
        binary_logit,
        target,
        pos_weight=pos_weight,
    )


def train_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    class_weights: dict[str, torch.Tensor],
    soft_label_mode: str,
    active_heads: tuple[str, ...],
    temporal_delta_weight: float,
    cr_social_class3_aux_weight: float,
) -> float:
    model.train()
    total = torch.zeros((), device=device)
    n_batches = 0
    non_blocking = device.type == "cuda"
    for batch in loader:
        if not batch["has_supervision"]:
            continue
        batch = {
            key: value.to(device, non_blocking=non_blocking)
            for key, value in batch.items()
            if key not in ("window_indices", "has_supervision")
        }
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch["x"], batch["domain_id"], metadata=batch.get("metadata"))
        loss = masked_multitask_cross_entropy(
            logits,
            batch,
            class_weights,
            soft_label_mode=soft_label_mode,
            active_heads=active_heads,
        )
        if temporal_delta_weight > 0.0:
            loss = loss + temporal_delta_weight * masked_probability_delta_loss(
                logits,
                batch,
                active_heads,
            )
        if cr_social_class3_aux_weight > 0.0 and "social" in active_heads:
            loss = loss + cr_social_class3_aux_weight * cr_social_class3_auxiliary_loss(
                logits,
                batch,
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total += loss.detach()
        n_batches += 1
    return float((total / n_batches).item()) if n_batches else float("nan")


def active_head_organizer_score(run_dir: Path, active_heads: tuple[str, ...]) -> float:
    import csv

    path = run_dir / "metrics_by_domain.csv"
    if not path.exists():
        return float("nan")
    kappas = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("head") not in active_heads:
                continue
            try:
                value = float(row.get("kappa", "nan"))
            except ValueError:
                continue
            if np.isfinite(value):
                kappas.append(value)
    return float(np.mean(kappas)) if kappas else float("nan")


def window_has_cr_social_class3(window: PinSoRoWindow, dataset: PinSoRoWindowDataset) -> bool:
    if window.domain != "CR":
        return False
    for role_idx, supervised in enumerate(window.supervised):
        if not supervised:
            continue
        data = dataset.load_full_role(window, role_idx)
        labels = np.asarray(data["social_y"], dtype=np.int64)[window.start_frame : window.end_frame]
        mask = np.asarray(data["social_mask"])[window.start_frame : window.end_frame].astype(bool)
        if np.any(mask & (labels == 3)):
            return True
    return False

def oversample_cr_social_class3_windows(
    windows: list[PinSoRoWindow],
    args: argparse.Namespace,
    dataset_cls: type[PinSoRoWindowDataset],
    metadata_table: dict[tuple[str, str, str], dict[str, str]],
    age_mean: float,
    age_std: float,
) -> list[PinSoRoWindow]:
    multiplier = max(1, int(args.cr_social_class3_oversample))
    if multiplier <= 1:
        return windows
    if args.metadata_mode != "none":
        probe_dataset = dataset_cls(
            windows,
            args.max_cached_tensors,
            args.mmap_cache_root,
            PROJECT_ROOT,
            metadata_mode=args.metadata_mode,
            metadata_table=metadata_table,
            age_mean=age_mean,
            age_std=age_std,
        )
    else:
        probe_dataset = dataset_cls(windows, args.max_cached_tensors, args.mmap_cache_root, PROJECT_ROOT)
    class3_windows = [window for window in windows if window_has_cr_social_class3(window, probe_dataset)]
    if not class3_windows:
        print("CR social class-3 oversampling requested, but no matching windows found.", flush=True)
        return windows
    expanded = list(windows) + class3_windows * (multiplier - 1)
    print(
        f"CR social class-3 oversampling: base_windows={len(windows)} "
        f"class3_windows={len(class3_windows)} multiplier={multiplier} expanded_windows={len(expanded)}",
        flush=True,
    )
    return expanded


def count_head_labels(
    windows: list[PinSoRoWindow],
    dataset: PinSoRoWindowDataset,
    domain: str,
    head: str,
) -> np.ndarray:
    counts = np.zeros(CLASS_COUNTS[head], dtype=np.int64)
    seen: set[Path] = set()
    for window in windows:
        if window.domain != domain:
            continue
        for role_idx, paths in enumerate(window.tensor_paths):
            label_path = paths[0]
            if not window.supervised[role_idx] or label_path in seen:
                continue
            seen.add(label_path)
            data = dataset.load_full_role(window, role_idx)
            labels = np.asarray(data[f"{head}_y"], dtype=np.int64)
            mask = np.asarray(data[f"{head}_mask"]).astype(bool)
            counts += np.bincount(labels[mask], minlength=CLASS_COUNTS[head])
    return counts


def compute_cc_task_weights(
    base_weights: torch.Tensor,
    windows: list[PinSoRoWindow],
    dataset: PinSoRoWindowDataset,
    args: argparse.Namespace,
) -> torch.Tensor | None:
    if args.cc_task_weighting == "shared_inverse":
        return None
    if args.cc_task_weighting != "targeted":
        raise ValueError(f"Unknown CC-task weighting mode: {args.cc_task_weighting}")
    counts = count_head_labels(windows, dataset, "CC", "task")
    present = counts > 0
    multipliers = np.asarray(
        [
            args.cc_task_target_class0_weight,
            args.cc_task_target_class1_weight,
            args.cc_task_target_class2_weight,
            args.cc_task_target_class3_weight,
        ],
        dtype=np.float32,
    )
    weights = base_weights.detach().cpu().numpy().astype(np.float32, copy=True)
    weights[present] *= multipliers[present]
    weights[~present] = 0.0
    total_weight = float(np.dot(counts, weights))
    if total_weight <= 0.0:
        raise RuntimeError("No supervised CC-task labels found for CC-task targeted weighting.")
    weights *= counts.sum() / total_weight
    return torch.from_numpy(weights)


def main() -> None:
    args = parse_args()
    active_heads = tuple(args.active_heads)
    if args.mmap_cache_root is not None and not args.mmap_cache_root.is_absolute():
        args.mmap_cache_root = PROJECT_ROOT / args.mmap_cache_root
    set_seed(args.seed)
    device = resolve_device(args.device)
    dims = modality_dims(args.manifest, args.train_split)

    train_windows = filter_domain(read_pinsoro_window_manifests(args.manifest, PROJECT_ROOT, args.train_split), args.domain_scope)
    val_windows = filter_domain(read_pinsoro_window_manifests(args.manifest, PROJECT_ROOT, args.val_split), args.domain_scope)
    if not train_windows or not val_windows:
        raise RuntimeError(f"Missing train/val windows for domain_scope={args.domain_scope}")
    if {len(window.roles) for window in train_windows + val_windows} != {2}:
        raise RuntimeError("Person-interaction fusion requires two-role dyadic manifests.")

    if args.metadata is None and args.metadata_mode != "none":
        default_metadata = PROJECT_ROOT / "MoE" / "moe_data" / "outputs" / "participant_metadata.csv"
        if default_metadata.is_file():
            args.metadata = default_metadata
            if args.metadata_mode == "role":
                args.metadata_mode = "age_gender_role"
    metadata_table = read_participant_metadata(args.metadata) if args.metadata is not None else {}
    age_mean, age_std = participant_age_stats(train_windows, metadata_table)
    dataset_cls = RoleMetadataPinSoRoWindowDataset if args.metadata_mode != "none" else PinSoRoWindowDataset
    train_windows = oversample_cr_social_class3_windows(
        train_windows,
        args,
        dataset_cls,
        metadata_table,
        age_mean,
        age_std,
    )

    if args.metadata_mode != "none":
        train_dataset = dataset_cls(
            train_windows,
            args.max_cached_tensors,
            args.mmap_cache_root,
            PROJECT_ROOT,
            metadata_mode=args.metadata_mode,
            metadata_table=metadata_table,
            age_mean=age_mean,
            age_std=age_std,
        )
        val_dataset = dataset_cls(
            val_windows,
            args.max_cached_tensors,
            args.mmap_cache_root,
            PROJECT_ROOT,
            metadata_mode=args.metadata_mode,
            metadata_table=metadata_table,
            age_mean=age_mean,
            age_std=age_std,
        )
    else:
        train_dataset = dataset_cls(train_windows, args.max_cached_tensors, args.mmap_cache_root, PROJECT_ROOT)
        val_dataset = dataset_cls(val_windows, args.max_cached_tensors, args.mmap_cache_root, PROJECT_ROOT)
    pin_memory = device.type == "cuda"
    train_loader = make_loader(train_dataset, args, shuffle=True, pin_memory=pin_memory)
    val_loader = make_loader(val_dataset, args, shuffle=False, pin_memory=pin_memory)

    class_weights = {head: value.to(device) for head, value in compute_class_weights(train_windows, train_dataset).items()}
    cc_task_weights = compute_cc_task_weights(
        class_weights["task"],
        train_windows,
        train_dataset,
        args,
    )
    if cc_task_weights is not None:
        class_weights["task"] = cc_task_weights.to(device)
    cr_social_weights = compute_cr_social_weights(
        train_windows,
        train_dataset,
        args.cr_social_weighting,
        args.cr_social_weight_cap,
        args.cr_social_target_class2_weight,
        args.cr_social_target_class3_weight,
    )
    if cr_social_weights is not None:
        class_weights["cr_social"] = cr_social_weights.to(device)

    model = SharedPersonFusionInteractionTCN(
        modality_dims=dims,
        fusion_mode=args.fusion_mode,
        fusion_channels=args.fusion_channels,
        person_hidden_channels=args.person_hidden_channels,
        person_levels=args.person_levels,
        person_kernel_size=args.person_kernel_size,
        dropout=args.dropout,
        modality_dropout=args.modality_dropout,
        causal_tcn=args.causal_tcn,
        encoder_sharing=args.encoder_sharing,
        interaction_mode=args.interaction_mode,
        interaction_hidden_channels=args.interaction_hidden_channels,
        interaction_kernel_size=args.interaction_kernel_size,
        interaction_scale=args.interaction_scale,
        head_architecture=args.head_architecture,
        head_adapter_levels=args.head_adapter_levels,
        metadata_dim=metadata_dim_for_mode(args.metadata_mode),
        metadata_embedding_dim=args.metadata_embedding_dim,
        metadata_dropout=args.metadata_dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    feature_name = "__".join(dims)
    run_name = args.run_name or (
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_pinsoro_{args.domain_scope.lower()}_"
        f"{feature_name}_{args.fusion_mode}_person_{args.interaction_mode}_seed{args.seed}"
    )
    run_dir = args.output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    config = serializable_args(args) | {
        "architecture": "projected_fusion_role_encoder_post_logit_interaction_tcn",
        "head_architecture": args.head_architecture,
        "modality_dims": dims,
        "n_train_windows": len(train_dataset),
        "n_val_windows": len(val_dataset),
        "class_weights": {head: value.detach().cpu().tolist() for head, value in class_weights.items()},
        "active_heads": list(active_heads),
        "metadata_dim": metadata_dim_for_mode(args.metadata_mode),
        "metadata_age_mean": age_mean,
        "metadata_age_std": age_std,
        "cr_social_class3_aux_weight": args.cr_social_class3_aux_weight,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    best_score = -float("inf")
    best_epoch = 0
    stale_epochs = 0
    log_rows: list[dict[str, object]] = []
    start_epoch = 1
    last_checkpoint_path = run_dir / "model_last.pt"
    if args.resume and last_checkpoint_path.exists():
        checkpoint = load_checkpoint(last_checkpoint_path, device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_score = float(checkpoint["best_val_organizer_score"])
        best_epoch = int(checkpoint["best_epoch"])
        stale_epochs = int(checkpoint["stale_epochs"])
        log_rows = list(checkpoint["log_rows"])
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if device.type == "cuda" and "cuda_rng_state_all" in checkpoint:
            torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng_state_all"]])

    if args.eval_only and not (run_dir / "model_best.pt").exists():
        raise FileNotFoundError(f"--eval-only requires an existing checkpoint: {run_dir / 'model_best.pt'}")

    for epoch in range(start_epoch, args.epochs + 1):
        if args.eval_only:
            break
        started = time.perf_counter()
        train_started = time.perf_counter()
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            class_weights,
            args.soft_label_mode,
            active_heads,
            args.temporal_delta_weight,
            args.cr_social_class3_aux_weight,
        )
        train_seconds = time.perf_counter() - train_started
        val_started = time.perf_counter()
        reconstructed = reconstruct(model, val_dataset, val_loader, device)
        val_metrics = write_metric_outputs(run_dir, reconstructed)
        val_seconds = time.perf_counter() - val_started
        score = (
            val_metrics["organizer_score"]
            if set(active_heads) == {"task", "social"}
            else active_head_organizer_score(run_dir, active_heads)
        )
        improved = np.isfinite(score) and score > best_score + args.min_delta
        if improved:
            best_score = score
            best_epoch = epoch
            stale_epochs = 0
            save_checkpoint(
                run_dir / "model_best.pt",
                {"epoch": epoch, "model_state_dict": model.state_dict(), "val_organizer_score": score},
            )
        else:
            stale_epochs += 1
        log_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_organizer_score": score,
                "best_epoch": best_epoch,
                "best_val_organizer_score": best_score,
                "stale_epochs": stale_epochs,
                "train_seconds": train_seconds,
                "val_seconds": val_seconds,
                "epoch_seconds": time.perf_counter() - started,
            }
        )
        write_csv(run_dir / "training_log.csv", log_rows)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_epoch": best_epoch,
            "best_val_organizer_score": best_score,
            "stale_epochs": stale_epochs,
            "log_rows": log_rows,
            "torch_rng_state": torch.get_rng_state(),
        }
        if device.type == "cuda":
            checkpoint["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
        save_checkpoint(last_checkpoint_path, checkpoint)
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.5f} "
            f"val_organizer_score={score:.5f} best_epoch={best_epoch}",
            flush=True,
        )
        if args.patience > 0 and epoch >= args.min_epochs and stale_epochs >= args.patience:
            break

    if (run_dir / "model_best.pt").exists():
        model.load_state_dict(load_checkpoint(run_dir / "model_best.pt", device)["model_state_dict"])
        reconstructed = reconstruct(model, val_dataset, val_loader, device)
        write_metric_outputs(run_dir, reconstructed)
        coverage_rows = prediction_coverage_rows(reconstructed, "validation")
        write_predictions(run_dir / "val_predictions.csv", reconstructed)
        write_prediction_scores(run_dir / "val_prediction_scores.csv.gz", reconstructed)
        test_windows = filter_domain(read_pinsoro_window_manifests(args.manifest, PROJECT_ROOT, args.test_split), args.domain_scope)
        if test_windows:
            if args.metadata_mode != "none":
                test_dataset = RoleMetadataPinSoRoWindowDataset(
                    test_windows,
                    args.max_cached_tensors,
                    args.mmap_cache_root,
                    PROJECT_ROOT,
                    metadata_mode=args.metadata_mode,
                    metadata_table=metadata_table,
                    age_mean=age_mean,
                    age_std=age_std,
                )
            else:
                test_dataset = PinSoRoWindowDataset(test_windows, args.max_cached_tensors, args.mmap_cache_root, PROJECT_ROOT)
            test_loader = make_loader(test_dataset, args, shuffle=False, pin_memory=pin_memory)
            test_reconstructed = reconstruct(model, test_dataset, test_loader, device)
            coverage_rows.extend(prediction_coverage_rows(test_reconstructed, "test"))
            write_test_predictions(run_dir / "test_predictions.csv", test_reconstructed)
            write_prediction_scores(run_dir / "test_prediction_scores.csv.gz", test_reconstructed, supervised_only=False)
            write_pinsoro_submission_tree(run_dir / "test_submission_format", test_reconstructed)
        write_csv(run_dir / "prediction_coverage.csv", coverage_rows)
    print(f"Run directory: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
