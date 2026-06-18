"""Fit a temporal attention router over frozen MoE 1 expert scores."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = PROJECT_ROOT / "MoE" / "experiments"
FEATURES = ("visual_videomae", "audio_w2vbert2", "text_xlm_roberta")
HEADS = ("task", "social")
CLASS_COUNTS = {"task": 4, "social": 5}
MAX_CLASSES = max(CLASS_COUNTS.values())
Key = tuple[str, str, str, str, str, int]
GroupKey = tuple[str, str, str, str]


@dataclass(frozen=True)
class SequenceItem:
    group: GroupKey
    frame_idx: np.ndarray
    labels: np.ndarray
    logits: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a causal temporal attention router from frozen MoE expert logits."
    )
    parser.add_argument("--domain", choices=("CC", "CR", "cc", "cr"), default="CC")
    parser.add_argument("--expert-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--run-name", default="temporal_attention_router")
    parser.add_argument("--input-kind", choices=("logits", "probs"), default="logits")
    parser.add_argument("--combine-kind", choices=("logits", "probs"), default="logits")
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--entropy-penalty", type=float, default=0.0)
    parser.add_argument("--smoothness-penalty", type=float, default=1e-3)
    parser.add_argument("--max-train-chunks", type=int, default=0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.domain = args.domain.upper()
    domain_lower = args.domain.lower()
    if args.expert_root is None:
        args.expert_root = EXPERIMENT_ROOT / f"moe1_{domain_lower}_experts"
    if args.output_root is None:
        args.output_root = EXPERIMENT_ROOT / f"moe1_{domain_lower}_temporal_attention_router"
    if args.stride <= 0 or args.chunk_size <= 0:
        raise ValueError("chunk-size and stride must be positive.")
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return torch.device(name)


def expert_run_dir(root: Path, feature: str, domain: str) -> Path:
    return root / f"{domain.lower()}_{feature}_dyadic_tcn_k11_seed13"


def score_path(root: Path, feature: str, split: str, domain: str) -> Path:
    run = expert_run_dir(root, feature, domain)
    if split == "train":
        return run / "diagnostics" / "train_internal" / "val_prediction_scores.csv.gz"
    if split == "val":
        return run / "val_prediction_scores.csv.gz"
    raise ValueError(split)


def read_scores(path: Path) -> dict[Key, dict[str, object]]:
    opener = gzip.open if path.suffix == ".gz" else open
    rows: dict[Key, dict[str, object]] = {}
    with opener(path, "rt", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            head = row["head"]
            n_classes = CLASS_COUNTS[head]
            key = (
                row["domain"],
                row["source_split"],
                row["session_id"],
                row["role"],
                head,
                int(row["frame_idx"]),
            )
            logits = np.full(MAX_CLASSES, -1.0e9, dtype=np.float32)
            logits[:n_classes] = [float(row[f"logit_{idx}"]) for idx in range(n_classes)]
            rows[key] = {"y_true": int(row["y_true"]), "logits": logits}
    return rows


def aligned_split(root: Path, split: str, domain: str) -> tuple[list[Key], np.ndarray, np.ndarray]:
    by_feature = {feature: read_scores(score_path(root, feature, split, domain)) for feature in FEATURES}
    keys = sorted(set.intersection(*(set(rows) for rows in by_feature.values())))
    if not keys:
        raise RuntimeError(f"No aligned score rows for split {split}.")
    labels = np.asarray([by_feature[FEATURES[0]][key]["y_true"] for key in keys], dtype=np.int64)
    logits = np.stack(
        [
            np.stack([by_feature[feature][key]["logits"] for feature in FEATURES], axis=0)
            for key in keys
        ],
        axis=0,
    ).astype(np.float32)
    return keys, labels, logits


def softmax_np(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def router_inputs(logits: np.ndarray, input_kind: str) -> np.ndarray:
    if input_kind == "logits":
        clipped = logits.copy()
        clipped[clipped < -1.0e8] = 0.0
        return clipped.reshape(logits.shape[0], -1)
    if input_kind == "probs":
        probs = np.zeros_like(logits)
        for head, n_classes in CLASS_COUNTS.items():
            del head
            probs[..., :n_classes] = softmax_np(logits[..., :n_classes])
        return probs.reshape(logits.shape[0], -1)
    raise ValueError(input_kind)


def sequence_items(
    keys: list[Key],
    labels: np.ndarray,
    logits: np.ndarray,
) -> list[SequenceItem]:
    grouped: dict[GroupKey, list[int]] = defaultdict(list)
    for idx, key in enumerate(keys):
        grouped[key[:5]].append(idx)
    items = []
    for group, indices in sorted(grouped.items()):
        order = np.asarray(indices, dtype=np.int64)
        order = order[np.argsort([keys[idx][5] for idx in order])]
        items.append(
            SequenceItem(
                group=group,
                frame_idx=np.asarray([keys[idx][5] for idx in order], dtype=np.int64),
                labels=labels[order],
                logits=logits[order],
            )
        )
    return items


class ChunkDataset(Dataset):
    def __init__(
        self,
        sequences: list[SequenceItem],
        input_kind: str,
        chunk_size: int,
        stride: int,
        max_chunks: int,
        seed: int,
    ) -> None:
        self.examples: list[tuple[int, int, int]] = []
        self.sequences = sequences
        self.input_kind = input_kind
        for seq_idx, sequence in enumerate(sequences):
            n = len(sequence.labels)
            if n <= chunk_size:
                self.examples.append((seq_idx, 0, n))
                continue
            starts = list(range(0, n - chunk_size + 1, stride))
            if starts[-1] != n - chunk_size:
                starts.append(n - chunk_size)
            self.examples.extend((seq_idx, start, start + chunk_size) for start in starts)
        if max_chunks > 0 and len(self.examples) > max_chunks:
            rng = np.random.default_rng(seed)
            chosen = np.sort(rng.choice(len(self.examples), size=max_chunks, replace=False))
            self.examples = [self.examples[idx] for idx in chosen]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        seq_idx, start, end = self.examples[idx]
        sequence = self.sequences[seq_idx]
        logits = sequence.logits[start:end]
        return {
            "x": torch.from_numpy(router_inputs(logits, self.input_kind)),
            "labels": torch.from_numpy(sequence.labels[start:end]),
            "logits": torch.from_numpy(logits),
        }


def collate_chunks(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    max_len = max(item["labels"].shape[0] for item in batch)
    input_dim = batch[0]["x"].shape[1]
    x = torch.zeros((len(batch), max_len, input_dim), dtype=torch.float32)
    labels = torch.full((len(batch), max_len), -1, dtype=torch.long)
    logits = torch.zeros((len(batch), max_len, len(FEATURES), MAX_CLASSES), dtype=torch.float32)
    mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
    for idx, item in enumerate(batch):
        n = item["labels"].shape[0]
        x[idx, :n] = item["x"]
        labels[idx, :n] = item["labels"]
        logits[idx, :n] = item["logits"]
        mask[idx, :n] = True
    return {"x": x, "labels": labels, "logits": logits, "mask": mask}


class TemporalAttentionRouter(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_heads: int, layers: int, dropout: float) -> None:
        super().__init__()
        if hidden_dim % n_heads != 0:
            raise ValueError("hidden-dim must be divisible by heads.")
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.gate = nn.Linear(hidden_dim, len(FEATURES))

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.input_projection(x)
        time = hidden.shape[1]
        causal_mask = torch.triu(
            torch.ones((time, time), dtype=torch.bool, device=hidden.device),
            diagonal=1,
        )
        hidden = self.encoder(hidden, mask=causal_mask, src_key_padding_mask=padding_mask)
        return torch.softmax(self.gate(hidden), dim=-1)


def masked_router_loss(
    weights: torch.Tensor,
    expert_logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    combine_kind: str,
    entropy_penalty: float,
    smoothness_penalty: float,
) -> torch.Tensor:
    if combine_kind == "logits":
        combined = (weights.unsqueeze(-1) * expert_logits).sum(dim=2)
        valid_logits = combined[mask]
    elif combine_kind == "probs":
        probs = torch.softmax(expert_logits, dim=-1)
        combined_probs = (weights.unsqueeze(-1) * probs).sum(dim=2).clamp_min(1.0e-8)
        valid_logits = torch.log(combined_probs[mask])
    else:
        raise ValueError(combine_kind)
    loss = nn.functional.cross_entropy(valid_logits, labels[mask])
    if entropy_penalty:
        entropy = -(weights.clamp_min(1.0e-8).log() * weights).sum(dim=-1)
        loss = loss + entropy_penalty * entropy[mask].mean()
    if smoothness_penalty and weights.shape[1] > 1:
        smooth = (weights[:, 1:] - weights[:, :-1]).square().sum(dim=-1)
        smooth_mask = mask[:, 1:] & mask[:, :-1]
        if smooth_mask.any():
            loss = loss + smoothness_penalty * smooth[smooth_mask].mean()
    return loss


def train_one_epoch(
    model: TemporalAttentionRouter,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
) -> float:
    model.train()
    losses = []
    non_blocking = device.type == "cuda"
    for batch in loader:
        batch = {key: value.to(device, non_blocking=non_blocking) for key, value in batch.items()}
        weights = model(batch["x"], ~batch["mask"])
        loss = masked_router_loss(
            weights,
            batch["logits"],
            batch["labels"],
            batch["mask"],
            args.combine_kind,
            args.entropy_penalty,
            args.smoothness_penalty,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


@torch.inference_mode()
def predict_sequences(
    model: TemporalAttentionRouter,
    sequences: list[SequenceItem],
    input_kind: str,
    combine_kind: str,
    chunk_size: int,
    device: torch.device,
) -> tuple[dict[GroupKey, np.ndarray], dict[GroupKey, np.ndarray]]:
    model.eval()
    predictions = {}
    mean_weights = {}
    for sequence in sequences:
        preds = np.empty(len(sequence.labels), dtype=np.int64)
        weights_out = np.empty((len(sequence.labels), len(FEATURES)), dtype=np.float32)
        for start in range(0, len(sequence.labels), chunk_size):
            end = min(start + chunk_size, len(sequence.labels))
            logits = sequence.logits[start:end]
            x = torch.from_numpy(router_inputs(logits, input_kind)).unsqueeze(0).to(device)
            mask = torch.zeros((1, end - start), dtype=torch.bool, device=device)
            weights = model(x, mask).squeeze(0).cpu().numpy()
            if combine_kind == "logits":
                combined = (weights[:, :, None] * logits).sum(axis=1)
            else:
                combined = (weights[:, :, None] * softmax_np(logits)).sum(axis=1)
            preds[start:end] = combined.argmax(axis=1)
            weights_out[start:end] = weights
        predictions[sequence.group] = preds
        mean_weights[sequence.group] = weights_out.mean(axis=0)
    return predictions, mean_weights


def kappa(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(confusion, (y_true, y_pred), 1)
    n = confusion.sum()
    if n == 0:
        return float("nan")
    accuracy = np.trace(confusion) / n
    expected = confusion.sum(axis=1) @ confusion.sum(axis=0) / (n * n)
    return float((accuracy - expected) / (1.0 - expected)) if expected < 1.0 else float("nan")


def metric_rows(
    sequences: list[SequenceItem],
    predictions: dict[GroupKey, np.ndarray],
    mean_weights: dict[GroupKey, np.ndarray],
    mode: str,
) -> tuple[list[dict[str, object]], float]:
    rows = []
    kappas = []
    for head in HEADS:
        labels = []
        preds = []
        weights = []
        domain = ""
        for sequence in sequences:
            if sequence.group[4] != head:
                continue
            domain = sequence.group[0]
            labels.append(sequence.labels)
            preds.append(predictions[sequence.group])
            weights.append(mean_weights[sequence.group])
        y_true = np.concatenate(labels)
        y_pred = np.concatenate(preds)
        head_weights = np.stack(weights).mean(axis=0)
        score = kappa(y_true, y_pred, CLASS_COUNTS[head])
        kappas.append(score)
        rows.append(
            {
                "domain": domain,
                "mode": mode,
                "feature": "combined",
                "head": head,
                "n_frames": int(len(y_true)),
                "kappa": score,
                "accuracy": float((y_true == y_pred).mean()),
                "mean_weights": json.dumps(
                    {feature: float(weight) for feature, weight in zip(FEATURES, head_weights)},
                    sort_keys=True,
                ),
            }
        )
    return rows, float(np.nanmean(kappas))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    train_keys, train_labels, train_logits = aligned_split(args.expert_root, "train", args.domain)
    val_keys, val_labels, val_logits = aligned_split(args.expert_root, "val", args.domain)
    train_sequences = sequence_items(train_keys, train_labels, train_logits)
    val_sequences = sequence_items(val_keys, val_labels, val_logits)
    train_dataset = ChunkDataset(
        train_sequences,
        args.input_kind,
        args.chunk_size,
        args.stride,
        args.max_train_chunks,
        args.seed,
    )
    input_dim = len(FEATURES) * MAX_CLASSES
    run_dir = args.output_root / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args) | {
        "features": list(FEATURES),
        "target_heads": list(HEADS),
        "n_train_rows": len(train_keys),
        "n_val_rows": len(val_keys),
        "n_train_sequences": len(train_sequences),
        "n_val_sequences": len(val_sequences),
        "n_train_chunks": len(train_dataset),
    }
    config = {key: str(value) if isinstance(value, Path) else value for key, value in config.items()}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    if args.dry_run:
        print(json.dumps(config, indent=2), flush=True)
        return

    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_chunks,
        pin_memory=device.type == "cuda",
    )
    model = TemporalAttentionRouter(input_dim, args.hidden_dim, args.heads, args.layers, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    log_rows = []
    best_score = -float("inf")
    best_payload = None
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, loader, optimizer, device, args)
        val_pred, val_weights = predict_sequences(
            model,
            val_sequences,
            args.input_kind,
            args.combine_kind,
            args.chunk_size,
            device,
        )
        rows, val_score = metric_rows(val_sequences, val_pred, val_weights, args.run_name)
        log_rows.append({"epoch": epoch, "train_loss": train_loss, "val_mean_kappa": val_score})
        write_csv(run_dir / "training_log.csv", log_rows)
        write_csv(run_dir / "metrics_by_domain.csv", rows)
        if val_score > best_score:
            best_score = val_score
            best_payload = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_mean_kappa": val_score,
            }
            torch.save(best_payload, run_dir / "model_best.pt")
        print(f"epoch={epoch:03d} train_loss={train_loss:.6f} val_mean_kappa={val_score:.6f}", flush=True)

    if best_payload is not None:
        best_checkpoint = torch.load(run_dir / "model_best.pt", map_location=device, weights_only=False)
        model.load_state_dict(best_checkpoint["model_state_dict"])
    val_pred, val_weights = predict_sequences(
        model,
        val_sequences,
        args.input_kind,
        args.combine_kind,
        args.chunk_size,
        device,
    )
    rows, val_score = metric_rows(val_sequences, val_pred, val_weights, args.run_name)
    write_csv(run_dir / "metrics_by_domain.csv", rows)
    summary = {
        "mode": args.run_name,
        "domain": args.domain,
        "features": list(FEATURES),
        "target_heads": list(HEADS),
        "input_kind": args.input_kind,
        "combine_kind": args.combine_kind,
        "causal_attention": True,
        "optimistic_upper_bound": False,
        "val_mean_kappa": val_score,
        "best_epoch": None if best_payload is None else best_payload["epoch"],
        "n_train_rows": len(train_keys),
        "n_val_rows": len(val_keys),
        "n_train_sequences": len(train_sequences),
        "n_val_sequences": len(val_sequences),
        "n_train_chunks": len(train_dataset),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_root / "comparison.json").write_text(json.dumps([summary], indent=2), encoding="utf-8")
    print(f"{args.run_name} {args.domain} val_mean_kappa={val_score:.6f}", flush=True)


if __name__ == "__main__":
    main()
