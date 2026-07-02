from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
UPSTREAM_SCRIPTS = PROJECT_ROOT / "scripts" / "upstream"
if str(UPSTREAM_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_SCRIPTS))

from src.acm_pipeline.pinsoro_data import read_pinsoro_window_manifests  # noqa: E402
from src.acm_pipeline.pinsoro_train_utils import write_metric_outputs, write_prediction_scores, write_predictions  # noqa: E402
from train_gated_fusion import filter_domain, reconstruct  # noqa: E402
from train_pinsoro_tcn import load_checkpoint, make_loader, resolve_device  # noqa: E402
from train_person_interaction_fusion_temporal import (  # noqa: E402
    RoleMetadataPinSoRoWindowDataset,
    SharedPersonFusionInteractionTCN,
    metadata_dim_for_mode,
    read_participant_metadata,
)


MODALITY_SETS: dict[str, tuple[str, ...]] = {
    "atv": ("audio_w2vbert2", "text_xlm_roberta", "visual_videomae"),
    "at": ("audio_w2vbert2", "text_xlm_roberta"),
    "av": ("audio_w2vbert2", "visual_videomae"),
    "tv": ("text_xlm_roberta", "visual_videomae"),
    "a": ("audio_w2vbert2",),
    "t": ("text_xlm_roberta",),
    "v": ("visual_videomae",),
}


class ModalityMaskedModel(nn.Module):
    def __init__(self, model: nn.Module, modality_dims: dict[str, int], keep: tuple[str, ...]) -> None:
        super().__init__()
        self.model = model
        self.modality_dims = dict(modality_dims)
        self.keep = set(keep)
        slices: dict[str, slice] = {}
        start = 0
        for name, dim in self.modality_dims.items():
            slices[name] = slice(start, start + int(dim))
            start += int(dim)
        self.slices = slices

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor | None = None, metadata: torch.Tensor | None = None):
        masked = x.clone()
        for name, span in self.slices.items():
            if name not in self.keep:
                masked[:, :, span, :] = 0
        return self.model(masked, domain_ids, metadata=metadata)


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def build_model(config: dict[str, object], device: torch.device) -> SharedPersonFusionInteractionTCN:
    model = SharedPersonFusionInteractionTCN(
        modality_dims={str(k): int(v) for k, v in dict(config["modality_dims"]).items()},
        fusion_mode=str(config["fusion_mode"]),
        fusion_channels=int(config["fusion_channels"]),
        person_hidden_channels=int(config["person_hidden_channels"]),
        person_levels=int(config["person_levels"]),
        person_kernel_size=int(config["person_kernel_size"]),
        dropout=float(config["dropout"]),
        modality_dropout=float(config["modality_dropout"]),
        causal_tcn=bool(config["causal_tcn"]),
        encoder_sharing=str(config["encoder_sharing"]),
        interaction_mode=str(config["interaction_mode"]),
        interaction_hidden_channels=int(config["interaction_hidden_channels"]),
        interaction_kernel_size=int(config["interaction_kernel_size"]),
        interaction_scale=float(config["interaction_scale"]),
        head_architecture=str(config.get("head_architecture", "shared_tcn")),
        head_adapter_levels=int(config.get("head_adapter_levels", 1)),
        metadata_dim=metadata_dim_for_mode(str(config.get("metadata_mode", "none"))),
        metadata_embedding_dim=int(config.get("metadata_embedding_dim", 16)),
        metadata_dropout=float(config.get("metadata_dropout", 0.2)),
    ).to(device)
    return model


def read_head_kappa(metrics_path: Path, head: str) -> float:
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["head"] == head and row["group"] == "overall":
                return float(row["kappa"])
    raise RuntimeError(f"Missing overall/{head} row in {metrics_path}")


def evaluate_one(run_dir: Path, head: str, output_root: Path, args: argparse.Namespace) -> list[dict[str, object]]:
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    device = resolve_device(args.device)
    model = build_model(config, device)
    checkpoint = load_checkpoint(run_dir / "model_best.pt", device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    manifests = [resolve(path) for path in config["manifest"]]
    windows = filter_domain(
        read_pinsoro_window_manifests(manifests, PROJECT_ROOT, str(config.get("val_split", "val_internal"))),
        str(config.get("domain_scope", "CC")),
    )
    metadata_mode = str(config.get("metadata_mode", "none"))
    if metadata_mode != "none":
        metadata_table = read_participant_metadata(resolve(config["metadata"]))
        dataset = RoleMetadataPinSoRoWindowDataset(
            windows,
            int(config.get("max_cached_tensors", 6)),
            None,
            PROJECT_ROOT,
            metadata_mode=metadata_mode,
            metadata_table=metadata_table,
            age_mean=float(config["metadata_age_mean"]),
            age_std=float(config["metadata_age_std"]),
        )
    else:
        raise RuntimeError("This evaluator currently expects metadata-enabled submitted checkpoints.")

    loader_args = argparse.Namespace(
        batch_size=args.batch_size or int(config.get("batch_size", 32)),
        num_workers=args.num_workers,
        seed=int(config.get("seed", 13)),
    )
    loader = make_loader(dataset, loader_args, shuffle=False, pin_memory=(device.type == "cuda"))
    rows: list[dict[str, object]] = []
    for tag, keep in MODALITY_SETS.items():
        masked_model = ModalityMaskedModel(model, config["modality_dims"], keep).to(device)
        out_dir = output_root / run_dir.name / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        reconstructed = reconstruct(masked_model, dataset, loader, device)
        write_metric_outputs(out_dir, reconstructed)
        if args.write_predictions:
            write_predictions(out_dir / "val_predictions.csv", reconstructed)
            write_prediction_scores(out_dir / "val_prediction_scores.csv.gz", reconstructed)
        kappa = read_head_kappa(out_dir / "metrics_overall.csv", head)
        rows.append(
            {
                "run": run_dir.name,
                "head": head,
                "modality_tag": tag,
                "modalities": "+".join(keep),
                "raw_kappa": f"{kappa:.9f}",
                "output_dir": str(out_dir),
            }
        )
        print(f"{run_dir.name} {head} {tag} {kappa:.6f}", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-run", type=Path, required=True)
    parser.add_argument("--social-run", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--write-predictions", action="store_true")
    args = parser.parse_args()

    output_root = resolve(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    rows.extend(evaluate_one(resolve(args.task_run), "task", output_root, args))
    rows.extend(evaluate_one(resolve(args.social_run), "social", output_root, args))

    summary = output_root / "submitted_checkpoint_modality_mask_summary.csv"
    with summary.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["run", "head", "modality_tag", "modalities", "raw_kappa", "output_dir"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {summary}", flush=True)


if __name__ == "__main__":
    main()
