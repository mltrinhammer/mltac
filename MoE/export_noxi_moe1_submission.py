"""Export NOXI/NOXI-J metadata-head MoE1 combined test predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MoE.train_noxi_metadata_head_tcn import (
    MetadataStats,
    MetadataTurnDataset,
    build_model,
    metadata_collate_fn,
    read_metadata,
    reconstruct_validation,
)
from src.acm_pipeline.dyadic_train_utils import write_organizer_submission_tree
from src.acm_pipeline.turn_data import read_turn_manifest


EXPERIMENT_ROOT = PROJECT_ROOT / "MoE" / "experiments"
FEATURES = ("visual_videomae", "audio_w2vbert2", "text_xlm_roberta")
ROLES = ("novice", "expert")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", choices=("noxi", "noxi_j"), required=True)
    parser.add_argument("--expert-root", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("role", "shared", "uniform"), default="role")
    parser.add_argument("--checkpoint", default="model_best.pt")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def defaults(args: argparse.Namespace) -> None:
    if args.expert_root is None:
        args.expert_root = EXPERIMENT_ROOT / f"noxi_moe1_{args.corpus}_metadata_head_experts"
    if args.weights is None:
        args.weights = EXPERIMENT_ROOT / f"noxi_moe1_{args.corpus}_metadata_head_combiners" / "weights.json"


def run_name(corpus: str, feature: str) -> str:
    return f"{corpus}_{feature}_dyadic_tcn_k11_metadata_head_seed13"


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return torch.device(name)


def load_feature_predictions(
    run_dir: Path,
    split: str,
    checkpoint_name: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> list[dict[str, object]]:
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    manifest = Path(config["manifest"])
    metadata_path = Path(config["metadata"])
    turns = read_turn_manifest(manifest, PROJECT_ROOT, split=split)
    if not turns:
        return []
    stats_config = config["metadata_stats"]
    stats = MetadataStats(
        age_mean=float(stats_config["age_mean"]),
        age_std=float(stats_config["age_std"]),
        languages=tuple(str(item) for item in stats_config["languages"]),
    )
    dataset = MetadataTurnDataset(
        turns,
        read_metadata(metadata_path),
        stats,
        str(config.get("metadata_mode", "age_gender_language")),
        min_frames=int(config.get("min_turn_frames", 5)),
    )
    if len(dataset) == 0:
        return []
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=metadata_collate_fn,
    )
    model = build_model(
        argparse.Namespace(**config),
        int(config["n_features_per_role"]),
        int(config["metadata_dim"]),
    ).to(device)
    checkpoint = torch.load(run_dir / checkpoint_name, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return reconstruct_validation(model, dataset, loader, device)


def combine_reconstructed(
    by_feature: dict[str, list[dict[str, object]]],
    weights_by_role: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    keyed = {
        feature: {(item["example"].dataset, item["example"].session_id): item for item in rows}
        for feature, rows in by_feature.items()
    }
    keys = sorted(set.intersection(*(set(rows) for rows in keyed.values())))
    if not keys:
        raise RuntimeError("No aligned NOXI sessions across feature experts.")
    combined = []
    for key in keys:
        reference = keyed[FEATURES[0]][key]
        common_len = min(int(np.asarray(keyed[feature][key]["y_pred"]).shape[0]) for feature in FEATURES)
        y_pred = np.zeros((common_len, 2), dtype=np.float64)
        covered = np.asarray(reference["covered"][:common_len], dtype=np.float32) > 0
        for feature in FEATURES:
            covered &= np.asarray(keyed[feature][key]["covered"][:common_len], dtype=np.float32) > 0
        for channel, role in enumerate(ROLES):
            for weight, feature in zip(weights_by_role[role], FEATURES):
                y_pred[:, channel] += (
                    float(weight)
                    * np.asarray(keyed[feature][key]["y_pred"][:common_len], dtype=np.float64)[:, channel]
                )
        combined.append(
            {
                "example": reference["example"],
                "y_true": reference["y_true"][:common_len],
                "target_mask": reference["target_mask"][:common_len],
                "y_pred": y_pred.astype(np.float32),
                "covered": covered.astype(np.float32),
            }
        )
    return combined


def weights_for_mode(path: Path, mode: str) -> dict[str, np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    weights = data["weights"][mode]
    if mode == "shared":
        values = np.asarray(weights["all"], dtype=np.float64)
        return {role: values for role in ROLES}
    if mode == "uniform":
        values = np.asarray(weights["all"], dtype=np.float64)
        return {role: values for role in ROLES}
    return {role: np.asarray(weights[role], dtype=np.float64) for role in ROLES}


def main() -> None:
    args = parse_args()
    defaults(args)
    device = resolve_device(args.device)
    weights = weights_for_mode(args.weights, args.mode)
    splits = ["test_internal", "test_additional"] if args.corpus == "noxi" else ["test_internal"]
    total_sessions = 0
    for split in splits:
        by_feature = {
            feature: load_feature_predictions(
                args.expert_root / run_name(args.corpus, feature),
                split,
                args.checkpoint,
                args.batch_size,
                args.num_workers,
                device,
            )
            for feature in FEATURES
        }
        if not any(by_feature.values()):
            continue
        reconstructed = combine_reconstructed(by_feature, weights)
        write_organizer_submission_tree(args.output_dir, reconstructed)
        total_sessions += len(reconstructed)
        print(f"corpus={args.corpus} split={split} sessions={len(reconstructed)}", flush=True)
    (args.output_dir / f"{args.corpus}_export_manifest.json").write_text(
        json.dumps(
            {
                "corpus": args.corpus,
                "expert_root": str(args.expert_root),
                "weights": str(args.weights),
                "mode": args.mode,
                "sessions": total_sessions,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
