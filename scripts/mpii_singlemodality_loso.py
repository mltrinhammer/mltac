from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/work/ACM/mltac-main/ACM/.venv-gpu/bin/python")

FEATURES = (
    "audio_egemaps",
    "audio_w2vbert2",
    "text_xlm_roberta",
    "visual_swin",
    "visual_openpose",
    "visual_clip",
    "visual_dino",
    "visual_videomae",
)

HELDOUT_SESSIONS = ("008", "009", "010", "026", "027", "028")


def run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def real_session_id(session_id: str) -> str:
    return session_id.split("__pair__", 1)[0]


def ensure_25hz(feature: str, args: argparse.Namespace) -> Path:
    manifest = args.manifest_root / f"model_processed_manifest_{feature}_25hz.csv"
    status = args.manifest_root / f"feature_status_{feature}_25hz.csv"
    if manifest.exists():
        rows = read_rows(manifest)
        if rows:
            return manifest

    run(
        [
            str(PYTHON),
            str(PROJECT_ROOT / "scripts/noxi_prepare_feature_tensors_25hz.py"),
            "--feature-set",
            feature,
            "--manifest",
            str(args.raw_manifest),
            "--streams",
            str(args.raw_streams),
            "--out-root",
            str(PROJECT_ROOT / f"processed/mpiii_eval/{feature}_25hz"),
            "--processed-manifest",
            str(manifest),
            "--status-out",
            str(status),
            "--valid-roles",
            "subjectPos1",
            "subjectPos2",
            "subjectPos3",
            "subjectPos4",
        ]
    )
    rows = read_rows(manifest)
    if not rows:
        raise RuntimeError(f"No processed rows for feature {feature}; see {status}")
    return manifest


def write_fold_25hz_manifest(feature: str, source_manifest: Path, heldout: str, args: argparse.Namespace) -> Path:
    rows = read_rows(source_manifest)
    split_rows: list[dict[str, str]] = []
    for row in rows:
        out = dict(row)
        out["model_split"] = "val_internal" if row["session_id"] == heldout else "train_internal"
        split_rows.append(out)
    out_path = args.fold_root / feature / f"heldout_{heldout}" / f"model_processed_manifest_{feature}_25hz_loso.csv"
    write_rows(out_path, split_rows)
    return out_path


def transform_fold(feature: str, heldout: str, fold_25hz_manifest: Path, args: argparse.Namespace) -> Path:
    out_manifest = args.fold_root / feature / f"heldout_{heldout}" / f"model_processed_manifest_{feature}_raw_loso.csv"
    transform_dir = args.fold_root / feature / f"heldout_{heldout}" / "transform"
    if out_manifest.exists() and (transform_dir / "normalizer.npz").exists():
        return out_manifest
    run(
        [
            str(PYTHON),
            str(PROJECT_ROOT / "scripts/noxi_fit_apply_feature_transform.py"),
            "--input-manifest",
            str(fold_25hz_manifest),
            "--method",
            "raw",
            "--out-root",
            str(PROJECT_ROOT / f"processed/transformed/mpiii_loso/{feature}/heldout_{heldout}"),
            "--output-manifest",
            str(out_manifest),
            "--transform-dir",
            str(transform_dir),
        ]
    )
    return out_manifest


def build_pair_windows(feature: str, heldout: str, transformed_manifest: Path, args: argparse.Namespace) -> Path:
    out_manifest = args.fold_root / feature / f"heldout_{heldout}" / f"model_processed_manifest_{feature}_raw_loso_turns.csv"
    if out_manifest.exists():
        return out_manifest
    run(
        [
            str(PYTHON),
            str(PROJECT_ROOT / "scripts/build_allpairs_window_manifest.py"),
            "--input-manifest",
            str(transformed_manifest),
            "--output-manifest",
            str(out_manifest),
            "--window-size",
            str(args.window_size),
            "--stride",
            str(args.stride),
        ]
    )
    return out_manifest


def assert_fold_splits(turn_manifest: Path, heldout: str) -> None:
    rows = read_rows(turn_manifest)
    if not rows:
        raise RuntimeError(f"No turn rows in {turn_manifest}")
    train_real = {real_session_id(row["session_id"]) for row in rows if row["model_split"] == "train_internal"}
    val_real = {real_session_id(row["session_id"]) for row in rows if row["model_split"] == "val_internal"}
    if val_real != {heldout}:
        raise RuntimeError(f"{turn_manifest}: expected val={heldout}, got {sorted(val_real)}")
    if heldout in train_real:
        raise RuntimeError(f"{turn_manifest}: heldout {heldout} leaked into train split")


def train_fold(feature: str, heldout: str, turn_manifest: Path, args: argparse.Namespace) -> None:
    run_name = f"mpii_loso_{feature}_heldout_{heldout}_dyadic_shared_seed{args.seed}"
    run_dir = args.output_root / run_name
    if (run_dir / "model_best.pt").exists() and (run_dir / "metrics_overall.csv").exists():
        print(f"skip existing training run: {run_dir}", flush=True)
        return

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    run(
        [
            str(PYTHON),
            str(PROJECT_ROOT / "scripts/train_tcn_turns.py"),
            "--manifest",
            str(turn_manifest),
            "--output-root",
            str(args.output_root),
            "--run-name",
            run_name,
            "--train-split",
            "train_internal",
            "--val-split",
            "val_internal",
            "--model",
            "dyadic_shared",
            "--hidden-channels",
            "64",
            "--levels",
            "4",
            "--kernel-size",
            "5",
            "--dropout",
            "0.2",
            "--batch-size",
            str(args.batch_size),
            "--epochs",
            str(args.epochs),
            "--patience",
            str(args.patience),
            "--min-epochs",
            str(args.min_epochs),
            "--min-delta",
            "0.001",
            "--lr",
            "1e-3",
            "--weight-decay",
            "1e-4",
            "--ccc-weight",
            "1.0",
            "--mse-weight",
            "0.0",
            "--seed",
            str(args.seed),
            "--device",
            "cuda",
        ],
        env=env,
    )


def summarize(args: argparse.Namespace) -> None:
    rows: list[dict[str, object]] = []
    for feature in args.features:
        for heldout in args.heldout_sessions:
            run_name = f"mpii_loso_{feature}_heldout_{heldout}_dyadic_shared_seed{args.seed}"
            metrics_path = args.output_root / run_name / "metrics_overall.csv"
            if not metrics_path.exists():
                continue
            metric_rows = read_rows(metrics_path)
            if not metric_rows:
                continue
            metric = metric_rows[0]
            rows.append(
                {
                    "feature": feature,
                    "heldout_session": heldout,
                    "run_name": run_name,
                    "n_frames": metric.get("n_frames", ""),
                    "ccc": metric.get("ccc", ""),
                    "mae": metric.get("mae", ""),
                    "rmse": metric.get("rmse", ""),
                    "pearson": metric.get("pearson", ""),
                }
            )
    if rows:
        out_path = args.output_root / "mpii_loso_singlemodality_summary.csv"
        write_rows(out_path, rows)
        print(f"summary: {out_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MPIIGroupInteraction single-modality LOSO benchmark runner.")
    parser.add_argument("--features", nargs="+", default=list(FEATURES))
    parser.add_argument("--heldout-sessions", nargs="+", default=list(HELDOUT_SESSIONS))
    parser.add_argument("--raw-manifest", type=Path, default=PROJECT_ROOT / "outputs/mpiii_eval/model_raw_manifest_train_with_split.csv")
    parser.add_argument("--raw-streams", type=Path, default=PROJECT_ROOT / "outputs/mpiii_eval/model_raw_manifest_streams_train.csv")
    parser.add_argument("--manifest-root", type=Path, default=PROJECT_ROOT / "outputs/mpiii_eval/manifests")
    parser.add_argument("--fold-root", type=Path, default=PROJECT_ROOT / "outputs/mpiii_eval/loso_singlemodality")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/experiments/mpii_loso_singlemodality")
    parser.add_argument("--window-size", type=int, default=500)
    parser.add_argument("--stride", type=int, default=125)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--min-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--cuda-visible-devices", default="1")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    for key, value in list(config.items()):
        if isinstance(value, Path):
            config[key] = str(value)
    with (args.output_root / "runner_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    for feature in args.features:
        print(f"=== feature: {feature} ===", flush=True)
        try:
            manifest_25hz = ensure_25hz(feature, args)
        except Exception as exc:
            print(f"skip feature {feature}: {exc}", flush=True)
            continue
        for heldout in args.heldout_sessions:
            print(f"--- heldout: {heldout} ---", flush=True)
            fold_25hz = write_fold_25hz_manifest(feature, manifest_25hz, heldout, args)
            transformed = transform_fold(feature, heldout, fold_25hz, args)
            turns = build_pair_windows(feature, heldout, transformed, args)
            assert_fold_splits(turns, heldout)
            if not args.prepare_only and not args.skip_training:
                train_fold(feature, heldout, turns, args)
            summarize(args)


if __name__ == "__main__":
    main()
