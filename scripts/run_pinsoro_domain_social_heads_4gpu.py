"""Compare domain-specific social heads with the shared-head k11 baseline."""

from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import statistics
import subprocess
import threading
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_ROOT / "model improvement" / "test 6 domain social heads"
BASELINE_ROOT = (
    PROJECT_ROOT
    / "model improvement/test 5 receptive field ablation/results/w2400_s1200_l5_k11_causal"
)
EXPERIMENT_NAME = "w2400_s1200_l5_k11_domain_social_heads_causal"
COMPLETE_MARKER = ".complete"
REQUIRED_OUTPUTS = (
    "config.json",
    "model_best.pt",
    "model_last.pt",
    "training_log.csv",
    "metrics_overall.csv",
    "metrics_by_domain.csv",
    "val_predictions.csv",
    "prediction_coverage.csv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=PROJECT_ROOT / ".venv-gpu/bin/python")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def manifest_path(fold: int) -> Path:
    return (
        PROJECT_ROOT
        / f"outputs/pinsoro/5foldcv/fold{fold}/windows_w2400_s1200"
        / "visual_videomae_w2400_s1200_dyadic.csv"
    )


def run_dir(args: argparse.Namespace, fold: int) -> Path:
    return args.root / "results" / EXPERIMENT_NAME / f"fold{fold}_seed{args.seed}"


def is_complete(path: Path) -> bool:
    return (path / COMPLETE_MARKER).is_file()


def training_complete(path: Path) -> bool:
    return all((path / name).is_file() and (path / name).stat().st_size > 0 for name in REQUIRED_OUTPUTS)


def train_command(args: argparse.Namespace, fold: int) -> list[str]:
    return [
        str(args.python), str(PROJECT_ROOT / "scripts/train_pinsoro_tcn.py"),
        "--manifest", str(manifest_path(fold)), "--model", "attention",
        "--causal-tcn", "--causal-attention", "--domain-social-heads",
        "--output-root", str(args.root / "results" / EXPERIMENT_NAME),
        "--run-name", f"fold{fold}_seed{args.seed}",
        "--levels", "5", "--kernel-size", "11",
        "--train-split", "train_internal", "--val-split", "val_internal", "--test-split", "none",
        "--mmap-cache-root", str(PROJECT_ROOT / f"processed/pinsoro/5foldcv_mmap/fold{fold}"),
        "--batch-size", "32", "--epochs", "60", "--min-epochs", "24", "--patience", "16",
        "--min-delta", "0.005", "--seed", str(args.seed), "--device", "cuda", "--resume",
    ]


def analysis_command(args: argparse.Namespace, fold: int) -> list[str]:
    return [
        str(args.python), str(PROJECT_ROOT / "scripts/analyze_pinsoro_prediction_errors.py"),
        "--run-dir", str(run_dir(args, fold)),
    ]


def metrics(path: Path) -> tuple[float, dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    regimes = {f"{row['domain']}_{row['head']}": float(row["kappa"]) for row in rows}
    return statistics.mean(regimes.values()), regimes


def aggregate(root: Path, seed: int) -> dict[str, object]:
    fold_scores: list[float] = []
    regimes: dict[str, list[float]] = {}
    for fold in range(1, 6):
        score, fold_regimes = metrics(root / f"fold{fold}_seed{seed}" / "metrics_by_domain.csv")
        fold_scores.append(score)
        for name, value in fold_regimes.items():
            regimes.setdefault(name, []).append(value)
    return {
        "mean_organizer_score": statistics.mean(fold_scores),
        "stdev_organizer_score": statistics.pstdev(fold_scores),
        "fold_scores": fold_scores,
        "regime_scores": {name: statistics.mean(values) for name, values in regimes.items()},
    }


def write_comparison(args: argparse.Namespace) -> None:
    if not all(is_complete(run_dir(args, fold)) for fold in range(1, 6)):
        return
    baseline = aggregate(BASELINE_ROOT, args.seed)
    variant = aggregate(args.root / "results" / EXPERIMENT_NAME, args.seed)
    comparison = {
        "baseline": baseline,
        "domain_social_heads": variant,
        "delta_organizer_score": variant["mean_organizer_score"] - baseline["mean_organizer_score"],
        "delta_regime_scores": {
            name: variant["regime_scores"][name] - baseline["regime_scores"][name]
            for name in baseline["regime_scores"]
        },
    }
    summary_dir = args.root / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / "shared_vs_domain_social_heads.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )


def write_plan(args: argparse.Namespace) -> None:
    args.root.mkdir(parents=True, exist_ok=True)
    plan = {
        "question": "Does replacing the shared social head with separate CC and CR social heads improve five-fold performance?",
        "single_change": "Separate CC and CR social classifiers; shared encoder, attention, and task head remain unchanged.",
        "fixed": {
            "architecture": "w2400_s1200_l5_k11_causal",
            "seed": args.seed,
            "folds": 5,
            "class_weights": "same shared task/social class-weight computation as baseline",
            "schedule": "epochs=60 min_epochs=24 patience=16 min_delta=0.005 batch_size=32",
            "metadata": False,
            "decoding_or_smoothing": False,
        },
        "baseline_root": str(BASELINE_ROOT),
        "variant": EXPERIMENT_NAME,
    }
    (args.root / "experiment_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.python.is_file():
        raise FileNotFoundError(args.python)
    for fold in range(1, 6):
        if not manifest_path(fold).is_file():
            raise FileNotFoundError(manifest_path(fold))
    gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if not gpus:
        raise ValueError("At least one GPU is required.")
    write_plan(args)
    pending = [fold for fold in range(1, 6) if not is_complete(run_dir(args, fold))]
    if args.dry_run:
        for fold in pending:
            print(" ".join(train_command(args, fold)))
        return

    work: queue.Queue[tuple[int, int]] = queue.Queue()
    for fold in pending:
        work.put((fold, 1))
    lock = threading.Lock()
    failures: list[tuple[int, int]] = []

    def worker(gpu: str) -> None:
        while True:
            try:
                fold, attempt = work.get_nowait()
            except queue.Empty:
                return
            log_path = args.root / "logs" / f"fold{fold}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            env["PYTHONUNBUFFERED"] = "1"
            with lock:
                print(f"gpu={gpu} starting fold={fold} attempt={attempt}", flush=True)
            with log_path.open("a", encoding="utf-8") as log:
                result = subprocess.run(train_command(args, fold), cwd=PROJECT_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
                if result.returncode == 0 and training_complete(run_dir(args, fold)):
                    result = subprocess.run(analysis_command(args, fold), cwd=PROJECT_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
                if result.returncode == 0 and training_complete(run_dir(args, fold)):
                    (run_dir(args, fold) / COMPLETE_MARKER).write_text("training and diagnostics complete\n", encoding="utf-8")
                elif attempt < 2:
                    work.put((fold, attempt + 1))
                else:
                    failures.append((fold, result.returncode))
            with lock:
                print(f"gpu={gpu} finished fold={fold} complete={is_complete(run_dir(args, fold))}", flush=True)
            work.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    write_comparison(args)
    if failures:
        raise RuntimeError(f"Failed folds: {failures}")
    print(f"Finished comparison under {args.root}", flush=True)


if __name__ == "__main__":
    main()
