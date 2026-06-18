"""Run the focused causal PinSoRo temporal ablation across four GPUs."""

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
from dataclasses import asdict, dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_ROOT / "model improvement" / "temporal ablation causal"
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


@dataclass(frozen=True)
class Experiment:
    window: int
    stride: int
    levels: int
    kernel: int = 7

    @property
    def name(self) -> str:
        return f"w{self.window}_s{self.stride}_l{self.levels}_k{self.kernel}_causal"

    @property
    def receptive_field(self) -> int:
        return 1 + 2 * (self.kernel - 1) * sum(2**index for index in range(self.levels))


EXPERIMENTS = (
    Experiment(window=600, stride=300, levels=5),
    Experiment(window=600, stride=150, levels=5),
    Experiment(window=1200, stride=600, levels=5),
    Experiment(window=900, stride=300, levels=6),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run focused causal PinSoRo temporal ablations sequentially.")
    parser.add_argument("--python", type=Path, default=PROJECT_ROOT / ".venv-gpu/bin/python")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--min-epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=0.005)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def is_complete(run_dir: Path) -> bool:
    return all((run_dir / item).is_file() and (run_dir / item).stat().st_size > 0 for item in REQUIRED_OUTPUTS)


def manifest_path(experiment: Experiment, fold: int) -> Path:
    folder = PROJECT_ROOT / "outputs/pinsoro/5foldcv" / f"fold{fold}" / f"windows_w{experiment.window}_s{experiment.stride}"
    return folder / f"visual_videomae_w{experiment.window}_s{experiment.stride}_dyadic.csv"


def run_dir(args: argparse.Namespace, experiment: Experiment, fold: int) -> Path:
    return args.root / "results" / experiment.name / f"fold{fold}_seed{args.seed}"


def build_manifest(args: argparse.Namespace, experiment: Experiment, fold: int) -> None:
    manifest = manifest_path(experiment, fold)
    if manifest.is_file() and manifest.stat().st_size > 0:
        return
    command = [
        str(args.python), str(PROJECT_ROOT / "scripts/pinsoro_build_window_manifests.py"),
        "--input-manifest", str(PROJECT_ROOT / "outputs/pinsoro/5foldcv" / f"fold{fold}" / "visual_videomae_normalized.csv"),
        "--window-size", str(experiment.window), "--stride", str(experiment.stride),
        "--out-dir", str(manifest.parent),
    ]
    if args.dry_run:
        print(" ".join(command), flush=True)
    else:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def train_command(args: argparse.Namespace, experiment: Experiment, fold: int) -> list[str]:
    return [
        str(args.python), str(PROJECT_ROOT / "scripts/train_pinsoro_tcn.py"),
        "--manifest", str(manifest_path(experiment, fold)), "--model", "attention",
        "--causal-tcn", "--causal-attention",
        "--output-root", str(args.root / "results" / experiment.name),
        "--run-name", f"fold{fold}_seed{args.seed}",
        "--levels", str(experiment.levels), "--kernel-size", str(experiment.kernel),
        "--train-split", "train_internal", "--val-split", "val_internal", "--test-split", "none",
        "--mmap-cache-root", str(PROJECT_ROOT / "processed/pinsoro/5foldcv_mmap" / f"fold{fold}"),
        "--batch-size", str(args.batch_size), "--epochs", str(args.epochs),
        "--min-epochs", str(args.min_epochs), "--patience", str(args.patience),
        "--min-delta", str(args.min_delta), "--seed", str(args.seed), "--device", "cuda", "--resume",
    ]


def analysis_command(args: argparse.Namespace, experiment: Experiment, fold: int) -> list[str]:
    return [str(args.python), str(PROJECT_ROOT / "scripts/analyze_pinsoro_prediction_errors.py"), "--run-dir", str(run_dir(args, experiment, fold))]


def organizer_score(path: Path) -> float:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    kappas = [float(row["kappa"]) for row in rows if row["group"] == "overall"]
    if len(kappas) != 2:
        raise RuntimeError(f"Expected task/social overall kappas in {path}")
    return statistics.mean(kappas)


def write_summary(args: argparse.Namespace, experiment: Experiment) -> None:
    rows = []
    completed_scores = []
    for fold in range(1, 6):
        current = run_dir(args, experiment, fold)
        complete = is_complete(current)
        score: float | str = organizer_score(current / "metrics_by_domain.csv") if complete else ""
        if complete:
            completed_scores.append(float(score))
        rows.append({
            **asdict(experiment), "receptive_field_frames": experiment.receptive_field,
            "fold": fold, "status": "complete" if complete else "incomplete",
            "organizer_score": score, "run_dir": str(current),
        })
    output = args.root / "summaries" / f"{experiment.name}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output)
    aggregate = {
        "experiment": experiment.name,
        "completed_folds": len(completed_scores),
        "mean_organizer_score": statistics.mean(completed_scores) if completed_scores else None,
        "stdev_organizer_score": statistics.pstdev(completed_scores) if len(completed_scores) > 1 else None,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
    }
    aggregate_path = output.with_suffix(".json")
    aggregate_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")


def write_plan(args: argparse.Namespace) -> None:
    args.root.mkdir(parents=True, exist_ok=True)
    plan = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "execution": "experiments sequential; folds parallel across GPUs; failed folds retried once",
        "causal_tcn": True, "causal_attention": True,
        "checkpointing": "model_last.pt atomically after every epoch; model_best.pt on improvement",
        "experiments": [asdict(item) | {"name": item.name, "receptive_field_frames": item.receptive_field} for item in EXPERIMENTS],
    }
    (args.root / "experiment_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")


def run_experiment(args: argparse.Namespace, experiment: Experiment, gpus: list[str]) -> None:
    for fold in range(1, 6):
        build_manifest(args, experiment, fold)
    pending = [fold for fold in range(1, 6) if not is_complete(run_dir(args, experiment, fold))]
    print(f"experiment={experiment.name} rf={experiment.receptive_field} complete={5-len(pending)} pending={len(pending)}", flush=True)
    if args.dry_run:
        for fold in pending:
            print(" ".join(train_command(args, experiment, fold)), flush=True)
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
            log_path = args.root / "logs" / experiment.name / f"fold{fold}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            env["PYTHONUNBUFFERED"] = "1"
            with lock:
                print(f"[{experiment.name}] gpu={gpu} starting fold={fold} attempt={attempt}", flush=True)
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} gpu={gpu} attempt={attempt} resume=true ===\n")
                log.flush()
                result = subprocess.run(train_command(args, experiment, fold), cwd=PROJECT_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
                if result.returncode == 0 and is_complete(run_dir(args, experiment, fold)):
                    analysis = subprocess.run(analysis_command(args, experiment, fold), cwd=PROJECT_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
                    if analysis.returncode != 0:
                        log.write(f"analysis_failed exit={analysis.returncode}\n")
                elif attempt < 2:
                    work.put((fold, attempt + 1))
                else:
                    failures.append((fold, result.returncode))
            with lock:
                write_summary(args, experiment)
                state = "completed" if is_complete(run_dir(args, experiment, fold)) else "retrying" if attempt < 2 else "FAILED"
                print(f"[{experiment.name}] gpu={gpu} {state} fold={fold}", flush=True)
            work.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    write_summary(args, experiment)
    if failures:
        print(f"experiment={experiment.name} failures={failures}; continuing to next experiment", flush=True)


def main() -> None:
    args = parse_args()
    if not args.python.is_file():
        raise FileNotFoundError(f"Python executable not found: {args.python}")
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpus:
        raise ValueError("At least one GPU is required.")
    write_plan(args)
    for experiment in EXPERIMENTS:
        run_experiment(args, experiment, gpus)
    print(f"Finished temporal ablation plan under {args.root}", flush=True)


if __name__ == "__main__":
    main()
