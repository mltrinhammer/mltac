"""Run shared-head k11 CR-social class-weight ablations on five folds."""

from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import statistics
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_ROOT / "model improvement" / "test 7 shared head CR weights"
BASELINE_ROOT = PROJECT_ROOT / "model improvement/test 5 receptive field ablation/results/w2400_s1200_l5_k11_causal"
COMPLETE_MARKER = ".complete"
REQUIRED_OUTPUTS = ("config.json", "model_best.pt", "model_last.pt", "training_log.csv", "metrics_by_domain.csv", "val_predictions.csv")


@dataclass(frozen=True)
class Experiment:
    mode: str
    suffix: str = ""
    extra_args: tuple[str, ...] = ()

    @property
    def name(self) -> str:
        suffix = f"_{self.suffix}" if self.suffix else ""
        return f"w2400_s1200_l5_k11_shared_head_cr_{self.mode}{suffix}_causal"


EXPERIMENTS = (
    Experiment("unweighted"),
    Experiment("sqrt_inverse"),
    Experiment("capped_inverse", "cap5", ("--cr-social-weight-cap", "5.0")),
    Experiment(
        "targeted",
        "class2x2_class3x0.5",
        ("--cr-social-target-class2-weight", "2.0", "--cr-social-target-class3-weight", "0.5"),
    ),
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
    return PROJECT_ROOT / f"outputs/pinsoro/5foldcv/fold{fold}/windows_w2400_s1200/visual_videomae_w2400_s1200_dyadic.csv"


def run_dir(args: argparse.Namespace, experiment: Experiment, fold: int) -> Path:
    return args.root / "results" / experiment.name / f"fold{fold}_seed{args.seed}"


def complete(path: Path) -> bool:
    return (path / COMPLETE_MARKER).is_file()


def outputs_ready(path: Path) -> bool:
    return all((path / name).is_file() and (path / name).stat().st_size > 0 for name in REQUIRED_OUTPUTS)


def train_command(args: argparse.Namespace, experiment: Experiment, fold: int) -> list[str]:
    return [
        str(args.python), str(PROJECT_ROOT / "scripts/train_pinsoro_tcn.py"),
        "--manifest", str(manifest_path(fold)), "--model", "attention",
        "--causal-tcn", "--causal-attention", "--no-domain-social-heads",
        "--cr-social-weighting", experiment.mode, *experiment.extra_args,
        "--output-root", str(args.root / "results" / experiment.name),
        "--run-name", f"fold{fold}_seed{args.seed}", "--levels", "5", "--kernel-size", "11",
        "--train-split", "train_internal", "--val-split", "val_internal", "--test-split", "none",
        "--mmap-cache-root", str(PROJECT_ROOT / f"processed/pinsoro/5foldcv_mmap/fold{fold}"),
        "--batch-size", "32", "--epochs", "60", "--min-epochs", "24", "--patience", "16",
        "--min-delta", "0.005", "--seed", str(args.seed), "--device", "cuda", "--resume",
    ]


def analysis_command(args: argparse.Namespace, experiment: Experiment, fold: int) -> list[str]:
    return [str(args.python), str(PROJECT_ROOT / "scripts/analyze_pinsoro_prediction_errors.py"), "--run-dir", str(run_dir(args, experiment, fold))]


def aggregate(root: Path, seed: int) -> dict[str, object]:
    folds: list[float] = []
    regimes: dict[str, list[float]] = {}
    for fold in range(1, 6):
        with (root / f"fold{fold}_seed{seed}/metrics_by_domain.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        values = {f"{row['domain']}_{row['head']}": float(row["kappa"]) for row in rows}
        folds.append(statistics.mean(values.values()))
        for name, value in values.items():
            regimes.setdefault(name, []).append(value)
    return {
        "mean_organizer_score": statistics.mean(folds),
        "stdev_organizer_score": statistics.pstdev(folds),
        "fold_scores": folds,
        "regime_scores": {name: statistics.mean(values) for name, values in regimes.items()},
    }


def write_summary(args: argparse.Namespace) -> None:
    if not all(complete(run_dir(args, experiment, fold)) for experiment in EXPERIMENTS for fold in range(1, 6)):
        return
    baseline = aggregate(BASELINE_ROOT, args.seed)
    rows = [{"name": "shared_inverse_baseline", **baseline}]
    for experiment in EXPERIMENTS:
        result = aggregate(args.root / "results" / experiment.name, args.seed)
        rows.append({
            "name": experiment.name,
            **result,
            "delta_organizer_score": result["mean_organizer_score"] - baseline["mean_organizer_score"],
            "delta_regime_scores": {name: result["regime_scores"][name] - baseline["regime_scores"][name] for name in baseline["regime_scores"]},
        })
    output = args.root / "summaries/shared_head_cr_weight_ablation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def run_experiment(args: argparse.Namespace, experiment: Experiment, gpus: list[str]) -> None:
    pending = [fold for fold in range(1, 6) if not complete(run_dir(args, experiment, fold))]
    print(f"experiment={experiment.name} pending={pending}", flush=True)
    if args.dry_run:
        for fold in pending:
            print(" ".join(train_command(args, experiment, fold)))
        return
    work: queue.Queue[tuple[int, int]] = queue.Queue()
    for fold in pending:
        work.put((fold, 1))
    failures: list[tuple[int, int]] = []
    lock = threading.Lock()

    def worker(gpu: str) -> None:
        while True:
            try:
                fold, attempt = work.get_nowait()
            except queue.Empty:
                return
            path = run_dir(args, experiment, fold)
            log_path = args.root / "logs" / experiment.name / f"fold{fold}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            env["PYTHONUNBUFFERED"] = "1"
            with lock:
                print(f"[{experiment.name}] gpu={gpu} fold={fold} attempt={attempt}", flush=True)
            with log_path.open("a", encoding="utf-8") as log:
                result = subprocess.run(train_command(args, experiment, fold), cwd=PROJECT_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
                if result.returncode == 0 and outputs_ready(path):
                    result = subprocess.run(analysis_command(args, experiment, fold), cwd=PROJECT_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
                if result.returncode == 0 and outputs_ready(path):
                    (path / COMPLETE_MARKER).write_text("training and diagnostics complete\n", encoding="utf-8")
                elif attempt < 2:
                    work.put((fold, attempt + 1))
                else:
                    failures.append((fold, result.returncode))
            work.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        raise RuntimeError(f"{experiment.name} failed folds: {failures}")


def main() -> None:
    args = parse_args()
    gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if not args.python.is_file() or not gpus:
        raise RuntimeError("A valid Python executable and at least one GPU are required.")
    for fold in range(1, 6):
        if not manifest_path(fold).is_file():
            raise FileNotFoundError(manifest_path(fold))
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "experiment_plan.json").write_text(json.dumps({
        "question": "Can CR-specific social class weights improve CR-social while preserving the shared-head k11 model's other regimes?",
        "architecture": "unchanged shared-head w2400_s1200_l5_k11_causal",
        "fixed": "folds, seed, schedule, task weights, CC-social weights, metadata, and decoding",
        "cr_weight_normalization": "Each CR vector has average supervised CR frame weight 1.0.",
        "experiments": [{"name": item.name, "mode": item.mode, "extra_args": item.extra_args} for item in EXPERIMENTS],
    }, indent=2), encoding="utf-8")
    for experiment in EXPERIMENTS:
        run_experiment(args, experiment, gpus)
    if not args.dry_run:
        write_summary(args)
    print(f"Finished shared-head CR-weight ablation setup under {args.root}", flush=True)


if __name__ == "__main__":
    main()
