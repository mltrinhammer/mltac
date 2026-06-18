"""Run Test 4 long-context ablations and retain only the round winner."""

from __future__ import annotations

import argparse
import csv
import hashlib
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
DEFAULT_ROOT = PROJECT_ROOT / "model improvement" / "test 4 long context ablation"
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


@dataclass(frozen=True)
class Experiment:
    window: int
    stride: int
    levels: int
    kernel: int = 7
    epochs: int = 30
    min_epochs: int = 12
    patience: int = 8

    @property
    def name(self) -> str:
        return f"w{self.window}_s{self.stride}_l{self.levels}_k{self.kernel}_causal"

    @property
    def receptive_field(self) -> int:
        return 1 + 2 * (self.kernel - 1) * sum(2**index for index in range(self.levels))


EXPERIMENTS = (
    Experiment(window=2400, stride=1200, levels=5, kernel=7, epochs=60, min_epochs=24, patience=16),
    Experiment(window=2400, stride=1200, levels=5, kernel=9, epochs=60, min_epochs=24, patience=16),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Test 4 long-context ablations sequentially.")
    parser.add_argument("--python", type=Path, default=PROJECT_ROOT / ".venv-gpu/bin/python")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--min-delta", type=float, default=0.005)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-cleanup", action="store_true")
    return parser.parse_args()


def training_complete(run_dir: Path) -> bool:
    return all((run_dir / item).is_file() and (run_dir / item).stat().st_size > 0 for item in REQUIRED_OUTPUTS)


def is_complete(run_dir: Path) -> bool:
    return (run_dir / COMPLETE_MARKER).is_file()


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
        "--batch-size", str(args.batch_size), "--epochs", str(experiment.epochs),
        "--min-epochs", str(experiment.min_epochs), "--patience", str(experiment.patience),
        "--min-delta", str(args.min_delta), "--seed", str(args.seed), "--device", "cuda", "--resume",
    ]


def analysis_command(args: argparse.Namespace, experiment: Experiment, fold: int) -> list[str]:
    return [str(args.python), str(PROJECT_ROOT / "scripts/analyze_pinsoro_prediction_errors.py"), "--run-dir", str(run_dir(args, experiment, fold))]


def organizer_score(path: Path) -> float:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    kappas = [float(row["kappa"]) for row in rows]
    if len(kappas) != 4:
        raise RuntimeError(f"Expected four domain/head kappas in {path}")
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
        "execution": "experiments sequential; folds parallel across GPUs; failed folds retried once; epoch/min-epoch/patience budgets scaled to approximately 2130 optimizer steps",
        "causal_tcn": True, "causal_attention": True,
        "checkpointing": "model_last.pt atomically after every epoch; model_best.pt on improvement",
        "experiments": [asdict(item) | {"name": item.name, "receptive_field_frames": item.receptive_field} for item in EXPERIMENTS],
    }
    (args.root / "experiment_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")


def run_experiment(args: argparse.Namespace, experiment: Experiment, gpus: list[str]) -> None:
    for fold in range(1, 6):
        build_manifest(args, experiment, fold)
    pending = [fold for fold in range(1, 6) if not is_complete(run_dir(args, experiment, fold))]
    print(f"experiment={experiment.name} rf={experiment.receptive_field} epochs={experiment.epochs} min_epochs={experiment.min_epochs} patience={experiment.patience} complete={5-len(pending)} pending={len(pending)}", flush=True)
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
                if result.returncode == 0 and training_complete(run_dir(args, experiment, fold)):
                    analysis = subprocess.run(analysis_command(args, experiment, fold), cwd=PROJECT_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
                    if analysis.returncode == 0:
                        (run_dir(args, experiment, fold) / COMPLETE_MARKER).write_text("training and diagnostics complete\n", encoding="utf-8")
                    elif attempt < 2:
                        log.write(f"analysis_failed exit={analysis.returncode}; retrying\n")
                        work.put((fold, attempt + 1))
                    else:
                        failures.append((fold, analysis.returncode))
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


def experiment_results(args: argparse.Namespace, experiment: Experiment) -> dict[str, object]:
    fold_scores: list[float] = []
    regimes: dict[str, list[float]] = {}
    transition_ratios: list[float] = []
    true_rates: list[float] = []
    pred_rates: list[float] = []
    for fold in range(1, 6):
        current = run_dir(args, experiment, fold)
        with (current / "metrics_by_domain.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        fold_scores.append(statistics.mean(float(row["kappa"]) for row in rows))
        for row in rows:
            regimes.setdefault(f"{row['domain']}_{row['head']}", []).append(float(row["kappa"]))
        with (current / "error_analysis/validation_group_summary.csv").open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                transition_ratios.append(float(row["transition_rate_ratio"]))
                true_rates.append(float(row["true_transition_rate"]))
                pred_rates.append(float(row["pred_transition_rate"]))
    return {
        **asdict(experiment),
        "name": experiment.name,
        "receptive_field_frames": experiment.receptive_field,
        "mean_organizer_score": statistics.mean(fold_scores),
        "stdev_organizer_score": statistics.pstdev(fold_scores),
        "fold_scores": fold_scores,
        "mean_transition_ratio": statistics.mean(transition_ratios),
        "mean_true_transition_rate": statistics.mean(true_rates),
        "mean_pred_transition_rate": statistics.mean(pred_rates),
        "regime_scores": {key: statistics.mean(values) for key, values in regimes.items()},
    }


def write_round_comparison(args: argparse.Namespace) -> list[dict[str, object]]:
    if not all(is_complete(run_dir(args, experiment, fold)) for experiment in EXPERIMENTS for fold in range(1, 6)):
        return []
    results = [experiment_results(args, experiment) for experiment in EXPERIMENTS]
    summary_dir = args.root / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / "round_comparison.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    rows = []
    for result in results:
        rows.append({key: value for key, value in result.items() if key not in {"fold_scores", "regime_scores"}} | result["regime_scores"])
    with (summary_dir / "round_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return results


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def retain_best_only(args: argparse.Namespace, results: list[dict[str, object]]) -> None:
    if not results:
        return
    winner = max(results, key=lambda row: float(row["mean_organizer_score"]))
    winner_name = str(winner["name"])
    removed: list[str] = []
    for experiment in EXPERIMENTS:
        for fold in range(1, 6):
            current = run_dir(args, experiment, fold)
            targets = [current / "model_last.pt", current / "val_predictions.csv"]
            if experiment.name != winner_name:
                targets.append(current / "model_best.pt")
            for target in targets:
                if target.is_file():
                    target.unlink()
                    removed.append(str(target.relative_to(PROJECT_ROOT)))

    checksum_paths: list[Path] = []
    winner_experiment = next(item for item in EXPERIMENTS if item.name == winner_name)
    for fold in range(1, 6):
        current = run_dir(args, winner_experiment, fold)
        checksum_paths.extend([current / "model_best.pt", current / "config.json", manifest_path(winner_experiment, fold)])
    checksum_file = args.root / "results" / winner_name / "SHA256SUMS"
    checksum_file.write_text(
        "".join(f"{sha256(path)}  {path.relative_to(PROJECT_ROOT)}\n" for path in checksum_paths),
        encoding="utf-8",
    )
    retention = {
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "winner": winner_name,
        "winner_mean_organizer_score": winner["mean_organizer_score"],
        "policy": "Keep all configs, logs, metrics, diagnostics, summaries, and completion markers; keep model_best.pt and manifests only for the round winner; remove model_last.pt and raw val_predictions.csv for all runs.",
        "removed_files": removed,
        "checksum_file": str(checksum_file.relative_to(PROJECT_ROOT)),
    }
    (args.root / "retention.json").write_text(json.dumps(retention, indent=2), encoding="utf-8")
    print(f"Retained full reproducibility artifacts for winner={winner_name}; removed_files={len(removed)}", flush=True)

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
    results = write_round_comparison(args)
    if results and not args.skip_cleanup and not args.dry_run:
        retain_best_only(args, results)
    print(f"Finished temporal ablation plan under {args.root}", flush=True)


if __name__ == "__main__":
    main()
