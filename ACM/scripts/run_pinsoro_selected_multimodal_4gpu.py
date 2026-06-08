"""Select and run a bounded PinSoRo audio-text-visual fusion grid."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import queue
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_FAMILIES = {
    "audio": ("audio_egemaps", "audio_w2vbert2"),
    "text": ("text_xlm_roberta",),
    "visual": (
        "visual_swin",
        "visual_openface",
        "visual_openpose",
        "visual_clip",
        "visual_dino",
        "visual_videomae",
    ),
}
MODELS = ("simple", "dyadic_shared", "attention")
REQUIRED_OUTPUTS = (
    "config.json",
    "model_best.pt",
    "training_log.csv",
    "metrics_overall.csv",
    "val_predictions.csv",
    "test_predictions.csv",
    "test_submission_format/.complete",
)


@dataclass(frozen=True)
class Run:
    features: tuple[str, str, str]
    model: str
    seed: int

    @property
    def combo(self) -> str:
        return "__".join(self.features)

    @property
    def name(self) -> str:
        return f"pinsoro_mm_{self.combo}_{self.model}_seed{self.seed}"

    @property
    def manifest_kind(self) -> str:
        return "individual" if self.model == "simple" else "dyadic"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select top PinSoRo modalities by family and run early fusion."
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=Path("/home/ucloud/.venvs/acm-pinsoro-gpu/bin/python"),
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=PROJECT_ROOT / "outputs/pinsoro/results_summary.csv",
    )
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--models", nargs="+", choices=MODELS)
    parser.add_argument("--top-audio", type=int, default=2)
    parser.add_argument("--top-text", type=int, default=1)
    parser.add_argument("--top-visual", type=int, default=3)
    parser.add_argument(
        "--require-complete-unimodal",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/pinsoro/multimodal_experiments",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/pinsoro/multimodal_training_logs",
    )
    parser.add_argument(
        "--plan-output",
        type=Path,
        default=PROJECT_ROOT / "outputs/pinsoro/selected_multimodal_plan.csv",
    )
    parser.add_argument(
        "--mmap-cache-root",
        type=Path,
        default=PROJECT_ROOT / "processed/pinsoro_mmap",
    )
    parser.add_argument("--max-output-gb", type=float, default=8.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--min-epochs", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-cached-tensors", type=int, default=6)
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--rerun-completed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_unimodal_results(path: Path, seed: int) -> dict[tuple[str, str], float]:
    if not path.is_file():
        raise FileNotFoundError(
            f"PinSoRo result summary not found: {path}. Run collect_pinsoro_results.py "
            "after the unimodal grid completes."
        )
    known_features = {feature for values in FEATURE_FAMILIES.values() for feature in values}
    results: dict[tuple[str, str], float] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            feature = row["feature_set"]
            model = row["model"]
            if feature not in known_features or model not in MODELS:
                continue
            if int(row["seed"]) != seed:
                continue
            results[(feature, model)] = float(row["organizer_score"])
    return results


def select_runs(args: argparse.Namespace) -> tuple[list[Run], list[dict[str, object]]]:
    results = read_unimodal_results(args.results, args.seed)
    expected = {
        (feature, model)
        for features in FEATURE_FAMILIES.values()
        for feature in features
        for model in MODELS
    }
    missing = expected - set(results)
    if args.require_complete_unimodal and missing:
        raise RuntimeError(
            f"Unimodal PinSoRo grid is incomplete: missing {len(missing)}/{len(expected)} "
            "feature/model results. Multimodal selection must wait for the completed grid."
        )
    if args.models:
        selected_models = list(args.models)
    else:
        means = {
            model: sum(
                score for (feature, candidate), score in results.items() if candidate == model
            )
            / sum(1 for feature, candidate in results if candidate == model)
            for model in MODELS
            if any(candidate == model for feature, candidate in results)
        }
        if not means:
            raise RuntimeError("No unimodal PinSoRo results are available for model selection.")
        selected_models = [max(means, key=means.get)]
    top_counts = {
        "audio": args.top_audio,
        "text": args.top_text,
        "visual": args.top_visual,
    }
    selected_features: dict[str, list[str]] = {}
    plan_rows: list[dict[str, object]] = []
    for family, features in FEATURE_FAMILIES.items():
        ranked = sorted(
            features,
            key=lambda feature: max(
                (results.get((feature, model), -float("inf")) for model in selected_models),
                default=-float("inf"),
            ),
            reverse=True,
        )
        available = [
            feature
            for feature in ranked
            if any((feature, model) in results for model in selected_models)
        ]
        count = top_counts[family]
        if count < 1 or count > len(features):
            raise ValueError(f"top-{family} must be between 1 and {len(features)}.")
        if len(available) < count:
            raise RuntimeError(
                f"Only {len(available)} ranked {family} features are available; requested {count}."
            )
        selected_features[family] = available[:count]
        for rank, feature in enumerate(available[:count], start=1):
            plan_rows.append(
                {
                    "row_type": "selected_feature",
                    "family": family,
                    "rank": rank,
                    "feature_set": feature,
                    "model": "|".join(selected_models),
                    "score": max(results[(feature, model)] for model in selected_models if (feature, model) in results),
                    "combination": "",
                    "run_name": "",
                }
            )
    runs = [
        Run(tuple(combo), model, args.seed)
        for combo in itertools.product(
            selected_features["audio"],
            selected_features["text"],
            selected_features["visual"],
        )
        for model in selected_models
    ]
    for run in runs:
        plan_rows.append(
            {
                "row_type": "multimodal_run",
                "family": "",
                "rank": "",
                "feature_set": "",
                "model": run.model,
                "score": "",
                "combination": run.combo,
                "run_name": run.name,
            }
        )
    return runs, plan_rows


def is_complete(run_dir: Path) -> bool:
    return all(
        (run_dir / name).is_file() and (run_dir / name).stat().st_size > 0
        for name in REQUIRED_OUTPUTS
    )


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) if path.exists() else 0


def command(args: argparse.Namespace, run: Run) -> list[str]:
    manifests = [
        PROJECT_ROOT
        / "outputs/pinsoro/windows"
        / f"{feature}_w300_s75_{run.manifest_kind}.csv"
        for feature in run.features
    ]
    result = [
        str(args.python),
        str(PROJECT_ROOT / "scripts/train_pinsoro_tcn.py"),
        "--manifest",
        *(str(path) for path in manifests),
        "--model",
        run.model,
        "--output-root",
        str(args.output_root),
        "--run-name",
        run.name,
        "--seed",
        str(run.seed),
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--min-epochs",
        str(args.min_epochs),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--max-cached-tensors",
        str(args.max_cached_tensors),
        "--mmap-cache-root",
        str(args.mmap_cache_root),
        "--device",
        "cuda",
    ]
    if args.resume:
        result.append("--resume")
    return result


def write_plan(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if not args.python.is_file():
        raise FileNotFoundError(f"Python executable not found: {args.python}")
    cache_marker = args.mmap_cache_root / ".complete"
    if not cache_marker.is_file() and not args.dry_run:
        raise FileNotFoundError(f"Complete PinSoRo mmap cache not found: {cache_marker}")
    gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if not gpus:
        raise ValueError("At least one GPU is required.")
    runs, plan_rows = select_runs(args)
    write_plan(args.plan_output, plan_rows)
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    pending = [
        run for run in runs if args.rerun_completed or not is_complete(args.output_root / run.name)
    ]
    print(
        f"selected_runs={len(runs)} already_complete={len(runs)-len(pending)} "
        f"pending={len(pending)} gpus={','.join(gpus)} plan={args.plan_output}",
        flush=True,
    )
    for run in pending:
        print(" ".join(command(args, run)), flush=True)
    if args.dry_run or not pending:
        return
    work: queue.Queue[Run] = queue.Queue()
    for run in pending:
        work.put(run)
    lock = threading.Lock()
    failures: list[tuple[str, int]] = []
    stop = threading.Event()
    byte_budget = int(args.max_output_gb * 1024**3)

    def worker(gpu: str) -> None:
        while not stop.is_set():
            try:
                run = work.get_nowait()
            except queue.Empty:
                return
            if directory_size(args.output_root) >= byte_budget:
                stop.set()
                work.task_done()
                return
            log_path = args.log_dir / f"{run.name}.log"
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            env["PYTHONUNBUFFERED"] = "1"
            with lock:
                print(f"[gpu {gpu}] starting {run.name}", flush=True)
            with log_path.open("a" if args.resume else "w", encoding="utf-8") as log:
                log.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} gpu={gpu} ===\n")
                log.flush()
                result = subprocess.run(
                    command(args, run),
                    cwd=PROJECT_ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            if result.returncode != 0 or not is_complete(args.output_root / run.name):
                with lock:
                    failures.append((run.name, result.returncode))
                    print(f"[gpu {gpu}] FAILED {run.name} exit={result.returncode}", flush=True)
            else:
                with lock:
                    print(f"[gpu {gpu}] completed {run.name}", flush=True)
            work.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in gpus]
    started = time.monotonic()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    print(
        f"finished failures={len(failures)} unscheduled={work.qsize()} "
        f"elapsed_hours={(time.monotonic()-started)/3600:.2f} "
        f"filesystem_free_gb={shutil.disk_usage(args.output_root).free/1024**3:.1f}",
        flush=True,
    )
    if failures or work.qsize() or stop.is_set():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
