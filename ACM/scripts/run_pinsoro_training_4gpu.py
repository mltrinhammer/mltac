"""Run the PinSoRo ablation grid across locally visible GPUs."""

from __future__ import annotations
import argparse
import os
import queue
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_SETS = (
    "audio_egemaps",
    "audio_w2vbert2",
    "text_xlm_roberta",
    "visual_swin",
    "visual_openface",
    "visual_openpose",
    "visual_clip",
    "visual_dino",
    "visual_videomae",
)
MODELS = ("simple", "dyadic_shared", "attention")
REQUIRED_OUTPUTS = (
    "config.json",
    "model_best.pt",
    "training_log.csv",
    "metrics_overall.csv",
    "val_predictions.csv",
    "test_predictions.csv",
)


@dataclass(frozen=True)
class Run:
    feature_set: str
    model: str
    seed: int

    @property
    def name(self) -> str:
        return f"pinsoro_{self.feature_set}_{self.model}_seed{self.seed}"

    @property
    def manifest_kind(self) -> str:
        return "individual" if self.model == "simple" else "dyadic"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run PinSoRo training across local GPUs.")
    p.add_argument(
        "--python",
        type=Path,
        default=Path("/home/ucloud/.venvs/acm-pinsoro-gpu/bin/python"),
    )
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument(
        "--features", nargs="+", choices=FEATURE_SETS, default=list(FEATURE_SETS)
    )
    p.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    p.add_argument(
        "--output-root", type=Path, default=PROJECT_ROOT / "outputs/pinsoro/experiments"
    )
    p.add_argument(
        "--log-dir", type=Path, default=PROJECT_ROOT / "outputs/pinsoro/training_logs"
    )
    p.add_argument("--max-output-gb", type=float, default=8.0)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--min-epochs", type=int, default=10)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-cached-tensors", type=int, default=2)
    p.add_argument("--rerun-completed", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def directory_size(path: Path) -> int:
    return (
        sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
        if path.exists()
        else 0
    )


def is_complete(run_dir: Path) -> bool:
    return all(
        (run_dir / name).is_file() and (run_dir / name).stat().st_size > 0
        for name in REQUIRED_OUTPUTS
    )


def command(args: argparse.Namespace, run: Run) -> list[str]:
    manifest = (
        PROJECT_ROOT
        / "outputs/pinsoro/windows"
        / f"{run.feature_set}_w300_s75_{run.manifest_kind}.csv"
    )
    return [
        str(args.python),
        str(PROJECT_ROOT / "scripts/train_pinsoro_tcn.py"),
        "--manifest",
        str(manifest),
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
        "--device",
        "cuda",
    ]


def main() -> None:
    args = parse_args()
    if not args.python.is_file():
        raise FileNotFoundError(f"Python executable not found: {args.python}")
    gpus = [x.strip() for x in args.gpus.split(",") if x.strip()]
    if not gpus:
        raise ValueError("At least one GPU is required.")
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    runs = [
        Run(feature, model, args.seed)
        for feature in args.features
        for model in args.models
    ]
    pending = [
        run
        for run in runs
        if args.rerun_completed or not is_complete(args.output_root / run.name)
    ]
    print(
        f"grid={len(runs)} already_complete={len(runs) - len(pending)} pending={len(pending)} gpus={','.join(gpus)}",
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
    successes: list[str] = []
    stop = threading.Event()
    byte_budget = int(args.max_output_gb * 1024**3)

    def worker(gpu: str) -> None:
        while not stop.is_set():
            try:
                run = work.get_nowait()
            except queue.Empty:
                return
            output_size = directory_size(args.output_root)
            if output_size >= byte_budget:
                with lock:
                    print(
                        f"Stopping before {run.name}: output root is {output_size / 1024**3:.2f} GiB, limit is {args.max_output_gb:.2f} GiB.",
                        flush=True,
                    )
                stop.set()
                work.task_done()
                return
            log_path = args.log_dir / f"{run.name}.log"
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            env["PYTHONUNBUFFERED"] = "1"
            with lock:
                print(f"[gpu {gpu}] starting {run.name}", flush=True)
            with log_path.open("w", encoding="utf-8") as log:
                result = subprocess.run(
                    command(args, run),
                    cwd=PROJECT_ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            with lock:
                if result.returncode == 0 and is_complete(args.output_root / run.name):
                    successes.append(run.name)
                    print(f"[gpu {gpu}] completed {run.name}", flush=True)
                else:
                    failures.append((run.name, result.returncode))
                    print(
                        f"[gpu {gpu}] FAILED {run.name} exit={result.returncode} log={log_path}",
                        flush=True,
                    )
            work.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in gpus]
    start = time.monotonic()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    print(
        f"finished successes={len(successes)} failures={len(failures)} unscheduled={work.qsize()} elapsed_hours={(time.monotonic() - start) / 3600:.2f} filesystem_free_gb={shutil.disk_usage(args.output_root).free / 1024**3:.1f}",
        flush=True,
    )
    if failures or work.qsize() or stop.is_set():
        for name, code in failures:
            print(f"failure {name} exit={code}", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
