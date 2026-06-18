"""Train NOXI/NOXI-J MoE modality experts and export train predictions."""

from __future__ import annotations

import argparse
import os
import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MOE_ROOT = PROJECT_ROOT / "MoE"
EXPERIMENT_ROOT = MOE_ROOT / "experiments"
COMPLETE_MARKER = ".complete"
FEATURES = ("visual_videomae", "audio_w2vbert2", "text_xlm_roberta")


@dataclass(frozen=True)
class Corpus:
    name: str
    data_dir: str
    test_splits: tuple[str, ...]


CORPORA = {
    "noxi": Corpus("noxi", "noxi_data", ("test_internal", "test_additional")),
    "noxi_j": Corpus("noxi_j", "noxi_j_data", ("test_internal",)),
}


@dataclass(frozen=True)
class Expert:
    corpus: Corpus
    feature: str
    window_size: int
    stride: int

    @property
    def manifest(self) -> Path:
        return (
            MOE_ROOT
            / self.corpus.data_dir
            / "outputs"
            / f"windows_w{self.window_size}_s{self.stride}"
            / f"{self.feature}_w{self.window_size}_s{self.stride}_dyadic.csv"
        )

    @property
    def run_name(self) -> str:
        return f"{self.corpus.name}_{self.feature}_dyadic_tcn_k11_seed13"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run NOXI MoE dyadic TCN experts across available GPUs."
    )
    parser.add_argument("--corpus", choices=sorted(CORPORA), default="noxi")
    parser.add_argument("--python", type=Path, default=PROJECT_ROOT / ".venv-gpu/bin/python")
    parser.add_argument("--gpus", default="0,1,2")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--features", nargs="+", choices=FEATURES, default=list(FEATURES))
    parser.add_argument("--window-size", type=int, default=2000)
    parser.add_argument("--stride", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--min-epochs", type=int, default=24)
    parser.add_argument("--patience", type=int, default=16)
    parser.add_argument("--min-delta", type=float, default=0.001)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    args.corpus_config = CORPORA[args.corpus]
    if args.root is None:
        args.root = EXPERIMENT_ROOT / f"noxi_moe1_{args.corpus}_experts"
    return args


def run_dir(args: argparse.Namespace, expert: Expert) -> Path:
    return args.root / expert.run_name


def is_complete(path: Path) -> bool:
    return (path / COMPLETE_MARKER).is_file()


def train_command(args: argparse.Namespace, expert: Expert) -> list[str]:
    return [
        str(args.python),
        str(PROJECT_ROOT / "scripts/train_tcn_turns.py"),
        "--manifest",
        str(expert.manifest),
        "--model",
        "dyadic_shared",
        "--output-root",
        str(args.root),
        "--run-name",
        expert.run_name,
        "--levels",
        "5",
        "--kernel-size",
        "11",
        "--hidden-channels",
        "64",
        "--dropout",
        "0.2",
        "--train-split",
        "train_internal",
        "--val-split",
        "val_internal",
        "--test-splits",
        *expert.corpus.test_splits,
        "--batch-size",
        str(args.batch_size),
        "--epochs",
        str(args.epochs),
        "--min-epochs",
        str(args.min_epochs),
        "--patience",
        str(args.patience),
        "--min-delta",
        str(args.min_delta),
        "--seed",
        str(args.seed),
        "--device",
        "cuda",
    ]


def eval_command(args: argparse.Namespace, expert: Expert) -> list[str]:
    return [
        str(args.python),
        str(MOE_ROOT / "evaluate_noxi_checkpoint.py"),
        "--run-dir",
        str(run_dir(args, expert)),
        "--split",
        "train_internal",
        "--checkpoint",
        "model_best.pt",
        "--output-dir",
        str(run_dir(args, expert) / "diagnostics" / "train_internal"),
        "--device",
        "cuda",
    ]


def run_expert(args: argparse.Namespace, expert: Expert, gpu: str) -> None:
    current = run_dir(args, expert)
    if is_complete(current):
        print(f"skip_complete {expert.run_name}", flush=True)
        return
    log_path = args.root / "logs" / f"{expert.run_name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONUNBUFFERED"] = "1"
    commands = (train_command(args, expert), eval_command(args, expert))
    if args.dry_run:
        for command in commands:
            print(" ".join(command), flush=True)
        return
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} gpu={gpu} {expert.run_name} ===\n")
        log.flush()
        for command in commands:
            log.write("$ " + " ".join(command) + "\n")
            log.flush()
            result = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(f"{expert.run_name} failed: {' '.join(command)}")
        (current / COMPLETE_MARKER).write_text(
            "training and train-prediction export complete\n", encoding="utf-8"
        )
    print(f"complete {expert.run_name}", flush=True)


def main() -> None:
    args = normalize_args(parse_args())
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpus:
        raise ValueError("At least one GPU must be provided.")
    experts = tuple(
        Expert(args.corpus_config, feature, args.window_size, args.stride)
        for feature in args.features
    )
    for expert in experts:
        if not expert.manifest.is_file():
            raise FileNotFoundError(expert.manifest)
    args.root.mkdir(parents=True, exist_ok=True)

    work: queue.Queue[Expert] = queue.Queue()
    for expert in experts:
        work.put(expert)
    failures: list[str] = []
    lock = threading.Lock()

    def worker(gpu: str) -> None:
        while True:
            try:
                expert = work.get_nowait()
            except queue.Empty:
                return
            try:
                with lock:
                    print(f"gpu={gpu} start {expert.run_name}", flush=True)
                run_expert(args, expert, gpu)
            except Exception as exc:
                failures.append(f"{expert.run_name}: {exc}")
            finally:
                work.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,), daemon=False) for gpu in gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        raise RuntimeError("; ".join(failures))


if __name__ == "__main__":
    main()
