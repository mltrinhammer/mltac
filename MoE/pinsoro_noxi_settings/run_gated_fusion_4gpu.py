"""Launch PinSoRo NOXI-settings gated-fusion runs across GPUs."""

from __future__ import annotations

import argparse
import os
import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MOE_ROOT = PROJECT_ROOT / "MoE"
SCRIPT = MOE_ROOT / "pinsoro_noxi_settings" / "train_gated_fusion.py"
DEFAULT_MANIFEST_ROOT = MOE_ROOT / "moe_data" / "outputs" / "windows_w2400_s1200"
DEFAULT_OUTPUT_ROOT = MOE_ROOT / "experiments" / "pinsoro_noxi_settings_gated_fusion"
FEATURES = ("audio_w2vbert2", "text_xlm_roberta", "visual_videomae")


@dataclass(frozen=True)
class RunSpec:
    domain_scope: str

    @property
    def run_name(self) -> str:
        return f"pinsoro_{self.domain_scope.lower()}_audio_text_visual_gated_dyadic_shared_noxi_settings_seed13"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PinSoRo gated-fusion NOXI-settings experiments.")
    parser.add_argument("--python", type=Path, default=PROJECT_ROOT / ".venv-gpu/bin/python")
    parser.add_argument("--gpus", default="0,1,2")
    parser.add_argument("--domain-scopes", nargs="+", choices=("both", "CC", "CR"), default=["both", "CC", "CR"])
    parser.add_argument("--manifest-root", type=Path, default=DEFAULT_MANIFEST_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--window-size", type=int, default=2400)
    parser.add_argument("--stride", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def manifest_paths(args: argparse.Namespace) -> list[Path]:
    return [
        args.manifest_root / f"{feature}_w{args.window_size}_s{args.stride}_dyadic.csv"
        for feature in FEATURES
    ]


def command(args: argparse.Namespace, spec: RunSpec) -> list[str]:
    cmd = [
        str(args.python),
        str(SCRIPT),
        "--manifest",
        *(str(path) for path in manifest_paths(args)),
        "--domain-scope",
        spec.domain_scope,
        "--output-root",
        str(args.output_root),
        "--run-name",
        spec.run_name,
        "--levels",
        "4",
        "--kernel-size",
        "5",
        "--hidden-channels",
        "64",
        "--fusion-channels",
        "64",
        "--dropout",
        "0.2",
        "--modality-dropout",
        "0.1",
        "--batch-size",
        str(args.batch_size),
        "--epochs",
        str(args.epochs),
        "--patience",
        "6",
        "--min-epochs",
        "5",
        "--min-delta",
        "0.001",
        "--lr",
        "1e-3",
        "--weight-decay",
        "1e-4",
        "--seed",
        "13",
        "--device",
        "cuda",
    ]
    if args.resume:
        cmd.append("--resume")
    return cmd


def worker(gpu_queue: queue.Queue[str], run_queue: queue.Queue[RunSpec], args: argparse.Namespace, errors: list[BaseException]) -> None:
    while True:
        try:
            spec = run_queue.get_nowait()
        except queue.Empty:
            return
        gpu = gpu_queue.get()
        try:
            run_dir = args.output_root / spec.run_name
            if (run_dir / "model_best.pt").is_file() and not args.resume:
                print(f"skip_existing {spec.run_name}", flush=True)
                continue
            log_path = args.output_root / "logs" / f"{spec.run_name}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            cmd = command(args, spec)
            if args.dry_run:
                print(" ".join(cmd), flush=True)
                continue
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            env["PYTHONUNBUFFERED"] = "1"
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} gpu={gpu} {spec.run_name} ===\n")
                log.write("$ " + " ".join(cmd) + "\n")
                log.flush()
                result = subprocess.run(cmd, cwd=PROJECT_ROOT.parent, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
            if result.returncode != 0:
                raise RuntimeError(f"{spec.run_name} failed with exit code {result.returncode}")
            print(f"complete {spec.run_name}", flush=True)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
            print(f"error {spec.run_name}: {exc}", flush=True)
        finally:
            gpu_queue.put(gpu)
            run_queue.task_done()


def main() -> None:
    args = parse_args()
    missing = [path for path in manifest_paths(args) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing manifests:\n" + "\n".join(str(path) for path in missing))
    args.output_root.mkdir(parents=True, exist_ok=True)
    run_queue: queue.Queue[RunSpec] = queue.Queue()
    for scope in args.domain_scopes:
        run_queue.put(RunSpec(scope))
    gpu_queue: queue.Queue[str] = queue.Queue()
    for gpu in [item.strip() for item in args.gpus.split(",") if item.strip()]:
        gpu_queue.put(gpu)
    if gpu_queue.empty():
        raise ValueError("No GPUs specified.")
    errors: list[BaseException] = []
    threads = [
        threading.Thread(target=worker, args=(gpu_queue, run_queue, args, errors), daemon=True)
        for _ in range(min(gpu_queue.qsize(), run_queue.qsize()))
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
