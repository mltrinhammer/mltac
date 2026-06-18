"""Run MoE1 metadata-head experts on the soft-label PinSoRo data tree."""

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
DEFAULT_DATA_ROOT = MOE_ROOT / "moe_data_soft_labels"
DEFAULT_METADATA = MOE_ROOT / "moe_data" / "outputs" / "participant_metadata.csv"
EXPERIMENT_ROOT = MOE_ROOT / "experiments"
FEATURES = ("visual_videomae", "audio_w2vbert2", "text_xlm_roberta")
COMPLETE_MARKER = ".complete"


@dataclass(frozen=True)
class Expert:
    feature: str
    domain: str
    data_root: Path

    @property
    def domain_lower(self) -> str:
        return self.domain.lower()

    @property
    def manifest(self) -> Path:
        return (
            self.data_root
            / "outputs"
            / "windows_w2400_s1200_by_domain"
            / self.feature
            / f"{self.feature}_w2400_s1200_dyadic_{self.domain_lower}.csv"
        )

    @property
    def run_name(self) -> str:
        return f"{self.domain_lower}_{self.feature}_dyadic_tcn_k11_seed13"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MoE1 metadata-head soft-label experts.")
    parser.add_argument("--python", type=Path, default=PROJECT_ROOT / ".venv-gpu/bin/python")
    parser.add_argument("--gpus", default="0,1,2")
    parser.add_argument("--domains", nargs="+", choices=("CC", "CR", "cc", "cr"), default=("CC", "CR"))
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--soft-label-mode", choices=("soft_uniform", "soft_confidence"), required=True)
    parser.add_argument("--metadata-mode", choices=("age_gender", "age_only", "gender_only"), default="age_gender")
    parser.add_argument("--metadata-dropout", type=float, default=0.2)
    parser.add_argument("--cr-social-weighting", choices=("shared_inverse", "unweighted", "sqrt_inverse", "capped_inverse", "targeted"), default="shared_inverse")
    parser.add_argument("--cr-social-weight-cap", type=float, default=5.0)
    parser.add_argument("--cr-social-target-class0-weight", type=float, default=1.0)
    parser.add_argument("--cr-social-target-class2-weight", type=float, default=2.0)
    parser.add_argument("--cr-social-target-class3-weight", type=float, default=0.5)
    parser.add_argument("--cr-social-focal-gamma", type=float, default=0.0)
    parser.add_argument("--cr-social-class3-oversample", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--min-epochs", type=int, default=24)
    parser.add_argument("--patience", type=int, default=16)
    parser.add_argument("--min-delta", type=float, default=0.005)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def train_command(args: argparse.Namespace, expert: Expert) -> list[str]:
    cmd = [
        str(args.python),
        str(MOE_ROOT / "train_moe1_metadata_head_tcn.py"),
        "--manifest",
        str(expert.manifest),
        "--metadata",
        str(args.metadata),
        "--metadata-mode",
        args.metadata_mode,
        "--metadata-dropout",
        str(args.metadata_dropout),
        "--cr-social-weighting",
        args.cr_social_weighting,
        "--cr-social-weight-cap",
        str(args.cr_social_weight_cap),
        "--cr-social-target-class0-weight",
        str(args.cr_social_target_class0_weight),
        "--cr-social-target-class2-weight",
        str(args.cr_social_target_class2_weight),
        "--cr-social-target-class3-weight",
        str(args.cr_social_target_class3_weight),
        "--cr-social-focal-gamma",
        str(args.cr_social_focal_gamma),
        "--cr-social-class3-oversample",
        str(args.cr_social_class3_oversample),
        "--output-root",
        str(args.output_root),
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
        "--causal-tcn",
        "--train-split",
        "train_internal",
        "--val-split",
        "val_internal",
        "--test-split",
        "test_internal",
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
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
        "--soft-label-mode",
        args.soft_label_mode,
    ]
    if args.resume:
        cmd.append("--resume")
    return cmd


def eval_command(args: argparse.Namespace, expert: Expert) -> list[str]:
    return [
        str(args.python),
        str(MOE_ROOT / "evaluate_moe1_metadata_head_checkpoint.py"),
        "--run-dir",
        str(args.output_root / expert.run_name),
        "--split",
        "train_internal",
        "--checkpoint",
        "model_best.pt",
        "--output-dir",
        str(args.output_root / expert.run_name / "diagnostics" / "train_internal"),
        "--device",
        "cuda",
    ]


def run_expert(args: argparse.Namespace, expert: Expert, gpu: str) -> None:
    run_dir = args.output_root / expert.run_name
    if (run_dir / COMPLETE_MARKER).is_file():
        print(f"skip_complete {expert.run_name}", flush=True)
        return
    log_path = args.output_root / "logs" / f"{expert.run_name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    commands = (train_command(args, expert), eval_command(args, expert))
    if args.dry_run:
        for command in commands:
            print(" ".join(command), flush=True)
        return
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONUNBUFFERED"] = "1"
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} gpu={gpu} {expert.run_name} {args.soft_label_mode} ===\n")
        log.flush()
        for command in commands:
            log.write("$ " + " ".join(command) + "\n")
            log.flush()
            result = subprocess.run(command, cwd=PROJECT_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
            if result.returncode != 0:
                raise RuntimeError(f"{expert.run_name} failed with exit code {result.returncode}")
        (run_dir / COMPLETE_MARKER).write_text(f"{args.soft_label_mode} complete\n", encoding="utf-8")
    print(f"complete {expert.run_name}", flush=True)


def main() -> None:
    args = parse_args()
    args.domains = tuple(domain.upper() for domain in args.domains)
    if not args.metadata.is_file():
        raise FileNotFoundError(args.metadata)
    experts = [Expert(feature, domain, args.data_root) for domain in args.domains for feature in FEATURES]
    missing = [expert.manifest for expert in experts if not expert.manifest.is_file()]
    if missing:
        raise FileNotFoundError("Missing manifests:\n" + "\n".join(str(path) for path in missing))
    args.output_root.mkdir(parents=True, exist_ok=True)
    gpu_values = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpu_values:
        raise ValueError("No GPUs specified.")
    gpu_queue: queue.Queue[str] = queue.Queue()
    for gpu in gpu_values:
        gpu_queue.put(gpu)
    work: queue.Queue[Expert] = queue.Queue()
    for expert in experts:
        work.put(expert)
    failures: list[str] = []

    def worker() -> None:
        while True:
            try:
                expert = work.get_nowait()
            except queue.Empty:
                return
            gpu = gpu_queue.get()
            try:
                print(f"gpu={gpu} start {expert.run_name} {args.soft_label_mode}", flush=True)
                run_expert(args, expert, gpu)
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{expert.run_name}: {exc}")
                print(f"error {expert.run_name}: {exc}", flush=True)
            finally:
                gpu_queue.put(gpu)
                work.task_done()

    threads = [threading.Thread(target=worker, daemon=False) for _ in range(min(len(gpu_values), len(experts)))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        raise RuntimeError("; ".join(failures))


if __name__ == "__main__":
    main()
