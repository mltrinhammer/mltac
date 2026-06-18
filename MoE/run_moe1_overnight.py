"""Run the prepared MoE 1 expert and combiner experiments."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MOE_ROOT = PROJECT_ROOT / "MoE"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MoE 1 experiments sequentially by domain: experts first, then combiners."
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        choices=("CC", "CR", "cc", "cr"),
        default=("CC", "CR"),
        help="Domains to run in order. Defaults to CC then CR.",
    )
    parser.add_argument("--python", type=Path, default=PROJECT_ROOT / ".venv-gpu/bin/python")
    parser.add_argument("--gpus", default="0,1,2")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--min-epochs", type=int, default=24)
    parser.add_argument("--patience", type=int, default=16)
    parser.add_argument("--min-delta", type=float, default=0.005)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--combiner-step", type=float, default=0.05)
    parser.add_argument("--skip-experts", action="store_true")
    parser.add_argument("--skip-combiners", action="store_true")
    parser.add_argument("--skip-metadata", action="store_true")
    parser.add_argument("--skip-metadata-head", action="store_true")
    parser.add_argument("--metadata-mode", choices=("age_gender", "age_only", "gender_only"), default="age_gender")
    parser.add_argument("--metadata-dropout", type=float, default=0.2)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def metadata_command(args: argparse.Namespace) -> list[str]:
    return [
        str(args.python),
        str(MOE_ROOT / "prepare_moe_metadata.py"),
    ]


def expert_command(args: argparse.Namespace, domain: str) -> list[str]:
    return [
        str(args.python),
        str(MOE_ROOT / "run_moe1_experts_4gpu.py"),
        "--domain",
        domain,
        "--gpus",
        args.gpus,
        "--epochs",
        str(args.epochs),
        "--min-epochs",
        str(args.min_epochs),
        "--patience",
        str(args.patience),
        "--min-delta",
        str(args.min_delta),
        "--batch-size",
        str(args.batch_size),
    ]


def combiner_command(args: argparse.Namespace, domain: str) -> list[str]:
    return [
        str(args.python),
        str(MOE_ROOT / "fit_moe1_combiner.py"),
        "--domain",
        domain,
        "--step",
        str(args.combiner_step),
    ]


def metadata_head_expert_command(args: argparse.Namespace, domain: str) -> list[str]:
    return [
        str(args.python),
        str(MOE_ROOT / "run_moe1_metadata_head_experts_4gpu.py"),
        "--domain",
        domain,
        "--gpus",
        args.gpus,
        "--metadata-mode",
        args.metadata_mode,
        "--metadata-dropout",
        str(args.metadata_dropout),
        "--epochs",
        str(args.epochs),
        "--min-epochs",
        str(args.min_epochs),
        "--patience",
        str(args.patience),
        "--min-delta",
        str(args.min_delta),
        "--batch-size",
        str(args.batch_size),
    ]


def metadata_head_combiner_command(args: argparse.Namespace, domain: str) -> list[str]:
    domain_lower = domain.lower()
    return [
        str(args.python),
        str(MOE_ROOT / "fit_moe1_combiner.py"),
        "--domain",
        domain,
        "--expert-root",
        str(MOE_ROOT / "experiments" / f"moe1_{domain_lower}_metadata_head_experts"),
        "--output-root",
        str(MOE_ROOT / "experiments" / f"moe1_{domain_lower}_metadata_head_combiners"),
        "--step",
        str(args.combiner_step),
    ]


def run(command: list[str], dry_run: bool) -> None:
    print(" ".join(command), flush=True)
    if dry_run:
        return
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    args = parse_args()
    domains = [domain.upper() for domain in args.domains]
    if not args.skip_combiners and not args.skip_metadata:
        print("=== MoE metadata ===", flush=True)
        run(metadata_command(args), args.dry_run)
    for domain in domains:
        print(f"=== MoE 1 {domain} ===", flush=True)
        if not args.skip_experts:
            run(expert_command(args, domain), args.dry_run)
        if not args.skip_combiners:
            run(combiner_command(args, domain), args.dry_run)
        if not args.skip_metadata_head:
            print(f"=== MoE 1 {domain} metadata-head ===", flush=True)
            run(metadata_head_expert_command(args, domain), args.dry_run)
            if not args.skip_combiners:
                run(metadata_head_combiner_command(args, domain), args.dry_run)


if __name__ == "__main__":
    main()
