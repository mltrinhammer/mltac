"""Run CR metadata-head class-2 social ablations for MoE 1.

Each variant keeps the metadata-head expert architecture unchanged, trains the
three modality experts, exports train logits, then fits the standard MoE 1
combiners. Use --gpus 0,1 on two-GPU machines; the expert runner queues the
three modalities across the available devices.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MOE_ROOT = PROJECT_ROOT / "MoE"
EXPERIMENT_ROOT = MOE_ROOT / "experiments"
PYTHON = PROJECT_ROOT / ".venv-gpu/bin/python"


@dataclass(frozen=True)
class Variant:
    name: str
    description: str
    extra_args: tuple[str, ...]


VARIANTS = (
    Variant(
        "class2x2",
        "CR-social targeted class-2 weight 2.0, class-3 neutral",
        ("--cr-social-weighting", "targeted", "--cr-social-target-class2-weight", "2.0", "--cr-social-target-class3-weight", "1.0"),
    ),
    Variant(
        "class2x4",
        "CR-social targeted class-2 weight 4.0, class-3 neutral",
        ("--cr-social-weighting", "targeted", "--cr-social-target-class2-weight", "4.0", "--cr-social-target-class3-weight", "1.0"),
    ),
    Variant(
        "class2_focal2",
        "CR-social focal loss gamma 2.0 with existing shared inverse weights",
        ("--cr-social-focal-gamma", "2.0"),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CR-social class-2 MoE 1 ablations.")
    parser.add_argument("--python", type=Path, default=PYTHON)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--variants", nargs="+", choices=[variant.name for variant in VARIANTS], default=[variant.name for variant in VARIANTS])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--min-epochs", type=int, default=24)
    parser.add_argument("--patience", type=int, default=16)
    parser.add_argument("--min-delta", type=float, default=0.005)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run(command: list[str], dry_run: bool) -> None:
    print("$ " + " ".join(command), flush=True)
    if dry_run:
        return
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    args = parse_args()
    selected = [variant for variant in VARIANTS if variant.name in set(args.variants)]
    summary = []
    for variant in selected:
        expert_root = EXPERIMENT_ROOT / f"moe1_cr_metadata_head_{variant.name}_experts"
        combiner_root = EXPERIMENT_ROOT / f"moe1_cr_metadata_head_{variant.name}_combiners"
        run(
            [
                str(args.python),
                str(MOE_ROOT / "run_moe1_metadata_head_experts_4gpu.py"),
                "--domain", "CR",
                "--gpus", args.gpus,
                "--root", str(expert_root),
                "--batch-size", str(args.batch_size),
                "--epochs", str(args.epochs),
                "--min-epochs", str(args.min_epochs),
                "--patience", str(args.patience),
                "--min-delta", str(args.min_delta),
                "--num-workers", str(args.num_workers),
                "--seed", str(args.seed),
                *variant.extra_args,
            ],
            args.dry_run,
        )
        run(
            [
                str(args.python),
                str(MOE_ROOT / "fit_moe1_combiner.py"),
                "--domain", "CR",
                "--expert-root", str(expert_root),
                "--output-root", str(combiner_root),
            ],
            args.dry_run,
        )
        summary.append({
            "variant": variant.name,
            "description": variant.description,
            "expert_root": str(expert_root),
            "combiner_root": str(combiner_root),
            "extra_args": list(variant.extra_args),
        })
    if not args.dry_run:
        out = EXPERIMENT_ROOT / "moe1_cr_social_class2_ablation_summary.json"
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
