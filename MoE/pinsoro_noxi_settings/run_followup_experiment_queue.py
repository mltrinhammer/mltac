"""Queue follow-up PinSoRo experiments on one GPU.

This script intentionally runs jobs sequentially. It covers only experiment
variants that are supported by the current code paths:

- role_encoder_linear: shared vs separate purple/yellow encoders with fixed
  linear post-logit partner interaction and hard labels.

It does not claim to cover Noxi/Noxi-J/MPII variants; those need their own
regression/group runners because their labels and metrics are different.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MOE_ROOT = PROJECT_ROOT / "MoE"
TRAIN_SCRIPT = MOE_ROOT / "pinsoro_noxi_settings" / "train_person_interaction_fusion.py"
HMM_SCRIPT = MOE_ROOT / "pinsoro_noxi_settings" / "apply_person_interaction_hmm.py"
DEFAULT_MANIFEST_ROOT = MOE_ROOT / "moe_data" / "outputs" / "windows_w2400_s1200"
DEFAULT_OUTPUT_ROOT = MOE_ROOT / "experiments" / "pinsoro_followup_no_interaction_specialists"
FEATURES = ("audio_w2vbert2", "text_xlm_roberta", "visual_videomae")


@dataclass(frozen=True)
class TrainSpec:
    family: str
    domain_scope: str
    active_heads: tuple[str, ...]
    encoder_sharing: str = "shared"
    interaction_mode: str = "linear"
    soft_label_mode: str = "none"

    @property
    def run_name(self) -> str:
        head_label = "both" if set(self.active_heads) == {"task", "social"} else "_".join(self.active_heads)
        return (
            f"pinsoro_{self.domain_scope.lower()}_audio_text_visual_concat_"
            f"{self.encoder_sharing}_encoder_{self.interaction_mode}_{self.soft_label_mode}_{head_label}_seed13"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--families", nargs="+", choices=("role_encoder_linear", "no_interaction_soft_confidence", "head_specialists_no_interaction"), default=["role_encoder_linear"])
    parser.add_argument("--encoder-sharing", nargs="+", choices=("shared", "separate"), default=["shared", "separate"])
    parser.add_argument("--domain-scopes", nargs="+", choices=("CC", "CR"), default=["CC", "CR"])
    parser.add_argument("--manifest-root", type=Path, default=DEFAULT_MANIFEST_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--window-size", type=int, default=2400)
    parser.add_argument("--stride", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--apply-hmm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def manifest_paths(args: argparse.Namespace) -> list[Path]:
    return [args.manifest_root / f"{feature}_w{args.window_size}_s{args.stride}_dyadic.csv" for feature in FEATURES]


def specs(args: argparse.Namespace) -> list[TrainSpec]:
    out: list[TrainSpec] = []
    for domain in args.domain_scopes:
        if "role_encoder_linear" in args.families:
            for encoder_sharing in args.encoder_sharing:
                out.append(TrainSpec("role_encoder_linear", domain, ("task", "social"), encoder_sharing=encoder_sharing))
        if "no_interaction_soft_confidence" in args.families:
            out.append(TrainSpec("no_interaction_soft_confidence", domain, ("task", "social"), encoder_sharing="shared", interaction_mode="none", soft_label_mode="soft_confidence"))
        if "head_specialists_no_interaction" in args.families:
            out.append(TrainSpec("head_specialists_no_interaction", domain, ("task",), encoder_sharing="shared", interaction_mode="none", soft_label_mode="soft_confidence"))
            out.append(TrainSpec("head_specialists_no_interaction", domain, ("social",), encoder_sharing="shared", interaction_mode="none", soft_label_mode="soft_confidence"))
    return out


def train_command(args: argparse.Namespace, spec: TrainSpec) -> list[str]:
    return [
        str(args.python),
        str(TRAIN_SCRIPT),
        "--manifest",
        *(str(path) for path in manifest_paths(args)),
        "--domain-scope",
        spec.domain_scope,
        "--output-root",
        str(args.output_root),
        "--run-name",
        spec.run_name,
        "--fusion-mode",
        "concat",
        "--fusion-channels",
        "64",
        "--person-hidden-channels",
        "64",
        "--person-levels",
        "5",
        "--person-kernel-size",
        "11",
        "--dropout",
        "0.2",
        "--modality-dropout",
        "0.1",
        "--causal-tcn",
        "--encoder-sharing",
        spec.encoder_sharing,
        "--interaction-mode",
        spec.interaction_mode,
        "--interaction-hidden-channels",
        "32",
        "--interaction-kernel-size",
        "5",
        "--interaction-scale",
        "0.1",
        "--soft-label-mode",
        spec.soft_label_mode,
        "--active-heads",
        *spec.active_heads,
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
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
        *( ["--resume"] if args.resume else [] ),
    ]


def hmm_command(args: argparse.Namespace, spec: TrainSpec) -> list[str]:
    run_dir = args.output_root / spec.run_name
    return [
        str(args.python),
        str(HMM_SCRIPT),
        "--run-dir",
        str(run_dir),
        "--manifest",
        *(str(path) for path in manifest_paths(args)),
        "--output-dir",
        str(run_dir / "hmm_smoothing"),
        "--domain",
        spec.domain_scope,
        "--write-test",
    ]


def run(cmd: list[str], log_path: Path, env: dict[str, str], dry_run: bool) -> None:
    print("$ " + " ".join(cmd), flush=True)
    if dry_run:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        result = subprocess.run(cmd, cwd=PROJECT_ROOT.parent, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")


def main() -> None:
    args = parse_args()
    missing = [path for path in manifest_paths(args) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing manifests:\n" + "\n".join(str(path) for path in missing))
    args.output_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    for spec in specs(args):
        run_dir = args.output_root / spec.run_name
        log_path = args.output_root / "logs" / f"{spec.run_name}.log"
        if args.skip_existing and (run_dir / "model_best.pt").is_file() and not args.resume:
            print(f"skip_existing {spec.run_name}", flush=True)
        else:
            run(train_command(args, spec), log_path, env, args.dry_run)
        if args.apply_hmm and set(spec.active_heads) == {"task", "social"}:
            if args.dry_run or (run_dir / "val_prediction_scores.csv.gz").is_file():
                run(hmm_command(args, spec), args.output_root / "logs" / f"{spec.run_name}.hmm.log", env, args.dry_run)
            else:
                print(f"skip_hmm_missing_scores {spec.run_name}", flush=True)


if __name__ == "__main__":
    main()
