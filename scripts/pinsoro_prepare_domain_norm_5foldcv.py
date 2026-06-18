"""Prepare domain-normalized PinSoRo five-fold artifacts."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "outputs/pinsoro/5foldcv"
DEFAULT_PROCESSED_ROOT = PROJECT_ROOT / "processed/pinsoro/5foldcv_domain_norm"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs/pinsoro/5foldcv_domain_norm"


def parse_window(value: str) -> tuple[int, int]:
    normalized = value.lower().replace("w", "").replace("s", "").replace("_", ":")
    parts = [part for part in normalized.split(":") if part]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("windows must look like 2400:1200")
    return int(parts[0]), int(parts[1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit CC/CR train-only normalizers for each PinSoRo fold, apply them "
            "to every row in that fold manifest, and build compatible windows."
        )
    )
    parser.add_argument("--feature-set", default="visual_videomae")
    parser.add_argument("--folds", nargs="+", type=int, default=(1, 2, 3, 4, 5))
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--windows",
        nargs="+",
        type=parse_window,
        default=((2400, 1200),),
        help="Window/stride pairs such as 2400:1200.",
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without writing artifacts.",
    )
    return parser.parse_args()


def run_command(command: list[str], dry_run: bool) -> None:
    print(" ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    args = parse_args()
    for fold in args.folds:
        fold_name = f"fold{fold}"
        input_manifest = args.input_root / fold_name / f"{args.feature_set}_30hz.csv"
        output_manifest = args.output_root / fold_name / f"{args.feature_set}_domain_normalized.csv"
        processed_root = args.processed_root / fold_name / args.feature_set
        transform_dir = args.output_root / fold_name / "domain_transform"
        if not input_manifest.is_file():
            raise FileNotFoundError(input_manifest)

        transform_command = [
            str(args.python),
            str(PROJECT_ROOT / "scripts/pinsoro_fit_apply_domain_feature_transform.py"),
            "--input-manifest",
            str(input_manifest),
            "--out-root",
            str(processed_root),
            "--output-manifest",
            str(output_manifest),
            "--transform-dir",
            str(transform_dir),
        ]
        if args.force:
            transform_command.append("--force")
        run_command(transform_command, args.dry_run)

        for window, stride in args.windows:
            windows_dir = args.output_root / fold_name / f"windows_w{window}_s{stride}"
            window_command = [
                str(args.python),
                str(PROJECT_ROOT / "scripts/pinsoro_build_window_manifests.py"),
                "--input-manifest",
                str(output_manifest),
                "--window-size",
                str(window),
                "--stride",
                str(stride),
                "--out-dir",
                str(windows_dir),
            ]
            run_command(window_command, args.dry_run)


if __name__ == "__main__":
    main()
