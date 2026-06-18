"""Prepare PinSoRo MoE data for selected modality experts."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MOE_ROOT = PROJECT_ROOT / "MoE"
DEFAULT_FEATURES = ("visual_videomae", "audio_w2vbert2", "text_xlm_roberta")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare 30 Hz, domain-normalized, windowed PinSoRo tensors for "
            "the MoE modality experts."
        )
    )
    parser.add_argument("--features", nargs="+", default=DEFAULT_FEATURES)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--data-root", type=Path, default=MOE_ROOT / "moe_data")
    parser.add_argument("--window-size", type=int, default=2400)
    parser.add_argument("--stride", type=int, default=1200)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-30hz", action="store_true")
    parser.add_argument("--skip-normalize", action="store_true")
    parser.add_argument("--skip-windows", action="store_true")
    parser.add_argument("--skip-mmap", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run(command: list[str], dry_run: bool) -> None:
    print(" ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def maybe_run(command: list[str], output: Path, force: bool, dry_run: bool) -> None:
    if output.is_file() and output.stat().st_size > 0 and not force:
        print(f"skip_existing {output}", flush=True)
        return
    run(command, dry_run)


def manifest_30hz(data_root: Path, feature: str) -> Path:
    return data_root / "outputs" / "manifests" / f"{feature}_30hz.csv"


def manifest_norm(data_root: Path, feature: str) -> Path:
    return data_root / "outputs" / "manifests" / f"{feature}_domain_normalized.csv"


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    status_dir = data_root / "outputs" / "status"
    transforms_dir = data_root / "outputs" / "domain_transforms"
    windows_dir = data_root / "outputs" / f"windows_w{args.window_size}_s{args.stride}"
    split_dir = data_root / "outputs" / f"windows_w{args.window_size}_s{args.stride}_by_domain"
    processed_30hz = data_root / "processed" / "30hz"
    processed_norm = data_root / "processed" / "domain_norm"
    mmap_root = data_root / "processed" / "domain_norm_mmap"

    if not args.skip_30hz:
        for feature in args.features:
            output_manifest = manifest_30hz(data_root, feature)
            command = [
                str(args.python),
                str(PROJECT_ROOT / "scripts/pinsoro_prepare_feature_tensors_30hz.py"),
                "--feature-set",
                feature,
                "--out-root",
                str(processed_30hz / feature),
                "--output-manifest",
                str(output_manifest),
                "--status-out",
                str(status_dir / f"{feature}_30hz_status.csv"),
            ]
            maybe_run(command, output_manifest, args.force, args.dry_run)

    if not args.skip_normalize:
        for feature in args.features:
            output_manifest = manifest_norm(data_root, feature)
            command = [
                str(args.python),
                str(PROJECT_ROOT / "scripts/pinsoro_fit_apply_domain_feature_transform.py"),
                "--input-manifest",
                str(manifest_30hz(data_root, feature)),
                "--out-root",
                str(processed_norm / feature),
                "--output-manifest",
                str(output_manifest),
                "--transform-dir",
                str(transforms_dir / feature),
            ]
            if args.force:
                command.append("--force")
            maybe_run(command, output_manifest, args.force, args.dry_run)

    dyadic_manifests: list[Path] = []
    if not args.skip_windows:
        command = [
            str(args.python),
            str(PROJECT_ROOT / "scripts/pinsoro_build_shared_window_manifests.py"),
            "--input-manifests",
            *[str(manifest_norm(data_root, feature)) for feature in args.features],
            "--window-size",
            str(args.window_size),
            "--stride",
            str(args.stride),
            "--out-dir",
            str(windows_dir),
        ]
        expected = windows_dir / f"{args.features[0]}_w{args.window_size}_s{args.stride}_dyadic.csv"
        maybe_run(command, expected, args.force, args.dry_run)

        for feature in args.features:
            dyadic = windows_dir / f"{feature}_w{args.window_size}_s{args.stride}_dyadic.csv"
            dyadic_manifests.append(dyadic)
            command = [
                str(args.python),
                str(PROJECT_ROOT / "scripts/pinsoro_split_domain_window_manifests.py"),
                "--input-manifest",
                str(dyadic),
                "--out-dir",
                str(split_dir / feature),
                "--prefix",
                f"{feature}_w{args.window_size}_s{args.stride}_dyadic",
            ]
            expected_split = split_dir / feature / f"{feature}_w{args.window_size}_s{args.stride}_dyadic_cc.csv"
            maybe_run(command, expected_split, args.force, args.dry_run)
    else:
        dyadic_manifests = [
            windows_dir / f"{feature}_w{args.window_size}_s{args.stride}_dyadic.csv"
            for feature in args.features
        ]

    if not args.skip_mmap:
        command = [
            str(args.python),
            str(PROJECT_ROOT / "scripts/pinsoro_build_mmap_cache.py"),
            "--manifests",
            *[str(path) for path in dyadic_manifests],
            "--cache-root",
            str(mmap_root),
            "--workers",
            str(args.workers),
        ]
        if args.force:
            command.append("--force")
        run(command, args.dry_run)


if __name__ == "__main__":
    main()
