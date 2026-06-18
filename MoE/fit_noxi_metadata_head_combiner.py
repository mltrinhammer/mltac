"""Fit/evaluate NOXI MoE combiners from metadata-head expert predictions."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MoE import fit_noxi_moe1_combiner as base


def metadata_prediction_path(root: Path, corpus: str, feature: str, split: str) -> Path:
    run = root / f"{corpus}_{feature}_dyadic_tcn_k11_metadata_head_seed13"
    if split == "train":
        return run / "diagnostics" / "train_internal" / "val_predictions.csv"
    if split == "val":
        return run / "val_predictions.csv"
    raise ValueError(split)


def inject_metadata_defaults(argv: list[str]) -> list[str]:
    corpus = "noxi"
    for idx, arg in enumerate(argv):
        if arg == "--corpus" and idx + 1 < len(argv):
            corpus = argv[idx + 1]
    result = list(argv)
    if "--expert-root" not in result:
        result.extend(
            [
                "--expert-root",
                str(base.EXPERIMENT_ROOT / f"noxi_moe1_{corpus}_metadata_head_experts"),
            ]
        )
    if "--output-root" not in result:
        result.extend(
            [
                "--output-root",
                str(base.EXPERIMENT_ROOT / f"noxi_moe1_{corpus}_metadata_head_combiners"),
            ]
        )
    return result


def main() -> None:
    base.prediction_path = metadata_prediction_path
    sys.argv = inject_metadata_defaults(sys.argv)
    base.main()


if __name__ == "__main__":
    main()
