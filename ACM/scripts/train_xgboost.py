from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# XGBoost is a tabular baseline, so this script summarizes each sequence window
# into fixed descriptors and predicts one engagement value per window. Window
# predictions are then averaged back over frames for CCC-compatible evaluation.
from src.acm_pipeline.data import build_window_table, load_session_tensor, read_model_manifest
from src.acm_pipeline.train_utils import grouped_metric_outputs, write_prediction_csv


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "experiments"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an XGBoost window-summary baseline from any transformed feature manifest.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--train-split", default="train_internal")
    parser.add_argument("--val-split", default="val_internal")
    parser.add_argument("--window-size", type=int, default=500)
    parser.add_argument("--stride", type=int, default=125)
    parser.add_argument("--include-minmax", action="store_true")
    parser.add_argument("--max-train-windows", type=int, default=None)
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def make_run_dir(args: argparse.Namespace) -> Path:
    if args.run_name.strip():
        run_name = args.run_name.strip()
    else:
        run_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_xgboost"
    run_dir = args.output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def serializable_args(args: argparse.Namespace) -> dict[str, object]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def reconstruct_from_window_predictions(examples, metadata: list[dict[str, object]], window_pred: np.ndarray) -> list[dict[str, object]]:
    sums = [np.zeros(example.aligned_len, dtype=np.float64) for example in examples]
    counts = [np.zeros(example.aligned_len, dtype=np.float64) for example in examples]

    # XGBoost predicts one scalar per window. Assign that value to every real
    # frame covered by the window, then average overlapping window predictions.
    for meta, pred in zip(metadata, window_pred):
        ex_idx = int(meta["example_idx"])
        start = int(meta["start"])
        valid_len = int(meta["valid_len"])
        end = start + valid_len
        sums[ex_idx][start:end] += float(pred)
        counts[ex_idx][start:end] += 1.0

    reconstructed = []
    for ex_idx, example in enumerate(examples):
        session = load_session_tensor(example)
        pred = np.full(example.aligned_len, np.nan, dtype=np.float32)
        covered = counts[ex_idx] > 0
        pred[covered] = (sums[ex_idx][covered] / counts[ex_idx][covered]).astype(np.float32)
        reconstructed.append(
            {
                "example": example,
                "y_true": session.y[: example.aligned_len],
                "target_mask": session.target_mask[: example.aligned_len],
                "y_pred": pred,
                "covered": covered.astype(np.float32),
            }
        )
    return reconstructed


def main() -> None:
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise RuntimeError("xgboost is not installed. Install requirements.txt or run: python -m pip install xgboost") from exc

    args = parse_args()
    run_dir = make_run_dir(args)

    train_examples = read_model_manifest(args.manifest, PROJECT_ROOT, split=args.train_split)
    val_examples = read_model_manifest(args.manifest, PROJECT_ROOT, split=args.val_split)
    if not train_examples or not val_examples:
        raise RuntimeError("Both train and validation examples are required.")

    # Build tabular window summaries. This is intentionally in-memory for now:
    # it avoids another persistent preprocessing artifact while the baseline is
    # still simple. If needed, this can become an export script later.
    x_train, y_train, w_train, _ = build_window_table(
        train_examples,
        window_size=args.window_size,
        stride=args.stride,
        include_minmax=args.include_minmax,
        max_windows=args.max_train_windows,
        seed=args.seed,
    )
    x_val, _, _, val_meta = build_window_table(
        val_examples,
        window_size=args.window_size,
        stride=args.stride,
        include_minmax=args.include_minmax,
    )

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_lambda=args.reg_lambda,
        random_state=args.seed,
        n_jobs=-1,
        tree_method="hist",
    )
    model.fit(x_train, y_train, sample_weight=w_train)

    val_pred = model.predict(x_val)
    reconstructed = reconstruct_from_window_predictions(val_examples, val_meta, val_pred)
    metrics = grouped_metric_outputs(run_dir, reconstructed)
    write_prediction_csv(run_dir / "val_predictions.csv", reconstructed)

    with (run_dir / "model.pkl").open("wb") as handle:
        pickle.dump(model, handle)
    with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
        payload = serializable_args(args)
        payload.update({"n_train_windows": int(len(x_train)), "n_val_windows": int(len(x_val)), "window_feature_dim": int(x_train.shape[1])})
        json.dump(payload, handle, indent=2)

    print(f"val_ccc={metrics['ccc']:.5f} val_mae={metrics['mae']:.5f}", flush=True)
    print(f"Run directory: {run_dir}", flush=True)


if __name__ == "__main__":
    main()

