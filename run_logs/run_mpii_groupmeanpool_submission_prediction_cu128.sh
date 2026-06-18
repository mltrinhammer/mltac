#!/usr/bin/env bash
set -euo pipefail

cd /workspace/ACM/ACM-clean

PYTHON_BIN="${PYTHON_BIN:-/workspace/venvs/torch-cu128/bin/python}"
GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH=/workspace/ACM/ACM-clean
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

MANIFEST_ROOT="outputs/mpiii_test_submission2/manifests"
GROUP_MANIFEST="$MANIFEST_ROOT/model_processed_manifest_visual_videomae__audio_egemaps__text_xlm_roberta_raw_group_windows.csv"
RUN_DIR="outputs/experiments/mpii_group_meanpool_loso/mpii_group_meanpool_visual_audio_text_gated_seed13_holdout_028"
WORK_ROOT="run_logs/mpii_groupmeanpool_submission_prediction"
SUBMISSION_DIR="submissions/mpii_groupmeanpool_holdout028_alpha1"

mkdir -p "$MANIFEST_ROOT" "$WORK_ROOT" "$SUBMISSION_DIR"

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "missing required file: $path" >&2
    exit 1
  fi
}

require_file "$MANIFEST_ROOT/model_processed_manifest_visual_videomae_raw.csv"
require_file "$MANIFEST_ROOT/model_processed_manifest_audio_egemaps_raw.csv"
require_file "$MANIFEST_ROOT/model_processed_manifest_text_xlm_roberta_raw.csv"
require_file "$RUN_DIR/model_best.pt"

if [[ ! -f "$GROUP_MANIFEST" ]]; then
  "$PYTHON_BIN" scripts/build_mpii_group_window_manifest.py \
    --input-manifests \
      "$MANIFEST_ROOT/model_processed_manifest_visual_videomae_raw.csv" \
      "$MANIFEST_ROOT/model_processed_manifest_audio_egemaps_raw.csv" \
      "$MANIFEST_ROOT/model_processed_manifest_text_xlm_roberta_raw.csv" \
    --output-manifest "$GROUP_MANIFEST" \
    --combo-name visual_videomae__audio_egemaps__text_xlm_roberta \
    --window-frames 500 \
    --stride-frames 125 \
    --test-session-ids 001 002 003 004 005 006
fi

"$PYTHON_BIN" MoE/noxi_joint_settings/evaluate_group_ema_smoothing.py \
  --run-dir "$RUN_DIR" \
  --manifest "$GROUP_MANIFEST" \
  --split test_internal \
  --alphas 1.0 \
  --output-dir "$WORK_ROOT/holdout028_alpha1" \
  --batch-size "${BATCH_SIZE:-16}" \
  --num-workers "${NUM_WORKERS:-0}" \
  --device cuda

cp -a "$WORK_ROOT/holdout028_alpha1/alpha_1_submission_format"/. "$SUBMISSION_DIR"/

echo "MPII group-meanpool submission folder ready:"
echo "  $SUBMISSION_DIR"
