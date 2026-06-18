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

MANIFEST_ROOT="MoE/noxi_joint_settings/manifests/group_meanpool"
NOXI_MANIFEST="$MANIFEST_ROOT/audio_w2vbert2__text_xlm_roberta__visual_videomae_noxi_w500_s125_group_windows.csv"
NOXIJ_MANIFEST="$MANIFEST_ROOT/audio_w2vbert2__text_xlm_roberta__visual_videomae_noxij_w500_s125_group_windows.csv"

SHARED_NOXI_RUN="MoE/noxi_joint_settings/experiments_group_meanpool/noxi_audio_text_visual_w500_s125_postpred_linear_shared_encoder_seed13"
SHARED_NOXIJ_RUN="MoE/noxi_joint_settings/experiments_group_meanpool/noxij_audio_text_visual_w500_s125_postpred_linear_shared_encoder_seed13"
TEMPORAL_NOXI_RUN="MoE/noxi_joint_settings/experiments_group_meanpool_temporal_loss/noxi_audio_text_visual_w500_s125_postpred_linear_shared_encoder_temporal_mse020_delta010_seed13"
CALIBRATED_NOXIJ_RUN="MoE/noxi_joint_settings/experiments_group_meanpool_calibration_sweep/noxij_audio_text_visual_w500_s125_postpred_linear_shared_encoder_cal_mean001_std001_jit000_seed13"

SUBMISSION_ROOT="submissions"
SHARED_SUBMISSION="$SUBMISSION_ROOT/noxi_shared_ema_seed13"
EDITED_SUBMISSION="$SUBMISSION_ROOT/noxi_temporal_noxij_calibrated_ema_seed13"
WORK_ROOT="run_logs/noxi_submission_prediction_queue"

mkdir -p "$SHARED_SUBMISSION" "$EDITED_SUBMISSION" "$WORK_ROOT"

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "missing required file: $path" >&2
    exit 1
  fi
}

require_dir() {
  local path="$1"
  if [[ ! -d "$path" ]]; then
    echo "missing required directory: $path" >&2
    exit 1
  fi
}

copy_submission_tree() {
  local source_dir="$1"
  local target_dir="$2"
  require_dir "$source_dir"
  cp -a "$source_dir"/. "$target_dir"/
}

run_export() {
  local label="$1"
  local run_dir="$2"
  local manifest="$3"
  local split="$4"
  local alpha="$5"
  local submission_dir="$6"
  local output_dir="$WORK_ROOT/$label"
  local log_path="$WORK_ROOT/$label.log"

  require_file "$manifest"
  require_file "$run_dir/model_best.pt"
  mkdir -p "$output_dir" "$submission_dir"

  echo "=== $label $(date '+%Y-%m-%d %H:%M:%S') ===" | tee -a "$log_path"
  "$PYTHON_BIN" MoE/noxi_joint_settings/evaluate_group_ema_smoothing.py \
    --run-dir "$run_dir" \
    --manifest "$manifest" \
    --split "$split" \
    --alphas "$alpha" \
    --output-dir "$output_dir" \
    --batch-size "${BATCH_SIZE:-32}" \
    --num-workers "${NUM_WORKERS:-1}" \
    --device cuda 2>&1 | tee -a "$log_path"

  copy_submission_tree "$output_dir/alpha_${alpha}_submission_format" "$submission_dir"
}

cat > "$WORK_ROOT/noxi_shared_ema_seed13.models.txt" <<'EOF'
NOXI base/additional:
  run: MoE/noxi_joint_settings/experiments_group_meanpool/noxi_audio_text_visual_w500_s125_postpred_linear_shared_encoder_seed13
  alpha: 0.05
  validation: EMA CCC 0.8508727852

NOXI-J:
  run: MoE/noxi_joint_settings/experiments_group_meanpool/noxij_audio_text_visual_w500_s125_postpred_linear_shared_encoder_seed13
  alpha: 0.05
  validation: EMA CCC 0.5717482968
EOF

cat > "$WORK_ROOT/noxi_temporal_noxij_calibrated_ema_seed13.models.txt" <<'EOF'
NOXI base/additional:
  run: MoE/noxi_joint_settings/experiments_group_meanpool_temporal_loss/noxi_audio_text_visual_w500_s125_postpred_linear_shared_encoder_temporal_mse020_delta010_seed13
  alpha: 0.3
  validation: EMA CCC 0.843656

NOXI-J:
  run: MoE/noxi_joint_settings/experiments_group_meanpool_calibration_sweep/noxij_audio_text_visual_w500_s125_postpred_linear_shared_encoder_cal_mean001_std001_jit000_seed13
  alpha: 0.05
  validation: EMA CCC 0.5717349251
EOF

run_export shared_noxi_base "$SHARED_NOXI_RUN" "$NOXI_MANIFEST" test_internal 0.05 "$SHARED_SUBMISSION"
run_export shared_noxi_additional "$SHARED_NOXI_RUN" "$NOXI_MANIFEST" test_additional 0.05 "$SHARED_SUBMISSION"
run_export shared_noxij "$SHARED_NOXIJ_RUN" "$NOXIJ_MANIFEST" test_internal 0.05 "$SHARED_SUBMISSION"

run_export edited_noxi_base "$TEMPORAL_NOXI_RUN" "$NOXI_MANIFEST" test_internal 0.3 "$EDITED_SUBMISSION"
run_export edited_noxi_additional "$TEMPORAL_NOXI_RUN" "$NOXI_MANIFEST" test_additional 0.3 "$EDITED_SUBMISSION"
run_export edited_noxij "$CALIBRATED_NOXIJ_RUN" "$NOXIJ_MANIFEST" test_internal 0.05 "$EDITED_SUBMISSION"

echo "submission folders ready:"
echo "  $SHARED_SUBMISSION"
echo "  $EDITED_SUBMISSION"
