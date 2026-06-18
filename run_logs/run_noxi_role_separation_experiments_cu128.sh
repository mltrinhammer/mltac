#!/usr/bin/env bash
set -euo pipefail

cd /workspace/ACM/ACM-clean

PYTHON_BIN=${PYTHON_BIN:-/workspace/venvs/torch-cu128/bin/python}
GPU=${GPU:-0}
BATCH_SIZE=${BATCH_SIZE:-32}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-32}
NUM_WORKERS=${NUM_WORKERS:-2}
OUTPUT_ROOT=${OUTPUT_ROOT:-/workspace/ACM/ACM-clean/MoE/noxi_joint_settings/experiments_group_meanpool_role_separation}
LOG_DIR="$OUTPUT_ROOT/logs"
mkdir -p "$LOG_DIR"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/workspace/ACM/ACM-clean
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

NOXI_MANIFEST=/workspace/ACM/ACM-clean/MoE/noxi_joint_settings/manifests/group_meanpool/audio_w2vbert2__text_xlm_roberta__visual_videomae_noxi_w500_s125_group_windows.csv
NOXIJ_MANIFEST=/workspace/ACM/ACM-clean/MoE/noxi_joint_settings/manifests/group_meanpool/audio_w2vbert2__text_xlm_roberta__visual_videomae_noxij_w500_s125_group_windows.csv
TRAIN_SCRIPT=/workspace/ACM/ACM-clean/scripts/train_mpii_group_meanpool_multimodal_calibration.py
EVAL_SCRIPT=/workspace/ACM/ACM-clean/MoE/noxi_joint_settings/evaluate_group_ema_smoothing.py

run_train() {
  local run_name="$1"
  shift
  local run_dir="$OUTPUT_ROOT/$run_name"
  if [[ -f "$run_dir/model_best.pt" ]]; then
    echo "skip_existing_train $run_name"
  else
    echo "train $run_name"
    "$PYTHON_BIN" "$TRAIN_SCRIPT"       --output-root "$OUTPUT_ROOT"       --run-name "$run_name"       --train-split train_internal       --val-split val_internal       --fusion-mode gated       --fusion-channels 64       --modality-dropout 0.1       --hidden-channels 64       --levels 4       --kernel-size 5       --dropout 0.2       --encoder-sharing shared       --prediction-interaction-scale 0.1       --batch-size "$BATCH_SIZE"       --epochs 50       --patience 6       --min-epochs 5       --min-delta 0.001       --lr 1e-3       --weight-decay 1e-4       --ccc-weight 1.0       --mse-weight 0.0       --num-workers "$NUM_WORKERS"       --seed 13       --device cuda       "$@" 2>&1 | tee "$LOG_DIR/$run_name.log"
  fi

  if [[ -f "$run_dir/ema_smoothing/ema_metrics_overall.csv" ]]; then
    echo "skip_existing_ema $run_name"
  else
    echo "ema $run_name"
    "$PYTHON_BIN" "$EVAL_SCRIPT"       --run-dir "$run_dir"       --alphas 1.0 0.9 0.7 0.5 0.3 0.2 0.1 0.05       --batch-size "$EVAL_BATCH_SIZE"       --num-workers "$NUM_WORKERS"       --device cuda 2>&1 | tee "$LOG_DIR/$run_name.ema.log"
  fi
}

# 1. Current architecture: shared TCN + shared prediction head, but role-specific loss.
run_train noxi_roleloss_sharedhead_true_temporal_mse005_delta002_seed13_bs32nw2   --manifest "$NOXI_MANIFEST"   --test-splits test_internal test_additional   --prediction-head-sharing shared   --expert-ccc-weight 1.0   --expert-mse-weight 0.0   --expert-delta-mse-weight 0.0   --novice-ccc-weight 1.0   --novice-mse-weight 0.05   --novice-delta-mse-weight 0.02

run_train noxij_roleloss_sharedhead_cal_mean001_std001_seed13_bs32nw2   --manifest "$NOXIJ_MANIFEST"   --test-splits test_internal   --prediction-head-sharing shared   --expert-ccc-weight 1.0   --expert-mean-calibration-weight 0.0   --expert-std-calibration-weight 0.0   --novice-ccc-weight 1.0   --novice-mean-calibration-weight 0.001   --novice-std-calibration-weight 0.001

# 2. New architecture: shared TCN + role-specific prediction heads, same role-specific loss.
run_train noxi_roleloss_rolehead_true_temporal_mse005_delta002_seed13_bs32nw2   --manifest "$NOXI_MANIFEST"   --test-splits test_internal test_additional   --prediction-head-sharing role_specific   --expert-ccc-weight 1.0   --expert-mse-weight 0.0   --expert-delta-mse-weight 0.0   --novice-ccc-weight 1.0   --novice-mse-weight 0.05   --novice-delta-mse-weight 0.02

run_train noxij_roleloss_rolehead_cal_mean001_std001_seed13_bs32nw2   --manifest "$NOXIJ_MANIFEST"   --test-splits test_internal   --prediction-head-sharing role_specific   --expert-ccc-weight 1.0   --expert-mean-calibration-weight 0.0   --expert-std-calibration-weight 0.0   --novice-ccc-weight 1.0   --novice-mean-calibration-weight 0.001   --novice-std-calibration-weight 0.001
