#!/usr/bin/env bash
set -euo pipefail

cd /workspace/ACM/ACM-clean
PYTHON_BIN="${PYTHON_BIN:-/tmp/torch-cu128/bin/python}"
GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/workspace/ACM/ACM-clean
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

ROOT="MoE/noxi_joint_settings/experiments_group_meanpool_calibration_sweep"
MANIFEST_ROOT="MoE/noxi_joint_settings/manifests/group_meanpool"
DATASET="noxij"
MANIFEST="$MANIFEST_ROOT/audio_w2vbert2__text_xlm_roberta__visual_videomae_${DATASET}_w500_s125_group_windows.csv"
mkdir -p "$ROOT/logs"

run_logged() {
  local log_path="$1"
  shift
  echo "$ $*"
  {
    echo
    echo "=== $(date '+%Y-%m-%d %H:%M:%S') ==="
    echo "$ $*"
    "$@"
  } 2>&1 | tee -a "$log_path"
}

ensure_manifest() {
  if [[ ! -f "$MANIFEST" ]]; then
    run_logged "$ROOT/logs/${DATASET}.build_manifest.log"       "$PYTHON_BIN" scripts/build_group_windows_from_multimodal_turn_manifest.py       --input-manifest MoE/noxi_joint_settings/manifests/audio_w2vbert2__text_xlm_roberta__visual_videomae_noxi_noxij_w500_s125_multimodal.csv       --output-manifest "$MANIFEST"       --datasets "$DATASET"
  fi
}

run_variant() {
  local tag="$1"
  local mean_weight="$2"
  local std_weight="$3"
  local jitter_weight="$4"
  local jitter_threshold="$5"
  local run_name="${DATASET}_audio_text_visual_w500_s125_postpred_linear_shared_encoder_${tag}_seed13"
  local run_dir="$ROOT/$run_name"
  if [[ ! -f "$run_dir/model_best.pt" ]]; then
    run_logged "$ROOT/logs/${run_name}.log"       "$PYTHON_BIN" scripts/train_mpii_group_meanpool_multimodal_calibration.py       --manifest "$MANIFEST"       --output-root "$ROOT"       --run-name "$run_name"       --train-split train_internal       --val-split val_internal       --test-splits test_internal       --fusion-mode gated       --fusion-channels 64       --modality-dropout 0.1       --hidden-channels 64       --levels 4       --kernel-size 5       --dropout 0.2       --encoder-sharing shared       --prediction-interaction-scale 0.1       --batch-size 32       --epochs 15       --patience 4       --min-epochs 5       --min-delta 0.001       --lr 1e-3       --weight-decay 1e-4       --mse-weight 0.0       --ccc-weight 1.0       --delta-mse-weight 0.0       --mean-calibration-weight "$mean_weight"       --std-calibration-weight "$std_weight"       --excess-jitter-weight "$jitter_weight"       --excess-jitter-threshold "$jitter_threshold"       --num-workers 2       --seed 13       --device cuda
  else
    echo "skip_existing $run_name"
  fi
  run_logged "$ROOT/logs/${run_name}.ema.log"     "$PYTHON_BIN" MoE/noxi_joint_settings/evaluate_group_ema_smoothing.py     --run-dir "$run_dir"     --alphas 1.0 0.9 0.7 0.5 0.3 0.2 0.1 0.05     --batch-size 32     --num-workers 2     --device cuda
}

ensure_manifest
run_variant cal_mean001_std001_jit000 0.01 0.01 0.0 0.01
run_variant cal_mean005_std005_jit000 0.05 0.05 0.0 0.01
run_variant cal_mean010_std005_jit000 0.10 0.05 0.0 0.01
run_variant cal_mean000_std000_jit001 0.0 0.0 0.01 0.01
run_variant cal_mean005_std005_jit001 0.05 0.05 0.01 0.01

echo "NOXI-J calibration/jitter sweep complete: $(date '+%Y-%m-%d %H:%M:%S')"
