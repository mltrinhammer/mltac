#!/usr/bin/env bash
set -euo pipefail

cd /workspace/ACM/ACM-clean

PYTHON_BIN=${PYTHON_BIN:-/workspace/venvs/torch-cu128/bin/python}
GPU=${GPU:-0}
BATCH_SIZE=${BATCH_SIZE:-8}
NUM_WORKERS=${NUM_WORKERS:-0}
OUTPUT_ROOT=${OUTPUT_ROOT:-/workspace/ACM/ACM-clean/MoE/noxi_joint_settings/experiments_noxi_noxij_dyadic_roleheads_joint}
LOG_DIR="$OUTPUT_ROOT/logs"
mkdir -p "$LOG_DIR"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/workspace/ACM/ACM-clean
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

TRAIN_SCRIPT=/workspace/ACM/ACM-clean/scripts/train_tcn_multimodal.py
JOINT_MANIFEST=/workspace/ACM/ACM-clean/MoE/noxi_joint_settings/manifests/audio_w2vbert2__text_xlm_roberta__visual_videomae_noxi_noxij_w500_s125_multimodal.csv

run_train() {
  local run_name="$1"
  shift
  local run_dir="$OUTPUT_ROOT/$run_name"
  if [[ -f "$run_dir/model_best.pt" ]]; then
    echo "skip_existing_train $run_name"
    return 0
  fi
  echo "train $run_name"
  "$PYTHON_BIN" "$TRAIN_SCRIPT" \
    --manifest "$JOINT_MANIFEST" \
    --output-root "$OUTPUT_ROOT" \
    --run-name "$run_name" \
    --train-split train_internal \
    --val-split val_internal \
    --test-splits \
    --fusion-mode gated \
    --fusion-channels 64 \
    --modality-dropout 0.1 \
    --hidden-channels 64 \
    --levels 4 \
    --kernel-size 5 \
    --dropout 0.2 \
    --batch-size "$BATCH_SIZE" \
    --epochs 30 \
    --patience 6 \
    --min-epochs 5 \
    --min-delta 0.001 \
    --lr 1e-3 \
    --weight-decay 1e-4 \
    --ccc-weight 1.0 \
    --mse-weight 0.0 \
    --num-workers "$NUM_WORKERS" \
    --progress-every 1 \
    --seed 13 \
    --device cuda \
    "$@" 2>&1 | tee "$LOG_DIR/$run_name.log"
}

# Main interpretable test: old joint NOXI+NOXI-J data, but role-specific output heads.
run_train noxi_noxij_dyadic_roleheads_ccconly_seed13 \
  --backbone dyadic_role_heads

# Control: old-style joint NOXI+NOXI-J dyadic shared 2-channel head, same clean repo/data path.
run_train noxi_noxij_dyadic_sharedhead_ccconly_seed13 \
  --backbone dyadic_shared
