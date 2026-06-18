#!/usr/bin/env bash
set -euo pipefail

cd /workspace/ACM/ACM-clean

PYTHON_BIN=${PYTHON_BIN:-/workspace/venvs/torch-cu128/bin/python}
GPU=${GPU:-0}
BATCH_SIZE=${BATCH_SIZE:-32}
NUM_WORKERS=${NUM_WORKERS:-0}
OUTPUT_ROOT=${OUTPUT_ROOT:-/workspace/ACM/ACM-clean/MoE/noxi_joint_settings/experiments_noxij_dyadic_partner_ladder}
LOG_DIR="$OUTPUT_ROOT/logs"
mkdir -p "$LOG_DIR"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/workspace/ACM/ACM-clean
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

MANIFEST=/workspace/ACM/ACM-clean/MoE/noxi_joint_settings/manifests/audio_w2vbert2__text_xlm_roberta__visual_videomae_noxij_w500_s125_multimodal.csv
TRAIN_SCRIPT=/workspace/ACM/ACM-clean/scripts/train_tcn_multimodal.py

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
    --manifest "$MANIFEST" \
    --output-root "$OUTPUT_ROOT" \
    --run-name "$run_name" \
    --train-split train_internal \
    --val-split val_internal \
    --test-splits test_internal \
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
    --seed 13 \
    --device cuda \
    "$@" 2>&1 | tee "$LOG_DIR/$run_name.log"
}

# 1. Explicit dyadic input, shared dyadic encoder, role-specific heads. No attention.
run_train noxij_dyadic_roleheads_ccconly_seed13 \
  --backbone dyadic_role_heads

# 2. Shared role encoder, learned gated partner branch, role-specific heads.
run_train noxij_gated_partner_roleheads_ccconly_seed13 \
  --backbone gated_partner

# 3. Shared role encoder, joint self+partner temporal attention, role-specific heads.
run_train noxij_shared_attention_joint_roleheads_ccconly_seed13 \
  --backbone shared_attention \
  --attention-context joint \
  --attention-heads 4 \
  --attention-past-frames 1500
