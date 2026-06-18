#!/usr/bin/env bash
set -euo pipefail

cd /workspace/ACM/ACM-clean

PYTHON_BIN=${PYTHON_BIN:-/workspace/venvs/torch-cu128/bin/python}
GPU=${GPU:-0}
BATCH_SIZE=${BATCH_SIZE:-32}
NUM_WORKERS=${NUM_WORKERS:-2}
OUTPUT_ROOT=${OUTPUT_ROOT:-/workspace/ACM/ACM-clean/MoE/experiments/pinsoro_cr_social_weight_oversample_delta010_metadata}
LOG_DIR="$OUTPUT_ROOT/logs"
mkdir -p "$LOG_DIR"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/workspace/ACM/ACM-clean
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

MANIFESTS=(
  MoE/moe_data/outputs/windows_w2400_s1200/audio_w2vbert2_w2400_s1200_dyadic.csv
  MoE/moe_data/outputs/windows_w2400_s1200/text_xlm_roberta_w2400_s1200_dyadic.csv
  MoE/moe_data/outputs/windows_w2400_s1200/visual_videomae_w2400_s1200_dyadic.csv
)
TRAIN_SCRIPT=MoE/pinsoro_noxi_settings/train_person_interaction_fusion_temporal.py
HMM_SCRIPT=MoE/pinsoro_noxi_settings/apply_person_interaction_hmm_active_heads.py

run_one() {
  local run_name="$1"
  local weighting="$2"
  local cap="$3"
  local oversample="$4"
  local run_dir="$OUTPUT_ROOT/$run_name"
  shift 4

  if [[ -f "$run_dir/model_best.pt" ]]; then
    echo "skip_existing_train $run_name"
  else
    echo "train $run_name"
    "$PYTHON_BIN" "$TRAIN_SCRIPT"       --manifest "${MANIFESTS[@]}"       --domain-scope CR       --output-root "$OUTPUT_ROOT"       --run-name "$run_name"       --fusion-mode concat       --fusion-channels 64       --person-hidden-channels 64       --person-levels 5       --person-kernel-size 11       --dropout 0.2       --modality-dropout 0.1       --causal-tcn       --encoder-sharing shared       --interaction-mode linear       --interaction-hidden-channels 32       --interaction-kernel-size 5       --interaction-scale 0.1       --metadata MoE/moe_data/outputs/participant_metadata.csv       --metadata-mode age_gender_role       --metadata-embedding-dim 16       --metadata-dropout 0.2       --soft-label-mode none       --active-heads social       --temporal-delta-weight 0.10       --cr-social-weighting "$weighting"       --cr-social-weight-cap "$cap"       --cr-social-class3-oversample "$oversample"       --batch-size "$BATCH_SIZE"       --num-workers "$NUM_WORKERS"       --epochs 30       --patience 6       --min-epochs 5       --min-delta 0.001       --lr 1e-3       --weight-decay 1e-4       --seed 13       --device cuda 2>&1 | tee "$LOG_DIR/$run_name.log"
  fi

  if [[ -f "$run_dir/hmm_smoothing_social/val_hmm_results.csv" ]]; then
    echo "skip_existing_hmm $run_name"
  else
    echo "hmm $run_name"
    "$PYTHON_BIN" "$HMM_SCRIPT"       --run-dir "$run_dir"       --manifest "${MANIFESTS[@]}"       --output-dir "$run_dir/hmm_smoothing_social"       --domain CR       --active-heads social       --write-test 2>&1 | tee "$LOG_DIR/$run_name.hmm.log"
  fi
}

run_one pinsoro_cr_social_delta010_metadata_cap5_seed13 capped_inverse 5.0 1
run_one pinsoro_cr_social_delta010_metadata_sqrt_inverse_seed13 sqrt_inverse 5.0 1
run_one pinsoro_cr_social_delta010_metadata_class3oversample3_unweighted_seed13 unweighted 5.0 3
run_one pinsoro_cr_social_delta010_metadata_class3oversample3_cap5_seed13 capped_inverse 5.0 3
