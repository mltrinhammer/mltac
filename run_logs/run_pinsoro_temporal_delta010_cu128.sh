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

PINSORO_ROOT="MoE/experiments/pinsoro_temporal_delta010_shared_encoder_metadata_head"
PINSORO_MANIFEST_ROOT="MoE/moe_data/outputs/windows_w2400_s1200"
mkdir -p "$PINSORO_ROOT/logs"

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

run_pinsoro_temporal() {
  local domain="$1"
  local lower
  lower="$(printf '%s' "$domain" | tr 'A-Z' 'a-z')"
  local run_name="pinsoro_${lower}_audio_text_visual_concat_shared_encoder_linear_none_both_age_gender_role_metadata_head_delta010_seed13"
  local run_dir="$PINSORO_ROOT/$run_name"
  if [[ ! -f "$run_dir/model_best.pt" ]]; then
    run_logged "$PINSORO_ROOT/logs/${run_name}.log"       "$PYTHON_BIN" MoE/pinsoro_noxi_settings/train_person_interaction_fusion_temporal.py       --manifest       "$PINSORO_MANIFEST_ROOT/audio_w2vbert2_w2400_s1200_dyadic.csv"       "$PINSORO_MANIFEST_ROOT/text_xlm_roberta_w2400_s1200_dyadic.csv"       "$PINSORO_MANIFEST_ROOT/visual_videomae_w2400_s1200_dyadic.csv"       --domain-scope "$domain"       --output-root "$PINSORO_ROOT"       --run-name "$run_name"       --fusion-mode concat       --fusion-channels 64       --person-hidden-channels 64       --person-levels 5       --person-kernel-size 11       --dropout 0.2       --modality-dropout 0.1       --causal-tcn       --encoder-sharing shared       --interaction-mode linear       --interaction-hidden-channels 32       --interaction-kernel-size 5       --interaction-scale 0.1       --metadata MoE/moe_data/outputs/participant_metadata.csv       --metadata-mode age_gender_role       --metadata-embedding-dim 16       --metadata-dropout 0.2       --soft-label-mode none       --active-heads task social       --temporal-delta-weight 0.10       --batch-size 32       --num-workers 2       --epochs 30       --patience 6       --min-epochs 5       --min-delta 0.001       --lr 1e-3       --weight-decay 1e-4       --seed 13       --device cuda
  else
    echo "skip_existing $run_name"
  fi
  run_logged "$PINSORO_ROOT/logs/${run_name}.hmm.log"     "$PYTHON_BIN" MoE/pinsoro_noxi_settings/apply_person_interaction_hmm.py     --run-dir "$run_dir"     --manifest     "$PINSORO_MANIFEST_ROOT/audio_w2vbert2_w2400_s1200_dyadic.csv"     "$PINSORO_MANIFEST_ROOT/text_xlm_roberta_w2400_s1200_dyadic.csv"     "$PINSORO_MANIFEST_ROOT/visual_videomae_w2400_s1200_dyadic.csv"     --output-dir "$run_dir/hmm_smoothing"     --domain "$domain"     --write-test
}

run_pinsoro_temporal CC
run_pinsoro_temporal CR

echo "pinsoro temporal delta010 queue complete: $(date '+%Y-%m-%d %H:%M:%S')"
