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

NOXI_ROOT="MoE/noxi_joint_settings/experiments_group_meanpool_metadata_head"
NOXI_MANIFEST_ROOT="MoE/noxi_joint_settings/manifests/group_meanpool"
PINSORO_ROOT="MoE/experiments/pinsoro_metadata_head_shared_encoder"
PINSORO_MANIFEST_ROOT="MoE/moe_data/outputs/windows_w2400_s1200"
mkdir -p "$NOXI_ROOT/logs" "$PINSORO_ROOT/logs"

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

run_noxij() {
  local dataset="noxij"
  local metadata="MoE/noxi_j_data/outputs/metadata/role_metadata.csv"
  local manifest="$NOXI_MANIFEST_ROOT/audio_w2vbert2__text_xlm_roberta__visual_videomae_${dataset}_w500_s125_group_windows.csv"
  local run_name="${dataset}_audio_text_visual_w500_s125_postpred_linear_shared_encoder_metadata_head_seed13"
  local run_dir="$NOXI_ROOT/$run_name"
  if [[ ! -f "$manifest" ]]; then
    run_logged "$NOXI_ROOT/logs/${dataset}.build_manifest.log" \
      "$PYTHON_BIN" scripts/build_group_windows_from_multimodal_turn_manifest.py \
      --input-manifest MoE/noxi_joint_settings/manifests/audio_w2vbert2__text_xlm_roberta__visual_videomae_noxi_noxij_w500_s125_multimodal.csv \
      --output-manifest "$manifest" \
      --datasets "$dataset"
  fi
  if [[ ! -f "$run_dir/model_best.pt" ]]; then
    run_logged "$NOXI_ROOT/logs/${run_name}.log" \
      "$PYTHON_BIN" scripts/train_mpii_group_meanpool_multimodal.py \
      --manifest "$manifest" \
      --output-root "$NOXI_ROOT" \
      --run-name "$run_name" \
      --train-split train_internal \
      --val-split val_internal \
      --test-splits test_internal test_additional \
      --fusion-mode gated \
      --fusion-channels 64 \
      --modality-dropout 0.1 \
      --hidden-channels 64 \
      --levels 4 \
      --kernel-size 5 \
      --dropout 0.2 \
      --encoder-sharing shared \
      --prediction-interaction-scale 0.1 \
      --metadata "$metadata" \
      --metadata-mode age_gender_language \
      --metadata-embedding-dim 16 \
      --metadata-dropout 0.2 \
      --batch-size 32 \
      --epochs 30 \
      --patience 6 \
      --min-epochs 5 \
      --min-delta 0.001 \
      --lr 1e-3 \
      --weight-decay 1e-4 \
      --mse-weight 0.0 \
      --ccc-weight 1.0 \
      --num-workers 2 \
      --seed 13 \
      --device cuda
  else
    echo "skip_existing $run_name"
  fi
  run_logged "$NOXI_ROOT/logs/${run_name}.ema.log" \
    "$PYTHON_BIN" MoE/noxi_joint_settings/evaluate_group_ema_smoothing.py \
    --run-dir "$run_dir" \
    --alphas 1.0 0.9 0.7 0.5 0.3 0.2 0.1 0.05 \
    --batch-size 32 \
    --num-workers 2 \
    --device cuda
}

run_pinsoro() {
  local domain="$1"
  local lower
  lower="$(printf '%s' "$domain" | tr 'A-Z' 'a-z')"
  local run_name="pinsoro_${lower}_audio_text_visual_concat_shared_encoder_linear_none_both_age_gender_role_metadata_head_seed13"
  local run_dir="$PINSORO_ROOT/$run_name"
  if [[ ! -f "$run_dir/model_best.pt" ]]; then
    run_logged "$PINSORO_ROOT/logs/${run_name}.log" \
      "$PYTHON_BIN" MoE/pinsoro_noxi_settings/train_person_interaction_fusion.py \
      --manifest \
      "$PINSORO_MANIFEST_ROOT/audio_w2vbert2_w2400_s1200_dyadic.csv" \
      "$PINSORO_MANIFEST_ROOT/text_xlm_roberta_w2400_s1200_dyadic.csv" \
      "$PINSORO_MANIFEST_ROOT/visual_videomae_w2400_s1200_dyadic.csv" \
      --domain-scope "$domain" \
      --output-root "$PINSORO_ROOT" \
      --run-name "$run_name" \
      --fusion-mode concat \
      --fusion-channels 64 \
      --person-hidden-channels 64 \
      --person-levels 5 \
      --person-kernel-size 11 \
      --dropout 0.2 \
      --modality-dropout 0.1 \
      --causal-tcn \
      --encoder-sharing shared \
      --interaction-mode linear \
      --interaction-hidden-channels 32 \
      --interaction-kernel-size 5 \
      --interaction-scale 0.1 \
      --metadata MoE/moe_data/outputs/participant_metadata.csv \
      --metadata-mode age_gender_role \
      --metadata-embedding-dim 16 \
      --metadata-dropout 0.2 \
      --soft-label-mode none \
      --active-heads task social \
      --batch-size 32 \
      --num-workers 2 \
      --epochs 30 \
      --patience 6 \
      --min-epochs 5 \
      --min-delta 0.001 \
      --lr 1e-3 \
      --weight-decay 1e-4 \
      --seed 13 \
      --device cuda
  else
    echo "skip_existing $run_name"
  fi
  run_logged "$PINSORO_ROOT/logs/${run_name}.hmm.log" \
    "$PYTHON_BIN" MoE/pinsoro_noxi_settings/apply_person_interaction_hmm.py \
    --run-dir "$run_dir" \
    --manifest \
    "$PINSORO_MANIFEST_ROOT/audio_w2vbert2_w2400_s1200_dyadic.csv" \
    "$PINSORO_MANIFEST_ROOT/text_xlm_roberta_w2400_s1200_dyadic.csv" \
    "$PINSORO_MANIFEST_ROOT/visual_videomae_w2400_s1200_dyadic.csv" \
    --output-dir "$run_dir/hmm_smoothing" \
    --domain "$domain" \
    --write-test
}

run_noxij
run_pinsoro CC
run_pinsoro CR

echo "metadata-head resume queue complete: $(date '+%Y-%m-%d %H:%M:%S')"
