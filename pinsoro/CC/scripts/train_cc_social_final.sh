#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEVICE="${DEVICE:-cuda}"
OUT="${OUT:-$ROOT/outputs/training}"
export PYTHONPATH="$ROOT:$ROOT/scripts/upstream:${PYTHONPATH:-}"
cd "$ROOT"
python3 scripts/upstream/train_person_interaction_fusion_temporal.py \
  --manifest \
    artifacts/manifests/windows_w2400_s1200/audio_w2vbert2_w2400_s1200_dyadic.csv \
    artifacts/manifests/windows_w2400_s1200/text_xlm_roberta_w2400_s1200_dyadic.csv \
    artifacts/manifests/windows_w2400_s1200/visual_videomae_w2400_s1200_dyadic.csv \
  --domain-scope CC \
  --output-root "$OUT" \
  --run-name pinsoro_cc_headarch_head_adapters_delta010_metadata_seed13 \
  --fusion-mode concat \
  --fusion-channels 64 \
  --person-hidden-channels 64 \
  --person-levels 5 \
  --person-kernel-size 11 \
  --dropout 0.2 \
  --modality-dropout 0.1 \
  --causal-tcn \
  --encoder-sharing shared \
  --head-architecture head_adapters \
  --head-adapter-levels 1 \
  --interaction-mode linear \
  --interaction-hidden-channels 32 \
  --interaction-kernel-size 5 \
  --interaction-scale 0.1 \
  --metadata-mode age_gender_role \
  --metadata-embedding-dim 16 \
  --metadata-dropout 0.2 \
  --temporal-delta-weight 0.1 \
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
  --device "$DEVICE" \
  "$@"
