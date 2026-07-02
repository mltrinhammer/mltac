#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${OUT:-$ROOT/outputs/cc_task_hmm_from_cached_logits}"
export PYTHONPATH="$ROOT:$ROOT/scripts/upstream:${PYTHONPATH:-}"
cd "$ROOT"
RUN_DIR="artifacts/runs/pinsoro_cc_task_shared_linear_shared_tcn_delta010_metadata_seed13"
if [[ ! -f "$RUN_DIR/val_prediction_scores.csv.gz" || ! -f "$RUN_DIR/test_prediction_scores.csv.gz" ]]; then
  echo "Missing regenerated score exports in $RUN_DIR." >&2
  echo "Run first: OUT=artifacts/runs DEVICE=cuda bash scripts/train_cc_task_final.sh --eval-only" >&2
  exit 1
fi
python3 scripts/upstream/apply_person_interaction_hmm_active_heads.py \
  --run-dir "$RUN_DIR" \
  --manifest \
    artifacts/manifests/windows_w2400_s1200/audio_w2vbert2_w2400_s1200_dyadic.csv \
    artifacts/manifests/windows_w2400_s1200/text_xlm_roberta_w2400_s1200_dyadic.csv \
    artifacts/manifests/windows_w2400_s1200/visual_videomae_w2400_s1200_dyadic.csv \
  --output-dir "$OUT" \
  --domain CC \
  --active-heads task \
  --transition-mixes 1.0 \
  --transition-strengths 12.0 \
  --transition-alpha 1.0
