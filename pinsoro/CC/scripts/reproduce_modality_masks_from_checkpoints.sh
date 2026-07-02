#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-2}"
OUT="${OUT:-$ROOT/outputs/reproduced_modality_masks}"
export PYTHONPATH="$ROOT:$ROOT/scripts/upstream:${PYTHONPATH:-}"
cd "$ROOT"
python3 scripts/upstream/evaluate_submitted_modality_masks.py \
  --task-run artifacts/runs/pinsoro_cc_task_shared_linear_shared_tcn_delta010_metadata_seed13 \
  --social-run artifacts/runs/pinsoro_cc_headarch_head_adapters_delta010_metadata_seed13 \
  --output-root "$OUT" \
  --device "$DEVICE" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS"
