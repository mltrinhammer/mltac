#!/usr/bin/env bash
set -euo pipefail

cd /workspace/ACM/ACM-clean

export PYTHONPATH=/workspace/ACM/ACM-clean
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

python3 MoE/noxi_joint_settings/run_group_ema_smoothing_queue.py \
  --python python3 \
  --num-workers 2 \
  --gpu 0

python3 MoE/pinsoro_noxi_settings/run_followup_experiment_queue.py \
  --python python3 \
  --num-workers 2 \
  --gpu 0 \
  --domain-scopes CC CR \
  --encoder-sharing shared separate \
  --apply-hmm
