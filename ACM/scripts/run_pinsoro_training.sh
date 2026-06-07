#!/bin/bash

# Run the 9-feature x 3-architecture PinSoRo unimodal ablation grid.

set -euo pipefail

ACM_DIR="${ACM_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
WINDOWS_DIR="${ACM_DIR}/outputs/pinsoro/windows"
OUTPUT_ROOT="${ACM_DIR}/outputs/pinsoro/experiments"
DRY_RUN="${DRY_RUN:-0}"
SEEDS="${SEEDS:-13}"
MODELS="${MODELS:-simple dyadic_shared attention}"
FEATURE_SETS="${FEATURE_SETS:-audio_egemaps audio_w2vbert2 text_xlm_roberta visual_swin visual_openface visual_openpose visual_clip visual_dino visual_videomae}"

EPOCHS="${EPOCHS:-50}"
PATIENCE="${PATIENCE:-12}"
MIN_EPOCHS="${MIN_EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-0}"
MAX_CACHED_TENSORS="${MAX_CACHED_TENSORS:-2}"

for seed in ${SEEDS}; do
    for feature_set in ${FEATURE_SETS}; do
        for model in ${MODELS}; do
            kind="dyadic"
            if [ "${model}" = "simple" ]; then
                kind="individual"
            fi
            manifest="${WINDOWS_DIR}/${feature_set}_w300_s75_${kind}.csv"
            run_name="pinsoro_${feature_set}_${model}_seed${seed}"
            cmd=(
                "${PYTHON_BIN}" "${ACM_DIR}/scripts/train_pinsoro_tcn.py"
                --manifest "${manifest}"
                --model "${model}"
                --output-root "${OUTPUT_ROOT}"
                --run-name "${run_name}"
                --seed "${seed}"
                --epochs "${EPOCHS}"
                --patience "${PATIENCE}"
                --min-epochs "${MIN_EPOCHS}"
                --batch-size "${BATCH_SIZE}"
                --num-workers "${NUM_WORKERS}"
                --max-cached-tensors "${MAX_CACHED_TENSORS}"
            )
            printf '%q ' "${cmd[@]}"
            printf '\n'
            if [ "${DRY_RUN}" != "1" ]; then
                "${cmd[@]}"
            fi
        done
    done
done
