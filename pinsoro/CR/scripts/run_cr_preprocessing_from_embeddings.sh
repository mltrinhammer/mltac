#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/PinSoRo}"
PYTHON_BIN="${PYTHON_BIN:-python}"
WINDOW_SIZE="${WINDOW_SIZE:-2400}"
WINDOW_STRIDE="${WINDOW_STRIDE:-1200}"

FEATURE_SETS=(
    audio_w2vbert2
    text_xlm_roberta
    visual_videomae
)

RAW_MANIFEST="${PROJECT_ROOT}/outputs/pinsoro/raw_manifest.csv"
RAW_STREAM_MANIFEST="${PROJECT_ROOT}/outputs/pinsoro/raw_stream_manifest.csv"
MANIFEST_DIR="${PROJECT_ROOT}/MoE/moe_data/outputs/manifests"
WINDOW_DIR="${PROJECT_ROOT}/MoE/moe_data/outputs/windows_w${WINDOW_SIZE}_s${WINDOW_STRIDE}"
PROCESSED_ROOT="${PROJECT_ROOT}/MoE/moe_data/processed/domain_norm"
TRANSFORM_ROOT="${PROJECT_ROOT}/MoE/moe_data/outputs/domain_transform"

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "WINDOW_SIZE=${WINDOW_SIZE}"
echo "WINDOW_STRIDE=${WINDOW_STRIDE}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/pinsoro_build_raw_manifests.py" \
    --data-root "${DATA_ROOT}" \
    --cache-root "${PROJECT_ROOT}/cache/pinsoro" \
    --out-dir "${PROJECT_ROOT}/outputs/pinsoro"

"${PYTHON_BIN}" "${SCRIPT_DIR}/pinsoro_report_stream_coverage.py" \
    --manifest "${RAW_STREAM_MANIFEST}" \
    --output "${PROJECT_ROOT}/outputs/pinsoro/validation/stream_coverage.csv"

for feature_set in "${FEATURE_SETS[@]}"; do
    aligned_manifest="${PROJECT_ROOT}/outputs/pinsoro/manifests/${feature_set}_25hz.csv"
    normalized_manifest="${MANIFEST_DIR}/${feature_set}_domain_normalized.csv"

    "${PYTHON_BIN}" "${SCRIPT_DIR}/pinsoro_prepare_feature_tensors_25hz.py" \
        --feature-set "${feature_set}" \
        --cache-root "${PROJECT_ROOT}/cache/pinsoro" \
        --manifest "${RAW_MANIFEST}" \
        --streams "${RAW_STREAM_MANIFEST}" \
        --out-root "${PROJECT_ROOT}/processed/pinsoro/${feature_set}_25hz" \
        --output-manifest "${aligned_manifest}" \
        --status-out "${PROJECT_ROOT}/outputs/pinsoro/manifests/${feature_set}_25hz_status.csv"

    "${PYTHON_BIN}" "${SCRIPT_DIR}/pinsoro_fit_apply_domain_feature_transform.py" \
        --input-manifest "${aligned_manifest}" \
        --out-root "${PROCESSED_ROOT}/${feature_set}" \
        --output-manifest "${normalized_manifest}" \
        --transform-dir "${TRANSFORM_ROOT}/${feature_set}" \
        --domains CC CR \
        --force

    "${PYTHON_BIN}" "${SCRIPT_DIR}/pinsoro_build_window_manifests.py" \
        --input-manifest "${normalized_manifest}" \
        --window-size "${WINDOW_SIZE}" \
        --stride "${WINDOW_STRIDE}" \
        --out-dir "${WINDOW_DIR}"
done

echo "CR preprocessing complete."
echo "Window manifests: ${WINDOW_DIR}"
