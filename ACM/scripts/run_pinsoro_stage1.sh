#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACM_DIR="${ACM_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-$(cd "${ACM_DIR}/../.." && pwd)/PinSoRo}"
PYTHON_BIN="${PYTHON_BIN:-python}"
WINDOW_SIZE="${WINDOW_SIZE:-300}"
WINDOW_STRIDE="${WINDOW_STRIDE:-75}"

FEATURE_SETS=(
    audio_egemaps audio_w2vbert2 text_xlm_roberta visual_swin visual_openface
    visual_openpose visual_clip visual_dino visual_videomae
)

"${PYTHON_BIN}" "${SCRIPT_DIR}/pinsoro_build_raw_manifests.py" --data-root "${DATA_ROOT}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/pinsoro_report_stream_coverage.py"

processed_manifests=()
window_manifests=()
for feature_set in "${FEATURE_SETS[@]}"; do
    aligned="${ACM_DIR}/outputs/pinsoro/manifests/${feature_set}_30hz.csv"
    normalized="${ACM_DIR}/outputs/pinsoro/manifests/${feature_set}_raw.csv"
    "${PYTHON_BIN}" "${SCRIPT_DIR}/pinsoro_prepare_feature_tensors_30hz.py" --feature-set "${feature_set}"
    "${PYTHON_BIN}" "${SCRIPT_DIR}/pinsoro_fit_apply_feature_transform.py" --input-manifest "${aligned}"
    processed_manifests+=("${normalized}")
    window_manifests+=(
        "${ACM_DIR}/outputs/pinsoro/windows/${feature_set}_w${WINDOW_SIZE}_s${WINDOW_STRIDE}_individual.csv"
        "${ACM_DIR}/outputs/pinsoro/windows/${feature_set}_w${WINDOW_SIZE}_s${WINDOW_STRIDE}_dyadic.csv"
    )
done

"${PYTHON_BIN}" "${SCRIPT_DIR}/pinsoro_build_shared_window_manifests.py" \
    --input-manifests "${processed_manifests[@]}" \
    --window-size "${WINDOW_SIZE}" --stride "${WINDOW_STRIDE}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/pinsoro_validate_preprocessing.py" \
    --manifests "${processed_manifests[@]}" \
    --window-manifests "${window_manifests[@]}"

echo "PinSoRo Stage 1 complete."
