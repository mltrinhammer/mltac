#!/usr/bin/env bash
# Full preprocessing pipeline for all modalities.
#
# Runs on HPC from the mltac project root:
#   bash ACM/scripts/run_preprocessing.sh
#
# Steps per modality:
#   1. Align features to 25 Hz target grid -> processed NPZ tensors
#   2. Fit train-only z-score normalizer   -> transformed "raw" tensors
#   3. Build dyadic (novice+expert) tensors -> dyadic tensors
#
# Prerequisites:
#   - Data directories (noxi/, noxij/) present in DATA_ROOT
#   - Manifests built via: python ACM/scripts/build_manifests_from_organizer.py --data-root $DATA_ROOT

set -euo pipefail

DATA_ROOT="${DATA_ROOT:-$(pwd)}"
ACM_DIR="${ACM_DIR:-${DATA_ROOT}/ACM}"
SCRIPTS="${ACM_DIR}/scripts"

echo "=== ACM Preprocessing Pipeline ==="
echo "DATA_ROOT: ${DATA_ROOT}"
echo "ACM_DIR:   ${ACM_DIR}"
echo ""

# ---- Step 0: Build manifests from organizer data (if not already done) ----
MANIFEST="${ACM_DIR}/outputs/model_raw_manifest_train_with_split.csv"
if [ ! -f "${MANIFEST}" ]; then
    echo "--- Step 0: Building manifests from organizer data ---"
    python "${SCRIPTS}/build_manifests_from_organizer.py" --data-root "${DATA_ROOT}"
    echo ""
fi

# ---- All feature sets registered in the pipeline ----
FEATURE_SETS=(
    audio_egemaps
    audio_w2vbert2
    text_xlm_roberta
    visual_swin
    visual_openface
    visual_openpose
    visual_clip
    visual_dino
    visual_videomae
)

# ---- Step 1: Align each feature set to 25 Hz ----
echo "=== Step 1: Aligning features to 25 Hz ==="
for fs in "${FEATURE_SETS[@]}"; do
    PROCESSED_MANIFEST="${ACM_DIR}/outputs/manifests/model_processed_manifest_${fs}_25hz.csv"
    if [ -f "${PROCESSED_MANIFEST}" ]; then
        echo "  [skip] ${fs} — already aligned (${PROCESSED_MANIFEST})"
        continue
    fi
    echo "  [run]  ${fs}"
    python "${SCRIPTS}/noxi_prepare_feature_tensors_25hz.py" --feature-set "${fs}"
    echo ""
done

# ---- Step 2: Fit normalizer and produce "raw" (z-score only) tensors ----
echo "=== Step 2: Fitting normalizers (train-only z-score) ==="
for fs in "${FEATURE_SETS[@]}"; do
    INPUT_MANIFEST="${ACM_DIR}/outputs/manifests/model_processed_manifest_${fs}_25hz.csv"
    OUTPUT_MANIFEST="${ACM_DIR}/outputs/manifests/model_processed_manifest_${fs}_raw.csv"
    if [ ! -f "${INPUT_MANIFEST}" ]; then
        echo "  [skip] ${fs} — no 25 Hz manifest found"
        continue
    fi
    if [ -f "${OUTPUT_MANIFEST}" ]; then
        echo "  [skip] ${fs} — already transformed (${OUTPUT_MANIFEST})"
        continue
    fi
    echo "  [run]  ${fs}"
    python "${SCRIPTS}/noxi_fit_apply_feature_transform.py" \
        --input-manifest "${INPUT_MANIFEST}" \
        --method raw
    echo ""
done

# ---- Step 3: Build dyadic tensors ----
echo "=== Step 3: Building dyadic tensors ==="
for fs in "${FEATURE_SETS[@]}"; do
    INPUT_MANIFEST="${ACM_DIR}/outputs/manifests/model_processed_manifest_${fs}_raw.csv"
    OUTPUT_MANIFEST="${ACM_DIR}/outputs/manifests/model_processed_manifest_${fs}_raw_dyadic.csv"
    if [ ! -f "${INPUT_MANIFEST}" ]; then
        echo "  [skip] ${fs} — no transformed manifest found"
        continue
    fi
    if [ -f "${OUTPUT_MANIFEST}" ]; then
        echo "  [skip] ${fs} — already built (${OUTPUT_MANIFEST})"
        continue
    fi
    echo "  [run]  ${fs}"
    python "${SCRIPTS}/noxi_build_dyadic_tensors.py" \
        --input-manifest "${INPUT_MANIFEST}"
    echo ""
done

echo "=== Preprocessing complete ==="
echo ""
echo "Role-level manifests (for simple TCN):"
for fs in "${FEATURE_SETS[@]}"; do
    m="${ACM_DIR}/outputs/manifests/model_processed_manifest_${fs}_raw.csv"
    [ -f "$m" ] && echo "  ${m}"
done
echo ""
echo "Dyadic manifests (for dyadic/interaction models):"
for fs in "${FEATURE_SETS[@]}"; do
    m="${ACM_DIR}/outputs/manifests/model_processed_manifest_${fs}_raw_dyadic.csv"
    [ -f "$m" ] && echo "  ${m}"
done
