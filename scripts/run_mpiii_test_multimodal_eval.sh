#!/bin/bash

#SBATCH --job-name=gcm_mpiii_eval
#SBATCH --output=/home/mlut/mltac/.garbage/gcm_mpiii_eval.out
#SBATCH --error=/home/mlut/mltac/.garbage/gcm_mpiii_eval.err
#SBATCH --time=22:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=94GB
#SBATCH --gres=gpu:1
#SBATCH --exclude=cn19

set -euo pipefail

module load Anaconda3

source activate sync-opentslm


resolve_python_bin() {
    if [ -n "${PYTHON_BIN:-}" ] && command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
        echo "${PYTHON_BIN}"
        return 0
    fi
    if command -v python >/dev/null 2>&1; then
        echo python
        return 0
    fi
    if command -v python3 >/dev/null 2>&1; then
        echo python3
        return 0
    fi
    if command -v py.exe >/dev/null 2>&1; then
        echo py.exe
        return 0
    fi
    if command -v py >/dev/null 2>&1; then
        echo py
        return 0
    fi
    return 1
}

DATA_ROOT="${DATA_ROOT:-$(pwd)}"
ACM_DIR="${ACM_DIR:-${DATA_ROOT}/ACM}"
SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_ROOT="${EVAL_ROOT:-${ACM_DIR}/outputs/mpiii_eval}"
MANIFESTS="${EVAL_ROOT}/manifests"
PYTHON_BIN="$(resolve_python_bin)"

DATASET_NAME="${DATASET_NAME:-mpiigroupinteraction}"
TEST_SPLIT="${TEST_SPLIT:-test}"
WINDOW_SIZE="${WINDOW_SIZE:-500}"
WINDOW_STRIDE="${WINDOW_STRIDE:-125}"

# ---------------------------------------------------------------------------
# Checkpoint & combo detection
# ---------------------------------------------------------------------------
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
if [ -z "${CHECKPOINT_PATH}" ]; then
    _candidate="${ACM_DIR}/outputs/experiments/audio_w2vbert2__text_xlm_roberta__visual_videomae_turns_multimodal_dyadic_shared_gated/model_best.pt"
    if [ -f "${_candidate}" ]; then
        CHECKPOINT_PATH="${_candidate}"
        echo "Auto-detected checkpoint: ${CHECKPOINT_PATH}"
    else
        echo "CHECKPOINT_PATH must point to a trained multimodal model_best.pt checkpoint." >&2
        echo "Set it via:  CHECKPOINT_PATH=/path/to/model_best.pt sbatch ..." >&2
        exit 1
    fi
fi

# Auto-detect COMBO_NAME from checkpoint when not explicitly set.
if [ -z "${COMBO_NAME:-}" ]; then
    _detected_combo=$("${PYTHON_BIN}" -c "
import torch, sys
ckpt = torch.load('${CHECKPOINT_PATH}', map_location='cpu', weights_only=False)
cn = ckpt.get('combo_name', '')
if cn:
    print(cn)
else:
    sys.exit(1)
" 2>/dev/null || echo "")
    if [ -n "${_detected_combo}" ]; then
        COMBO_NAME="${_detected_combo}"
        echo "Auto-detected COMBO_NAME from checkpoint: ${COMBO_NAME}"
    else
        COMBO_NAME="audio_w2vbert2__text_xlm_roberta__visual_videomae"
        echo "Could not detect combo from checkpoint, using default: ${COMBO_NAME}"
    fi
fi

# Parse COMBO_NAME into individual feature sets for the multimodal manifest.
# Combo names use "__" (double underscore) as separator between feature sets.
# NOTE: IFS is a character set, so IFS='__' would split on every single '_'.
# Use parameter expansion to split on the "__" string instead.
COMBO_FEATURE_SETS=()
_tmp="${COMBO_NAME}"
while [[ "${_tmp}" == *__* ]]; do
    COMBO_FEATURE_SETS+=("${_tmp%%__*}")
    _tmp="${_tmp#*__}"
done
COMBO_FEATURE_SETS+=("${_tmp}")

# Only preprocess the feature sets required by the checkpoint's combo.
FEATURE_SETS=("${COMBO_FEATURE_SETS[@]}")

RUN_NAME="${RUN_NAME:-mpiii_${COMBO_NAME}_windows_multimodal_test}"

# ---------------------------------------------------------------------------
# Normalizer paths (fitted on NoXi train; reused for MPIIG test)
# ---------------------------------------------------------------------------
NORMALIZER_AUDIO_EGEMAPS="${NORMALIZER_AUDIO_EGEMAPS:-${ACM_DIR}/outputs/transforms/audio_egemaps_raw/normalizer.npz}"
NORMALIZER_AUDIO_W2VBERT2="${NORMALIZER_AUDIO_W2VBERT2:-${ACM_DIR}/outputs/transforms/audio_w2vbert2_raw/normalizer.npz}"
NORMALIZER_TEXT_XLM_ROBERTA="${NORMALIZER_TEXT_XLM_ROBERTA:-${ACM_DIR}/outputs/transforms/text_xlm_roberta_raw/normalizer.npz}"
NORMALIZER_VISUAL_SWIN="${NORMALIZER_VISUAL_SWIN:-${ACM_DIR}/outputs/transforms/visual_swin_raw/normalizer.npz}"
NORMALIZER_VISUAL_OPENFACE="${NORMALIZER_VISUAL_OPENFACE:-${ACM_DIR}/outputs/transforms/visual_openface_raw/normalizer.npz}"
NORMALIZER_VISUAL_OPENPOSE="${NORMALIZER_VISUAL_OPENPOSE:-${ACM_DIR}/outputs/transforms/visual_openpose_raw/normalizer.npz}"
NORMALIZER_VISUAL_CLIP="${NORMALIZER_VISUAL_CLIP:-${ACM_DIR}/outputs/transforms/visual_clip_raw/normalizer.npz}"
NORMALIZER_VISUAL_DINO="${NORMALIZER_VISUAL_DINO:-${ACM_DIR}/outputs/transforms/visual_dino_raw/normalizer.npz}"
NORMALIZER_VISUAL_VIDEOMAE="${NORMALIZER_VISUAL_VIDEOMAE:-${ACM_DIR}/outputs/transforms/visual_videomae_raw/normalizer.npz}"
NORMALIZER_DEMOGRAPHIC="${NORMALIZER_DEMOGRAPHIC:-${ACM_DIR}/outputs/transforms/demographic_raw/normalizer.npz}"

normalizer_for_feature_set() {
    local fs="$1"
    case "${fs}" in
        audio_egemaps)     echo "${NORMALIZER_AUDIO_EGEMAPS}" ;;
        audio_w2vbert2)    echo "${NORMALIZER_AUDIO_W2VBERT2}" ;;
        text_xlm_roberta)  echo "${NORMALIZER_TEXT_XLM_ROBERTA}" ;;
        visual_swin)       echo "${NORMALIZER_VISUAL_SWIN}" ;;
        visual_openface)   echo "${NORMALIZER_VISUAL_OPENFACE}" ;;
        visual_openpose)   echo "${NORMALIZER_VISUAL_OPENPOSE}" ;;
        visual_clip)       echo "${NORMALIZER_VISUAL_CLIP}" ;;
        visual_dino)       echo "${NORMALIZER_VISUAL_DINO}" ;;
        visual_videomae)   echo "${NORMALIZER_VISUAL_VIDEOMAE}" ;;
        demographic)       echo "${NORMALIZER_DEMOGRAPHIC}" ;;
        *) echo "Unknown feature set: ${fs}" >&2; return 1 ;;
    esac
}

echo "=== MPIII Multimodal TEST Evaluation (All-Pairs, Sliding Windows) ==="
echo "DATA_ROOT:      ${DATA_ROOT}"
echo "ACM_DIR:        ${ACM_DIR}"
echo "EVAL_ROOT:      ${EVAL_ROOT}"
echo "DATASET_NAME:   ${DATASET_NAME}"
echo "TEST_SPLIT:     ${TEST_SPLIT}"
echo "CHECKPOINT:     ${CHECKPOINT_PATH}"
echo "COMBO_NAME:     ${COMBO_NAME}"
echo "COMBO_FEATURES: ${COMBO_FEATURE_SETS[*]}"
echo "WINDOW_SIZE:    ${WINDOW_SIZE}"
echo "WINDOW_STRIDE:  ${WINDOW_STRIDE}"
echo ""

mkdir -p "${MANIFESTS}"

# -----------------------------------------------------------------------
# Step 0: Build raw manifests (auto-discovers participant roles)
# -----------------------------------------------------------------------
RAW_MANIFEST="${EVAL_ROOT}/model_raw_manifest_train_with_split.csv"
RAW_STREAMS="${EVAL_ROOT}/model_raw_manifest_streams_train.csv"

if [ -f "${RAW_MANIFEST}" ] && [ -f "${RAW_STREAMS}" ]; then
    echo "--- Step 0: SKIP (raw manifests already exist) ---"
else
    echo "--- Step 0: Build raw manifests for MPIII test ---"
    "${PYTHON_BIN}" "${SCRIPTS}/build_manifests_from_organizer.py" \
        --data-root "${DATA_ROOT}" \
        --datasets "${DATASET_NAME}" \
        --out-dir "${EVAL_ROOT}" \
        --cache-root "${ACM_DIR}/cache"
    echo ""
fi

# Fail fast when no rows are produced (typically split-directory mismatch).
MANIFEST_ROW_COUNT=$("${PYTHON_BIN}" -c "
import csv
with open('${RAW_MANIFEST}', newline='') as f:
    print(sum(1 for _ in csv.DictReader(f)))
")
if [ "${MANIFEST_ROW_COUNT}" -eq 0 ]; then
    echo "No MPIII manifest rows were produced. Check available split directories under ${DATA_ROOT}/${DATASET_NAME}." >&2
    if [ -d "${DATA_ROOT}/${DATASET_NAME}" ]; then
        echo "Found directories:" >&2
        find "${DATA_ROOT}/${DATASET_NAME}" -maxdepth 1 -mindepth 1 -type d -printf '  %f\n' >&2 || true
    fi
    exit 1
fi

# Discover the roles that were found so we can pass them to --valid-roles.
VALID_ROLES=$("${PYTHON_BIN}" -c "
import csv, sys
roles = set()
with open('${EVAL_ROOT}/model_raw_manifest_train_with_split.csv', newline='') as f:
    for row in csv.DictReader(f):
        roles.add(row['role'])
print(' '.join(sorted(roles)))
")
echo "Discovered roles: ${VALID_ROLES}"
echo ""

# -----------------------------------------------------------------------
# Steps 1-3: Prepare 25 Hz tensors, transform, build all-pairs window manifests
# -----------------------------------------------------------------------
echo "--- Steps 1-3: Prepare, transform, and build all-pairs window manifests ---"
for fs in "${FEATURE_SETS[@]}"; do
    manifest_25hz="${MANIFESTS}/model_processed_manifest_${fs}_25hz.csv"
    manifest_raw="${MANIFESTS}/model_processed_manifest_${fs}_raw.csv"
    manifest_turns="${MANIFESTS}/model_processed_manifest_${fs}_raw_turns.csv"

    if [ -f "${manifest_25hz}" ]; then
        echo "  [${fs}] 25 Hz manifest exists, skipping alignment."
    elif [ "${fs}" = "demographic" ]; then
        # Demographic features use a dedicated script instead of stream alignment.
        _REF_MANIFEST=""
        for _ref_fs in "${FEATURE_SETS[@]}"; do
            [ "${_ref_fs}" = "demographic" ] && continue
            _ref="${MANIFESTS}/model_processed_manifest_${_ref_fs}_25hz.csv"
            if [ -f "${_ref}" ]; then
                _REF_MANIFEST="${_ref}"
                break
            fi
        done
        if [ -n "${_REF_MANIFEST}" ]; then
            echo "  [${fs}] Preparing demographic tensors ..."
            # shellcheck disable=SC2086
            "${PYTHON_BIN}" "${SCRIPTS}/prepare_demographic_tensors.py" \
                --reference-manifest "${_REF_MANIFEST}" \
                --data-root "${DATA_ROOT}" \
                --out-root "${ACM_DIR}/processed/mpiii_eval/${fs}_25hz" \
                --processed-manifest "${manifest_25hz}" \
                --valid-roles ${VALID_ROLES}
        else
            echo "  [${fs}] Skipping — no reference 25 Hz manifest available yet."
        fi
    else
        echo "  [${fs}] Aligning to 25 Hz ..."
        # shellcheck disable=SC2086
        "${PYTHON_BIN}" "${SCRIPTS}/noxi_prepare_feature_tensors_25hz.py" \
            --feature-set "${fs}" \
            --manifest "${RAW_MANIFEST}" \
            --streams "${RAW_STREAMS}" \
            --out-root "${ACM_DIR}/processed/mpiii_eval/${fs}_25hz" \
            --processed-manifest "${manifest_25hz}" \
            --status-out "${MANIFESTS}/feature_status_${fs}_25hz.csv" \
            --valid-roles ${VALID_ROLES}
    fi

    if [ -f "${manifest_raw}" ]; then
        echo "  [${fs}] Transformed manifest exists, skipping normalizer."
    else
        echo "  [${fs}] Applying NoXi normalizer ..."
        normalizer_path="$(normalizer_for_feature_set "${fs}")"
        if [ ! -f "${normalizer_path}" ]; then
            echo "Missing normalizer for ${fs}: ${normalizer_path}" >&2
            exit 1
        fi
        "${PYTHON_BIN}" "${SCRIPTS}/noxi_fit_apply_feature_transform.py" \
            --input-manifest "${manifest_25hz}" \
            --method raw \
            --normalizer-path "${normalizer_path}" \
            --out-root "${ACM_DIR}/processed/transformed/mpiii_eval/${fs}_raw" \
            --output-manifest "${manifest_raw}" \
            --transform-dir "${EVAL_ROOT}/transforms/${fs}_raw"
    fi

    if [ -f "${manifest_turns}" ]; then
        echo "  [${fs}] Window manifest exists, skipping all-pairs build."
    else
        echo "  [${fs}] Building all-pairs window manifest ..."
        "${PYTHON_BIN}" "${SCRIPTS}/build_allpairs_window_manifest.py" \
            --input-manifest "${manifest_raw}" \
            --output-manifest "${manifest_turns}" \
            --window-size "${WINDOW_SIZE}" \
            --stride "${WINDOW_STRIDE}"
    fi
    echo ""
done

# -----------------------------------------------------------------------
# Step 4: Build multimodal manifest (dynamically from COMBO_NAME)
# -----------------------------------------------------------------------
MULTIMODAL_MANIFEST="${MANIFESTS}/model_processed_manifest_${COMBO_NAME}_multimodal_turns.csv"
if [ -f "${MULTIMODAL_MANIFEST}" ]; then
    echo "--- Step 4: SKIP (multimodal manifest already exists) ---"
else
    echo "--- Step 4: Build multimodal manifest for combo: ${COMBO_NAME} ---"
    # Build --input-manifests list dynamically from the combo's feature sets.
    INPUT_MANIFESTS=()
    for fs in "${COMBO_FEATURE_SETS[@]}"; do
        _manifest="${MANIFESTS}/model_processed_manifest_${fs}_raw_turns.csv"
        if [ ! -f "${_manifest}" ]; then
            echo "Missing window manifest for combo feature ${fs}: ${_manifest}" >&2
            echo "Make sure this feature set is included in FEATURE_SETS." >&2
            exit 1
        fi
        INPUT_MANIFESTS+=("${_manifest}")
    done
    "${PYTHON_BIN}" "${SCRIPTS}/noxi_build_multimodal_turn_manifest.py" \
        --input-manifests "${INPUT_MANIFESTS[@]}" \
        --output-manifest "${MULTIMODAL_MANIFEST}" \
        --combo-name "${COMBO_NAME}"
fi
echo ""

# -----------------------------------------------------------------------
# Step 5: Run inference with all-pairs aggregation
# -----------------------------------------------------------------------
echo "--- Step 5: Run multimodal checkpoint inference (aggregate-pairs) ---"
"${PYTHON_BIN}" "${SCRIPTS}/infer_tcn_multimodal.py" \
    --manifest "${MULTIMODAL_MANIFEST}" \
    --checkpoint "${CHECKPOINT_PATH}" \
    --test-split "${TEST_SPLIT}" \
    --run-name "${RUN_NAME}" \
    --save-gates \
    --aggregate-pairs
echo ""

echo "=== MPIII test evaluation complete ==="
echo "Run outputs:"
echo "  ${ACM_DIR}/outputs/experiments/${RUN_NAME}"
