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
SCRIPTS="${ACM_DIR}/scripts"
EVAL_ROOT="${EVAL_ROOT:-${ACM_DIR}/outputs/mpiii_eval}"
MANIFESTS="${EVAL_ROOT}/manifests"
PYTHON_BIN="$(resolve_python_bin)"

DATASET_NAME="${DATASET_NAME:-mpiigroupinteraction}"
TEST_SPLIT="${TEST_SPLIT:-test}"
RUN_NAME="${RUN_NAME:-mpiii_w2vbert2_xlm_roberta_videomae_turns_multimodal_dyadic_shared_test}"
FEATURE_SETS=(
    audio_w2vbert2
    text_xlm_roberta
    visual_videomae
)
COMBO_NAME="${COMBO_NAME:-audio_w2vbert2__text_xlm_roberta__visual_videomae}"

CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
if [ -z "${CHECKPOINT_PATH}" ]; then
    # Auto-detect: look for the tri-modal NoXi checkpoint.
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

NORMALIZER_AUDIO_W2VBERT2="${NORMALIZER_AUDIO_W2VBERT2:-${ACM_DIR}/outputs/transforms/audio_w2vbert2_raw/normalizer.npz}"
NORMALIZER_TEXT_XLM_ROBERTA="${NORMALIZER_TEXT_XLM_ROBERTA:-${ACM_DIR}/outputs/transforms/text_xlm_roberta_raw/normalizer.npz}"
NORMALIZER_VISUAL_VIDEOMAE="${NORMALIZER_VISUAL_VIDEOMAE:-${ACM_DIR}/outputs/transforms/visual_videomae_raw/normalizer.npz}"

normalizer_for_feature_set() {
    local fs="$1"
    case "${fs}" in
        audio_w2vbert2) echo "${NORMALIZER_AUDIO_W2VBERT2}" ;;
        text_xlm_roberta) echo "${NORMALIZER_TEXT_XLM_ROBERTA}" ;;
        visual_videomae) echo "${NORMALIZER_VISUAL_VIDEOMAE}" ;;
        *) return 1 ;;
    esac
}

echo "=== MPIII Multimodal TEST Evaluation (All-Pairs) ==="
echo "DATA_ROOT:      ${DATA_ROOT}"
echo "ACM_DIR:        ${ACM_DIR}"
echo "EVAL_ROOT:      ${EVAL_ROOT}"
echo "DATASET_NAME:   ${DATASET_NAME}"
echo "TEST_SPLIT:     ${TEST_SPLIT}"
echo "CHECKPOINT:     ${CHECKPOINT_PATH}"
echo "COMBO_NAME:     ${COMBO_NAME}"
echo ""

mkdir -p "${MANIFESTS}"

# -----------------------------------------------------------------------
# Step 0: Build raw manifests (auto-discovers participant roles)
# -----------------------------------------------------------------------
echo "--- Step 0: Build raw manifests for MPIII test ---"
"${PYTHON_BIN}" "${SCRIPTS}/build_manifests_from_organizer.py" \
    --data-root "${DATA_ROOT}" \
    --datasets "${DATASET_NAME}" \
    --out-dir "${EVAL_ROOT}" \
    --cache-root "${ACM_DIR}/cache"
echo ""

# Fail fast when no rows are produced (typically split-directory mismatch).
RAW_MANIFEST="${EVAL_ROOT}/model_raw_manifest_train_with_split.csv"
RAW_STREAMS="${EVAL_ROOT}/model_raw_manifest_streams_train.csv"
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
# Steps 1-3: Prepare 25 Hz tensors, transform, build all-pairs turn manifests
# -----------------------------------------------------------------------
echo "--- Steps 1-3: Prepare, transform, and build all-pairs turn manifests ---"
for fs in "${FEATURE_SETS[@]}"; do
    manifest_25hz="${MANIFESTS}/model_processed_manifest_${fs}_25hz.csv"
    manifest_raw="${MANIFESTS}/model_processed_manifest_${fs}_raw.csv"
    manifest_turns="${MANIFESTS}/model_processed_manifest_${fs}_raw_turns.csv"

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

    echo "  [${fs}] Building all-pairs turn manifest ..."
    "${PYTHON_BIN}" "${SCRIPTS}/build_allpairs_turn_manifest.py" \
        --input-manifest "${manifest_raw}" \
        --transcript-root "${DATA_ROOT}" \
        --output-manifest "${manifest_turns}"
    echo ""
done

# -----------------------------------------------------------------------
# Step 4: Build multimodal turn manifest (works on composite session IDs)
# -----------------------------------------------------------------------
echo "--- Step 4: Build multimodal turn manifest ---"
MULTIMODAL_MANIFEST="${MANIFESTS}/model_processed_manifest_${COMBO_NAME}_multimodal_turns.csv"
"${PYTHON_BIN}" "${SCRIPTS}/noxi_build_multimodal_turn_manifest.py" \
    --input-manifests \
        "${MANIFESTS}/model_processed_manifest_audio_w2vbert2_raw_turns.csv" \
        "${MANIFESTS}/model_processed_manifest_text_xlm_roberta_raw_turns.csv" \
        "${MANIFESTS}/model_processed_manifest_visual_videomae_raw_turns.csv" \
    --output-manifest "${MULTIMODAL_MANIFEST}" \
    --combo-name "${COMBO_NAME}"
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
