#!/bin/bash

#SBATCH --job-name=gcm_dapa_mpiigi
#SBATCH --output=/home/mlut/mltac/.garbage/gcm_dapa_mpiigi.%j.out
#SBATCH --error=/home/mlut/mltac/.garbage/gcm_dapa_mpiigi.%j.err
#SBATCH --time=22:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=94GB
#SBATCH --gres=gpu:1
#SBATCH --exclude=cn19

# Usage (from mltac project root):
#   sbatch scripts/run_dapa_mpiigi.sh
#   bash scripts/run_dapa_mpiigi.sh           # interactive
#   DRY_RUN=1 bash scripts/run_dapa_mpiigi.sh # preview commands
#
# Environment overrides:
#   JOINT_TRAINING=1       Include NoXi data for joint training (default: 1)
#   DAPA_EPOCHS=40         Training epochs (default: 40)
#   DAPA_LR=5e-5           Learning rate (default: 5e-5)
#   DAPA_BATCH=32          Training batch size (default: 32)
#   DAPA_LSTM_HIDDEN=128   BiLSTM hidden size (default: 128)
#   NO_DOMAIN_PROMPTS=1    Disable domain prompts (default: 0)

set -euo pipefail

if type module >/dev/null 2>&1; then
    module load Python/3.11.3-GCCcore-12.3.0
    module load Anaconda3
fi
if type conda >/dev/null 2>&1 || [ -n "${CONDA_EXE:-}" ]; then
    source activate sync-opentslm
fi

resolve_python_bin() {
    if [ -n "${PYTHON_BIN:-}" ] && command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
        echo "${PYTHON_BIN}"
        return 0
    fi
    for candidate in python python3 py.exe py; do
        if command -v "${candidate}" >/dev/null 2>&1; then
            echo "${candidate}"
            return 0
        fi
    done
    return 1
}

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_ROOT="${DATA_ROOT:-$(pwd)}"
ACM_DIR="${ACM_DIR:-${DATA_ROOT}/ACM}"
SCRIPTS="${ACM_DIR}/../scripts"
MANIFESTS="${ACM_DIR}/outputs/manifests"
EXPERIMENTS="${ACM_DIR}/outputs/experiments"
PYTHON_BIN="$(resolve_python_bin)"
DRY_RUN="${DRY_RUN:-0}"

# ---------------------------------------------------------------------------
# DAPA-specific feature set: eGeMAPS + Swin + OpenFace + OpenPose
# (matches last year's DAPA winner minus Whisper)
# ---------------------------------------------------------------------------
DAPA_FEATURE_SETS=(audio_egemaps visual_swin visual_openface visual_openpose)
DAPA_COMBO_NAME="audio_egemaps__visual_openface__visual_openpose__visual_swin"

# ---------------------------------------------------------------------------
# MPIIGI-specific settings
# ---------------------------------------------------------------------------
DATASET_NAME="mpiigroupinteraction"
WINDOW_SIZE="${WINDOW_SIZE:-500}"
WINDOW_STRIDE="${WINDOW_STRIDE:-125}"

# MPIIGI val sessions used for validation (from organizer structure)
# The test set is held out by the organizers; val sessions serve as our dev set.
MPII_VAL_SESSIONS="${MPII_VAL_SESSIONS:-008 009 010 026 027 028}"

# ---------------------------------------------------------------------------
# Training hyperparameters (DAPA-aligned defaults)
# ---------------------------------------------------------------------------
JOINT_TRAINING="${JOINT_TRAINING:-1}"
DAPA_EPOCHS="${DAPA_EPOCHS:-40}"
DAPA_LR="${DAPA_LR:-5e-5}"
DAPA_BATCH="${DAPA_BATCH:-32}"
DAPA_VAL_BATCH="${DAPA_VAL_BATCH:-256}"
DAPA_LSTM_HIDDEN="${DAPA_LSTM_HIDDEN:-128}"
DAPA_LSTM_LAYERS="${DAPA_LSTM_LAYERS:-2}"
DAPA_TCN_HIDDEN="${DAPA_TCN_HIDDEN:-64}"
DAPA_TCN_LEVELS="${DAPA_TCN_LEVELS:-4}"
DAPA_WARMUP_STEPS="${DAPA_WARMUP_STEPS:-400}"
DAPA_EMA_DECAY="${DAPA_EMA_DECAY:-0.999}"
DAPA_SEED="${DAPA_SEED:-40}"
DAPA_DROPOUT="${DAPA_DROPOUT:-0.2}"
DAPA_LSTM_DROPOUT="${DAPA_LSTM_DROPOUT:-0.1}"
NO_DOMAIN_PROMPTS="${NO_DOMAIN_PROMPTS:-0}"

RUN_NAME="${RUN_NAME:-dapa_${DAPA_COMBO_NAME}_mpiigi}"

# ---------------------------------------------------------------------------
# Normalizer paths (fitted on NoXi train; reused for all datasets)
# ---------------------------------------------------------------------------
normalizer_for_feature_set() {
    local fs="$1"
    echo "${ACM_DIR}/outputs/transforms/${fs}_raw/normalizer.npz"
}

echo "=== DAPA MPIIGI Training Pipeline ==="
echo "DATA_ROOT:       ${DATA_ROOT}"
echo "ACM_DIR:         ${ACM_DIR}"
echo "FEATURE_SETS:    ${DAPA_FEATURE_SETS[*]}"
echo "COMBO_NAME:      ${DAPA_COMBO_NAME}"
echo "JOINT_TRAINING:  ${JOINT_TRAINING}"
echo "EPOCHS:          ${DAPA_EPOCHS}"
echo "LR:              ${DAPA_LR}"
echo "BATCH:           ${DAPA_BATCH}"
echo "LSTM_HIDDEN:     ${DAPA_LSTM_HIDDEN}"
echo "SEED:            ${DAPA_SEED}"
echo "RUN_NAME:        ${RUN_NAME}"
echo "DRY_RUN:         ${DRY_RUN}"
echo ""

mkdir -p "${MANIFESTS}"

# -----------------------------------------------------------------------
# Step 1: Prepare MPIIGI feature tensors (align to 25Hz + normalize)
# -----------------------------------------------------------------------
echo "=== Step 1: Prepare MPIIGI feature tensors ==="

# Build raw manifests if needed
MPII_EVAL_ROOT="${ACM_DIR}/outputs/mpiii_eval"
RAW_MANIFEST="${MPII_EVAL_ROOT}/model_raw_manifest_train_with_split.csv"
RAW_STREAMS="${MPII_EVAL_ROOT}/model_raw_manifest_streams_train.csv"

if [ ! -f "${RAW_MANIFEST}" ] || [ ! -f "${RAW_STREAMS}" ]; then
    echo "  Building raw manifests for ${DATASET_NAME} ..."
    if [ "${DRY_RUN}" = "1" ]; then
        echo "  [dry] build_manifests_from_organizer.py"
    else
        "${PYTHON_BIN}" "${SCRIPTS}/build_manifests_from_organizer.py" \
            --data-root "${DATA_ROOT}" \
            --datasets "${DATASET_NAME}" \
            --out-dir "${MPII_EVAL_ROOT}" \
            --cache-root "${ACM_DIR}/cache"
    fi
fi

# Discover available roles
VALID_ROLES=$("${PYTHON_BIN}" -c "
import csv
roles = set()
with open('${RAW_MANIFEST}', newline='') as f:
    for row in csv.DictReader(f):
        roles.add(row['role'])
print(' '.join(sorted(roles)))
" 2>/dev/null || echo "")
echo "  Discovered roles: ${VALID_ROLES}"

MPII_MANIFESTS="${MPII_EVAL_ROOT}/manifests"
mkdir -p "${MPII_MANIFESTS}"

for fs in "${DAPA_FEATURE_SETS[@]}"; do
    manifest_25hz="${MPII_MANIFESTS}/model_processed_manifest_${fs}_25hz.csv"
    manifest_raw="${MPII_MANIFESTS}/model_processed_manifest_${fs}_raw.csv"

    # Align to 25Hz
    if [ -f "${manifest_25hz}" ]; then
        echo "  [${fs}] 25Hz manifest exists, skipping."
    elif [ "${DRY_RUN}" = "1" ]; then
        echo "  [dry] [${fs}] align to 25Hz"
    else
        echo "  [${fs}] Aligning to 25Hz ..."
        # shellcheck disable=SC2086
        "${PYTHON_BIN}" "${SCRIPTS}/noxi_prepare_feature_tensors_25hz.py" \
            --feature-set "${fs}" \
            --manifest "${RAW_MANIFEST}" \
            --streams "${RAW_STREAMS}" \
            --out-root "${ACM_DIR}/processed/mpiii_eval/${fs}_25hz" \
            --processed-manifest "${manifest_25hz}" \
            --status-out "${MPII_MANIFESTS}/feature_status_${fs}_25hz.csv" \
            --valid-roles ${VALID_ROLES}
    fi

    # Apply normalizer (fitted on NoXi train data)
    if [ -f "${manifest_raw}" ]; then
        echo "  [${fs}] Normalized manifest exists, skipping."
    else
        normalizer_path="$(normalizer_for_feature_set "${fs}")"
        if [ ! -f "${normalizer_path}" ]; then
            echo "  [WARN] Missing normalizer for ${fs}: ${normalizer_path}" >&2
            echo "         Run the NoXi preprocessing pipeline first (run_preprocessing.sh)." >&2
            continue
        fi
        if [ "${DRY_RUN}" = "1" ]; then
            echo "  [dry] [${fs}] apply normalizer"
        else
            echo "  [${fs}] Applying NoXi normalizer ..."
            "${PYTHON_BIN}" "${SCRIPTS}/noxi_fit_apply_feature_transform.py" \
                --input-manifest "${manifest_25hz}" \
                --method raw \
                --normalizer-path "${normalizer_path}" \
                --out-root "${ACM_DIR}/processed/transformed/mpiii_eval/${fs}_raw" \
                --output-manifest "${manifest_raw}" \
                --transform-dir "${MPII_EVAL_ROOT}/transforms/${fs}_raw"
        fi
    fi
done
echo ""

# -----------------------------------------------------------------------
# Step 2: Build MPIIGI group-window manifest
# -----------------------------------------------------------------------
echo "=== Step 2: Build MPIIGI group-window manifest ==="

MPII_GROUP_MANIFEST="${MPII_MANIFESTS}/model_processed_manifest_${DAPA_COMBO_NAME}_group_windows.csv"

if [ -f "${MPII_GROUP_MANIFEST}" ]; then
    echo "  MPIIGI group manifest exists, skipping."
else
    # Collect per-modality normalized manifests
    INPUT_MANIFESTS=()
    for fs in "${DAPA_FEATURE_SETS[@]}"; do
        _m="${MPII_MANIFESTS}/model_processed_manifest_${fs}_raw.csv"
        if [ ! -f "${_m}" ]; then
            echo "  Missing manifest for ${fs}: ${_m}" >&2
            exit 1
        fi
        INPUT_MANIFESTS+=("${_m}")
    done

    if [ "${DRY_RUN}" = "1" ]; then
        echo "  [dry] build MPIIGI group windows"
    else
        echo "  Building MPIIGI group-window manifest ..."
        # shellcheck disable=SC2086
        "${PYTHON_BIN}" "${SCRIPTS}/build_mpii_group_window_manifest.py" \
            --input-manifests "${INPUT_MANIFESTS[@]}" \
            --output-manifest "${MPII_GROUP_MANIFEST}" \
            --combo-name "${DAPA_COMBO_NAME}" \
            --window-frames "${WINDOW_SIZE}" \
            --stride-frames "${WINDOW_STRIDE}" \
            --val-session-ids ${MPII_VAL_SESSIONS}
    fi
fi
echo ""

# -----------------------------------------------------------------------
# Step 3 (optional): Build NoXi group-window manifest for joint training
# -----------------------------------------------------------------------
NOXI_GROUP_MANIFEST=""
if [ "${JOINT_TRAINING}" = "1" ]; then
    echo "=== Step 3: Prepare NoXi data for joint training ==="

    # NoXi uses turn-based manifests; we need to convert to group-window format.
    # The per-modality raw manifests should already exist from the main pipeline.
    NOXI_INPUT_MANIFESTS=()
    NOXI_MISSING=0
    for fs in "${DAPA_FEATURE_SETS[@]}"; do
        _m="${MANIFESTS}/model_processed_manifest_${fs}_raw.csv"
        if [ ! -f "${_m}" ]; then
            echo "  [WARN] Missing NoXi manifest for ${fs}: ${_m}"
            NOXI_MISSING=1
        else
            NOXI_INPUT_MANIFESTS+=("${_m}")
        fi
    done

    if [ "${NOXI_MISSING}" = "1" ]; then
        echo "  Skipping NoXi joint training -- missing manifests."
        echo "  Run the NoXi preprocessing pipeline first (run_preprocessing.sh)."
        JOINT_TRAINING=0
    else
        NOXI_GROUP_MANIFEST="${MANIFESTS}/model_processed_manifest_${DAPA_COMBO_NAME}_noxi_group_windows.csv"
        if [ -f "${NOXI_GROUP_MANIFEST}" ]; then
            echo "  NoXi group manifest exists, skipping."
        elif [ "${DRY_RUN}" = "1" ]; then
            echo "  [dry] build NoXi group windows"
        else
            echo "  Building NoXi group-window manifest ..."
            "${PYTHON_BIN}" "${SCRIPTS}/build_mpii_group_window_manifest.py" \
                --input-manifests "${NOXI_INPUT_MANIFESTS[@]}" \
                --output-manifest "${NOXI_GROUP_MANIFEST}" \
                --combo-name "${DAPA_COMBO_NAME}" \
                --window-frames "${WINDOW_SIZE}" \
                --stride-frames "${WINDOW_STRIDE}"
        fi
    fi
    echo ""
fi

# -----------------------------------------------------------------------
# Step 4: Train DAPA model
# -----------------------------------------------------------------------
echo "=== Step 4: Train DAPA group model ==="

TRAIN_ARGS=(
    "${PYTHON_BIN}" "${SCRIPTS}/train_mpii_dapa_group.py"
    --manifest "${MPII_GROUP_MANIFEST}"
    --run-name "${RUN_NAME}"
    --epochs "${DAPA_EPOCHS}"
    --lr "${DAPA_LR}"
    --batch-size "${DAPA_BATCH}"
    --val-batch-size "${DAPA_VAL_BATCH}"
    --lstm-hidden "${DAPA_LSTM_HIDDEN}"
    --lstm-layers "${DAPA_LSTM_LAYERS}"
    --hidden-channels "${DAPA_TCN_HIDDEN}"
    --levels "${DAPA_TCN_LEVELS}"
    --dropout "${DAPA_DROPOUT}"
    --lstm-dropout "${DAPA_LSTM_DROPOUT}"
    --warmup-steps "${DAPA_WARMUP_STEPS}"
    --ema-decay "${DAPA_EMA_DECAY}"
    --seed "${DAPA_SEED}"
)

if [ "${NO_DOMAIN_PROMPTS}" = "1" ]; then
    TRAIN_ARGS+=(--no-domain-prompts)
else
    TRAIN_ARGS+=(--use-domain-prompts --n-domains 2)
fi

if [ "${JOINT_TRAINING}" = "1" ] && [ -n "${NOXI_GROUP_MANIFEST}" ]; then
    TRAIN_ARGS+=(--noxi-manifest "${NOXI_GROUP_MANIFEST}")
fi

# Skip if already trained
RUN_DIR="${EXPERIMENTS}/${RUN_NAME}"
if [ -f "${RUN_DIR}/model_best.pt" ]; then
    echo "  [skip] ${RUN_NAME} -- already trained"
elif [ "${DRY_RUN}" = "1" ]; then
    echo "  [dry] ${RUN_NAME}"
    printf '         '
    printf '%q ' "${TRAIN_ARGS[@]}"
    echo ""
else
    echo "  [run] ${RUN_NAME}"
    "${TRAIN_ARGS[@]}"
fi
echo ""

echo "=== DAPA MPIIGI pipeline complete ==="
echo "Run outputs: ${EXPERIMENTS}/${RUN_NAME}"
