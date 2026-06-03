#!/bin/bash

#SBATCH --job-name=gcm_train
#SBATCH --output=/home/mlut/mltac/.garbage/gcm_train.out      # Standard output and error log (%j is job ID)
#SBATCH --error=/home/mlut/mltac/.garbage/gcm_train.err       # Error log
#SBATCH --time=22:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=94GB
#SBATCH --gres=gpu:1
#SBATCH --exclude=cn19

#module load Python/3.11.3-GCCcore-12.3.0
module load Anaconda3
source activate sync-opentslm

# Usage (from mltac project root or inside the SLURM job):
#   bash ACM/scripts/run_training.sh
#   bash ACM/scripts/run_training.sh 1 2 3
#   bash ACM/scripts/run_training.sh --dry-run 2 3
#   bash ACM/scripts/submit_training_steps.sh 1 2 3
#   TRAINING_STEPS=1,2,3 bash ACM/scripts/run_training.sh
#   DRY_RUN=1 bash ACM/scripts/run_training.sh 2
#
# Steps are independent in this launcher: each ladder step reads manifests and
# writes its own experiment directory, so you can submit selected steps in
# separate SLURM jobs.

set -euo pipefail

ACM_DIR="${ACM_DIR:-$(pwd)/ACM}"
SCRIPTS="${ACM_DIR}/scripts"
MANIFESTS="${ACM_DIR}/outputs/manifests"
EXPERIMENTS="${ACM_DIR}/outputs/experiments"
DRY_RUN="${DRY_RUN:-0}"

ALL_STEPS=(1 2 3)
declare -A VALID_STEPS=()
for step in "${ALL_STEPS[@]}"; do
    VALID_STEPS["${step}"]=1
done

REQUESTED_STEPS=()
if [ "$#" -gt 0 ]; then
    for arg in "$@"; do
        case "${arg}" in
            --dry-run)
                DRY_RUN=1
                ;;
            --*)
                echo "Unknown option: ${arg}" >&2
                exit 1
                ;;
            *)
                REQUESTED_STEPS+=("${arg}")
                ;;
        esac
    done
elif [ -n "${TRAINING_STEPS:-}" ]; then
    IFS=', ' read -r -a REQUESTED_STEPS <<< "${TRAINING_STEPS}"
fi

declare -A SELECTED_STEP_MAP=()
SELECTED_STEPS=()
if [ "${#REQUESTED_STEPS[@]}" -eq 0 ]; then
    REQUESTED_STEPS=("${ALL_STEPS[@]}")
fi

for step in "${REQUESTED_STEPS[@]}"; do
    [ -z "${step}" ] && continue
    if [ -z "${VALID_STEPS[${step}]+x}" ]; then
        echo "Invalid training step: ${step}" >&2
        echo "Valid steps: ${ALL_STEPS[*]}" >&2
        exit 1
    fi
    if [ -z "${SELECTED_STEP_MAP[${step}]+x}" ]; then
        SELECTED_STEP_MAP["${step}"]=1
        SELECTED_STEPS+=("${step}")
    fi
done

should_run_step() {
    local step="$1"
    [ -n "${SELECTED_STEP_MAP[${step}]+x}" ]
}

# ---- Shared hyperparameters ----
EPOCHS=15
PATIENCE=3
MIN_EPOCHS=3
MIN_DELTA=0.001
LR=1e-3
WEIGHT_DECAY=1e-4
HIDDEN=64
LEVELS=4
KERNEL=5
DROPOUT=0.2
CCC_WEIGHT=0.5
BATCH=32

echo "=== ACM Training Pipeline ==="
echo "ACM_DIR:         ${ACM_DIR}"
echo "EPOCHS:          ${EPOCHS}"
echo "MIN_EPOCHS:      ${MIN_EPOCHS}"
echo "MIN_DELTA:       ${MIN_DELTA}"
echo "PATIENCE:        ${PATIENCE}"
echo "SELECTED_STEPS:  ${SELECTED_STEPS[*]}"
echo "DRY_RUN:         ${DRY_RUN}"
echo ""

# ---- Feature sets ----
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

# ---- Helper: skip if run directory already has a best model ----
run_if_needed() {
    local run_name="$1"
    shift
    local run_dir="${EXPERIMENTS}/${run_name}"
    local cmd=("$@" --run-name "${run_name}")
    if [ -f "${run_dir}/model_best.pt" ]; then
        echo "  [skip] ${run_name} — already trained"
        return 0
    fi
    if [ "${DRY_RUN}" = "1" ]; then
        echo "  [dry]  ${run_name}"
        printf '         '
        printf '%q ' "${cmd[@]}"
        echo ""
        echo ""
        return 0
    fi
    echo "  [run]  ${run_name}"
    "${cmd[@]}"
    echo ""
}

# Turn-based variant: no --window-size / --stride (turns replace fixed windows).
turn_common_args() {
    echo "--epochs ${EPOCHS} --patience ${PATIENCE} --lr ${LR}" \
         "--min-epochs ${MIN_EPOCHS} --min-delta ${MIN_DELTA}" \
         "--weight-decay ${WEIGHT_DECAY} --hidden-channels ${HIDDEN}" \
         "--levels ${LEVELS} --kernel-size ${KERNEL} --dropout ${DROPOUT}" \
         "--ccc-weight ${CCC_WEIGHT} --batch-size ${BATCH}"
}

# =====================================================================
# Model Ladder Step 1: Turn-level Simple TCN (independent per person)
# =====================================================================
if should_run_step 1; then
    echo "=== Step 1: Turn-level Simple TCN ==="
    for fs in "${FEATURE_SETS[@]}"; do
        manifest="${MANIFESTS}/model_processed_manifest_${fs}_raw_turns.csv"
        [ ! -f "${manifest}" ] && echo "  [skip] ${fs} — no paired turn manifest" && continue
        run_if_needed "${fs}_turns_simple_tcn" \
            python "${SCRIPTS}/train_tcn_turns.py" \
            --manifest "${manifest}" \
            --model simple \
            $(turn_common_args)
    done
fi

# =====================================================================
# Model Ladder Step 2: Turn-level Dyadic TCN — shared head
# =====================================================================
if should_run_step 2; then
    echo "=== Step 2: Turn-level Dyadic TCN (shared head) ==="
    for fs in "${FEATURE_SETS[@]}"; do
        manifest="${MANIFESTS}/model_processed_manifest_${fs}_raw_turns.csv"
        [ ! -f "${manifest}" ] && echo "  [skip] ${fs} — no paired turn manifest" && continue
        run_if_needed "${fs}_turns_dyadic_shared" \
            python "${SCRIPTS}/train_tcn_turns.py" \
            --manifest "${manifest}" \
            --model dyadic_shared \
            $(turn_common_args)
    done
fi

# =====================================================================
# Model Ladder Step 3: Turn-level Attention TCN (joint, 60s context)
# =====================================================================
if should_run_step 3; then
    echo "=== Step 3: Turn-level Attention TCN ==="
    for fs in "${FEATURE_SETS[@]}"; do
        manifest="${MANIFESTS}/model_processed_manifest_${fs}_raw_turns.csv"
        [ ! -f "${manifest}" ] && echo "  [skip] ${fs} — no paired turn manifest" && continue
        run_if_needed "${fs}_turns_attention" \
            python "${SCRIPTS}/train_tcn_turns.py" \
            --manifest "${manifest}" \
            --model attention \
            --attention-context joint \
            --attention-past-frames 1500 \
            --exclude-current-frame \
            $(turn_common_args)
    done
fi

echo "=== Training complete ==="
echo ""
echo "Collect results with:"
echo "  python ${SCRIPTS}/collect_results.py"
