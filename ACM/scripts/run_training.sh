#!/usr/bin/env bash
# Training orchestration: all modality x model-type combinations.
#
# Runs the model ladder from the handoff for every registered feature set.
# All evaluation is on val_internal (test labels are held by the organizers).
#
# Usage (from mltac project root):
#   bash ACM/scripts/run_training.sh
#
# Prerequisites:
#   - run_preprocessing.sh has completed (manifests exist)
#
# Each run writes to ACM/outputs/experiments/<modality>_<model_type>/
# Collect results afterwards with:
#   python ACM/scripts/collect_results.py

set -euo pipefail

ACM_DIR="${ACM_DIR:-$(pwd)/ACM}"
SCRIPTS="${ACM_DIR}/scripts"
MANIFESTS="${ACM_DIR}/outputs/manifests"
EXPERIMENTS="${ACM_DIR}/outputs/experiments"
TRANSCRIPT_ROOT="${TRANSCRIPT_ROOT:-$(pwd)}"

# ---- Shared hyperparameters ----
EPOCHS=50
PATIENCE=12
LR=1e-3
WEIGHT_DECAY=1e-4
HIDDEN=64
LEVELS=4
KERNEL=5
DROPOUT=0.2
CCC_WEIGHT=0.5
WINDOW=500
STRIDE=125
BATCH=32

echo "=== ACM Training Pipeline ==="
echo "ACM_DIR:         ${ACM_DIR}"
echo "TRANSCRIPT_ROOT: ${TRANSCRIPT_ROOT}"
echo "EPOCHS:          ${EPOCHS}"
echo "PATIENCE:        ${PATIENCE}"
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
    if [ -f "${run_dir}/model_best.pt" ]; then
        echo "  [skip] ${run_name} — already trained"
        return 0
    fi
    echo "  [run]  ${run_name}"
    "$@" --run-name "${run_name}"
    echo ""
}

# ---- Common args shared by all TCN variants ----
common_args() {
    echo "--epochs ${EPOCHS} --patience ${PATIENCE} --lr ${LR}" \
         "--weight-decay ${WEIGHT_DECAY} --hidden-channels ${HIDDEN}" \
         "--levels ${LEVELS} --kernel-size ${KERNEL} --dropout ${DROPOUT}" \
         "--ccc-weight ${CCC_WEIGHT} --window-size ${WINDOW} --stride ${STRIDE}" \
         "--batch-size ${BATCH}"
}

# Turn-based variant: no --window-size / --stride (turns replace fixed windows).
turn_common_args() {
    echo "--epochs ${EPOCHS} --patience ${PATIENCE} --lr ${LR}" \
         "--weight-decay ${WEIGHT_DECAY} --hidden-channels ${HIDDEN}" \
         "--levels ${LEVELS} --kernel-size ${KERNEL} --dropout ${DROPOUT}" \
         "--ccc-weight ${CCC_WEIGHT} --batch-size ${BATCH}"
}

# =====================================================================
# Model Ladder Step 1: Simple role-level TCN
# =====================================================================
echo "=== Step 1: Simple TCN (role-level) ==="
for fs in "${FEATURE_SETS[@]}"; do
    manifest="${MANIFESTS}/model_processed_manifest_${fs}_raw.csv"
    [ ! -f "${manifest}" ] && echo "  [skip] ${fs} — no manifest" && continue
    run_if_needed "${fs}_simple_tcn" \
        python "${SCRIPTS}/train_tcn.py" \
        --manifest "${manifest}" \
        $(common_args)
done

# =====================================================================
# Model Ladder Step 2: Dyadic TCN — shared head
# =====================================================================
echo "=== Step 2: Dyadic TCN (shared head) ==="
for fs in "${FEATURE_SETS[@]}"; do
    manifest="${MANIFESTS}/model_processed_manifest_${fs}_raw_dyadic.csv"
    [ ! -f "${manifest}" ] && echo "  [skip] ${fs} — no dyadic manifest" && continue
    run_if_needed "${fs}_dyadic_shared" \
        python "${SCRIPTS}/train_tcn_dyadic.py" \
        --manifest "${manifest}" \
        --head-type shared \
        $(common_args)
done

# =====================================================================
# Model Ladder Step 3: Dyadic TCN — role-specific heads
# =====================================================================
echo "=== Step 3: Dyadic TCN (role-specific heads) ==="
for fs in "${FEATURE_SETS[@]}"; do
    manifest="${MANIFESTS}/model_processed_manifest_${fs}_raw_dyadic.csv"
    [ ! -f "${manifest}" ] && echo "  [skip] ${fs} — no dyadic manifest" && continue
    run_if_needed "${fs}_dyadic_role" \
        python "${SCRIPTS}/train_tcn_dyadic.py" \
        --manifest "${manifest}" \
        --head-type role_specific \
        $(common_args)
done

# =====================================================================
# Model Ladder Step 4: Partner-Lag TCN (3s + 30s lags)
# =====================================================================
echo "=== Step 4: Partner-Lag TCN ==="
for fs in "${FEATURE_SETS[@]}"; do
    manifest="${MANIFESTS}/model_processed_manifest_${fs}_raw_dyadic.csv"
    [ ! -f "${manifest}" ] && echo "  [skip] ${fs} — no dyadic manifest" && continue
    run_if_needed "${fs}_partner_lag" \
        python "${SCRIPTS}/train_tcn_partner_lag.py" \
        --manifest "${manifest}" \
        --partner-lags -75 -750 \
        $(common_args)
done

# =====================================================================
# Model Ladder Step 5: Gated Pooled TCN (30s partner pool)
# =====================================================================
echo "=== Step 5: Gated Pooled TCN ==="
for fs in "${FEATURE_SETS[@]}"; do
    manifest="${MANIFESTS}/model_processed_manifest_${fs}_raw_dyadic.csv"
    [ ! -f "${manifest}" ] && echo "  [skip] ${fs} — no dyadic manifest" && continue
    run_if_needed "${fs}_gated_pool" \
        python "${SCRIPTS}/train_tcn_gated_pool.py" \
        --manifest "${manifest}" \
        --partner-pool-frames 750 \
        --gate-type scalar \
        --save-gates \
        $(common_args)
done

# =====================================================================
# Model Ladder Step 6: Attention TCN (joint, 60s context)
# =====================================================================
echo "=== Step 6: Attention TCN ==="
for fs in "${FEATURE_SETS[@]}"; do
    manifest="${MANIFESTS}/model_processed_manifest_${fs}_raw_dyadic.csv"
    [ ! -f "${manifest}" ] && echo "  [skip] ${fs} — no dyadic manifest" && continue
    run_if_needed "${fs}_attention" \
        python "${SCRIPTS}/train_tcn_attention.py" \
        --manifest "${manifest}" \
        --attention-context joint \
        --attention-past-frames 1500 \
        --exclude-current-frame \
        --save-attention \
        $(common_args)
done

# =====================================================================
# Model Ladder Step 7: Turn-segmented Partner-Lag TCN
# =====================================================================
echo "=== Step 7: Turn-Segmented Partner-Lag TCN ==="
for fs in "${FEATURE_SETS[@]}"; do
    manifest="${MANIFESTS}/model_processed_manifest_${fs}_raw.csv"
    [ ! -f "${manifest}" ] && echo "  [skip] ${fs} — no role-level manifest" && continue
    run_if_needed "${fs}_turns_partner_lag" \
        python "${SCRIPTS}/train_tcn_turns.py" \
        --manifest "${manifest}" \
        --transcript-root "${TRANSCRIPT_ROOT}" \
        --model partner_lag \
        --partner-lags 0 \
        $(turn_common_args)
done

echo "=== Training complete ==="
echo ""
echo "Collect results with:"
echo "  python ${SCRIPTS}/collect_results.py"
