#!/bin/bash

#SBATCH --job-name=gcm_train
#SBATCH --output=/home/mlut/mltac/.garbage/gcm_train.out      # Standard output and error log (%j is job ID)
#SBATCH --error=/home/mlut/mltac/.garbage/gcm_train.err       # Error log
#SBATCH --time=22:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=94GB
#SBATCH --gres=gpu:1
#SBATCH --exclude=cn19

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

ACM_DIR="${ACM_DIR:-$(pwd)/ACM}"
SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFESTS="${ACM_DIR}/outputs/manifests"
EXPERIMENTS="${ACM_DIR}/outputs/experiments"
DRY_RUN="${DRY_RUN:-0}"
AUTO_SELECT_TURN_BACKBONE="${AUTO_SELECT_TURN_BACKBONE:-1}"
ALLOW_PARTIAL_BACKBONE_GRID="${ALLOW_PARTIAL_BACKBONE_GRID:-0}"
TURN_BACKBONE_TRAINER_MODEL="${TURN_BACKBONE_TRAINER_MODEL:-}"
BEST_AUDIO_FEATURE_SET="${BEST_AUDIO_FEATURE_SET:-}"
BEST_TEXT_FEATURE_SET="${BEST_TEXT_FEATURE_SET:-}"
BEST_VISUAL_FEATURE_SET="${BEST_VISUAL_FEATURE_SET:-}"
FUSION_MODE="${FUSION_MODE:-gated}"
FUSION_CHANNELS="${FUSION_CHANNELS:-64}"
MODALITY_DROPOUT="${MODALITY_DROPOUT:-0.1}"
INCLUDE_CONCAT_BASELINE="${INCLUDE_CONCAT_BASELINE:-0}"
INCLUDE_EXPANDED_MULTIMODAL="${INCLUDE_EXPANDED_MULTIMODAL:-0}"
PYTHON_BIN="$(resolve_python_bin)"

ALL_STEPS=(1 2 3 4 5)
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
CCC_WEIGHT="${CCC_WEIGHT:-0.5}"
MSE_WEIGHT="${MSE_WEIGHT:-1.0}"
BATCH="${BATCH:-32}"

resolve_turn_backbone_selection() {
    if [ -n "${TURN_BACKBONE_TRAINER_MODEL}" ] \
        && [ -n "${BEST_AUDIO_FEATURE_SET}" ] \
        && [ -n "${BEST_TEXT_FEATURE_SET}" ] \
        && [ -n "${BEST_VISUAL_FEATURE_SET}" ]; then
        return 0
    fi

    if [ "${AUTO_SELECT_TURN_BACKBONE}" != "1" ]; then
        echo "Automatic backbone selection disabled, but one or more selection variables are missing." >&2
        echo "Required: TURN_BACKBONE_TRAINER_MODEL, BEST_AUDIO_FEATURE_SET, BEST_TEXT_FEATURE_SET, BEST_VISUAL_FEATURE_SET" >&2
        exit 1
    fi

    local resolver_cmd=(
        "${PYTHON_BIN}" "${SCRIPTS}/collect_results.py"
        --experiments-dir "${EXPERIMENTS}"
        --resolve-turn-backbone
        --selection-format env
    )
    if [ "${ALLOW_PARTIAL_BACKBONE_GRID}" = "1" ]; then
        resolver_cmd+=(--allow-partial-grid)
    fi

    while IFS='=' read -r key value; do
        case "${key}" in
            TURN_BACKBONE_TRAINER_MODEL|BEST_AUDIO_FEATURE_SET|BEST_TEXT_FEATURE_SET|BEST_VISUAL_FEATURE_SET)
                printf -v "${key}" '%s' "${value}"
                export "${key}"
                ;;
        esac
    done < <("${resolver_cmd[@]}")

    if [ -z "${TURN_BACKBONE_TRAINER_MODEL}" ] \
        || [ -z "${BEST_AUDIO_FEATURE_SET}" ] \
        || [ -z "${BEST_TEXT_FEATURE_SET}" ] \
        || [ -z "${BEST_VISUAL_FEATURE_SET}" ]; then
        echo "Failed to resolve winner backbone or representative feature sets." >&2
        exit 1
    fi
}

combo_name_from_parts() {
    local parts=("$@")
    local joined="${parts[0]}"
    local part
    for part in "${parts[@]:1}"; do
        joined+="__${part}"
    done
    echo "${joined}"
}

build_multimodal_manifest_if_needed() {
    local combo_name="$1"
    local source_suffix="$2"
    local output_suffix="$3"
    shift 3
    local feature_sets=("$@")
    local output_manifest="${MANIFESTS}/model_processed_manifest_${combo_name}_${output_suffix}.csv"
    local input_manifests=()
    local fs=""

    for fs in "${feature_sets[@]}"; do
        local input_manifest="${MANIFESTS}/model_processed_manifest_${fs}_${source_suffix}.csv"
        if [ ! -f "${input_manifest}" ] && [ "${source_suffix}" = "raw_windows" ]; then
            build_window_manifest_if_needed "${fs}" || true
        fi
        if [ ! -f "${input_manifest}" ]; then
            echo "  [skip] ${combo_name} — missing source interval manifest ${input_manifest}"
            return 1
        fi
        input_manifests+=("${input_manifest}")
    done

    if [ -f "${output_manifest}" ]; then
        return 0
    fi

    local cmd=(
        "${PYTHON_BIN}" "${SCRIPTS}/noxi_build_multimodal_turn_manifest.py"
        --input-manifests "${input_manifests[@]}"
        --output-manifest "${output_manifest}"
        --combo-name "${combo_name}"
    )
    if [ "${DRY_RUN}" = "1" ]; then
        echo "  [dry]  ${combo_name} multimodal manifest (${output_suffix})"
        printf '         '
        printf '%q ' "${cmd[@]}"
        echo ""
        echo ""
        return 0
    fi

    echo "  [prep] ${combo_name} multimodal manifest (${output_suffix})"
    "${cmd[@]}"
    echo ""
    return 0
}

build_window_manifest_if_needed() {
    local feature_set="$1"
    local input_manifest="${MANIFESTS}/model_processed_manifest_${feature_set}_raw.csv"
    local output_manifest="${MANIFESTS}/model_processed_manifest_${feature_set}_raw_windows.csv"

    if [ ! -f "${input_manifest}" ]; then
        echo "  [skip] ${feature_set} — missing source raw manifest ${input_manifest}"
        return 1
    fi
    if [ -f "${output_manifest}" ]; then
        return 0
    fi

    local cmd=(
        "${PYTHON_BIN}" "${SCRIPTS}/noxi_build_window_manifest.py"
        --input-manifest "${input_manifest}"
        --output-manifest "${output_manifest}"
    )
    if [ "${DRY_RUN}" = "1" ]; then
        echo "  [dry]  ${feature_set} window manifest"
        printf '         '
        printf '%q ' "${cmd[@]}"
        echo ""
        echo ""
        return 0
    fi

    echo "  [prep] ${feature_set} window manifest"
    "${cmd[@]}"
    echo ""
    return 0
}

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
    demographic
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
         "--ccc-weight ${CCC_WEIGHT} --mse-weight ${MSE_WEIGHT} --batch-size ${BATCH}"
}

attention_args_for_model() {
    local model_name="$1"
    if [ "${model_name}" = "attention" ]; then
        echo "--attention-context joint --attention-past-frames 1500 --exclude-current-frame"
    fi
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
            "${PYTHON_BIN}" "${SCRIPTS}/train_tcn_turns.py" \
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
            "${PYTHON_BIN}" "${SCRIPTS}/train_tcn_turns.py" \
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
            "${PYTHON_BIN}" "${SCRIPTS}/train_tcn_turns.py" \
            --manifest "${manifest}" \
            --model attention \
            --attention-context joint \
            --attention-past-frames 1500 \
            --exclude-current-frame \
            $(turn_common_args)
    done
fi

# =====================================================================
# Model Ladder Step 4: Winner-only multimodal fusion
# =====================================================================
if should_run_step 4; then
    resolve_turn_backbone_selection
    echo "=== Step 4: Winner-only multimodal fusion (${TURN_BACKBONE_TRAINER_MODEL}) ==="
    echo "AUDIO:  ${BEST_AUDIO_FEATURE_SET}"
    echo "TEXT:   ${BEST_TEXT_FEATURE_SET}"
    echo "VISUAL: ${BEST_VISUAL_FEATURE_SET}"

    FUSION_MODES=("${FUSION_MODE}")
    if [ "${INCLUDE_CONCAT_BASELINE}" = "1" ] && [ "${FUSION_MODE}" != "concat" ]; then
        FUSION_MODES=("concat" "${FUSION_MODE}")
    fi

    MULTIMODAL_COMBOS=(
        "${BEST_AUDIO_FEATURE_SET}|${BEST_TEXT_FEATURE_SET}"
        "${BEST_AUDIO_FEATURE_SET}|${BEST_VISUAL_FEATURE_SET}"
        "${BEST_TEXT_FEATURE_SET}|${BEST_VISUAL_FEATURE_SET}"
        "${BEST_AUDIO_FEATURE_SET}|${BEST_TEXT_FEATURE_SET}|${BEST_VISUAL_FEATURE_SET}"
    )

    for fusion in "${FUSION_MODES[@]}"; do
        for combo_spec in "${MULTIMODAL_COMBOS[@]}"; do
            IFS='|' read -r -a combo_parts <<< "${combo_spec}"
            combo_name="$(combo_name_from_parts "${combo_parts[@]}")"
            manifest="${MANIFESTS}/model_processed_manifest_${combo_name}_multimodal_turns.csv"
            if [ ! -f "${manifest}" ]; then
                build_multimodal_manifest_if_needed "${combo_name}" "raw_turns" "multimodal_turns" "${combo_parts[@]}" || continue
            fi
            run_if_needed "${combo_name}_turns_multimodal_${TURN_BACKBONE_TRAINER_MODEL}_${fusion}" \
                "${PYTHON_BIN}" "${SCRIPTS}/train_tcn_multimodal.py" \
                --manifest "${manifest}" \
                --backbone "${TURN_BACKBONE_TRAINER_MODEL}" \
                --fusion-mode "${fusion}" \
                --fusion-channels "${FUSION_CHANNELS}" \
                --modality-dropout "${MODALITY_DROPOUT}" \
                $(attention_args_for_model "${TURN_BACKBONE_TRAINER_MODEL}") \
                $(turn_common_args)
        done
    done

    # --- Expanded multimodal combos with behavioral features ---
    if [ "${INCLUDE_EXPANDED_MULTIMODAL}" = "1" ]; then
        echo ""
        echo "--- Step 4b: Expanded multimodal combos (behavioral features) ---"
        _EXPANDED_EXTRAS=(audio_egemaps visual_openface visual_openpose demographic)
        _BASE_SET=("${BEST_AUDIO_FEATURE_SET}" "${BEST_TEXT_FEATURE_SET}" "${BEST_VISUAL_FEATURE_SET}")

        _EXPANDED_COMBOS=()
        # Individual expansions: best_3 + one extra behavioral feature
        for extra in "${_EXPANDED_EXTRAS[@]}"; do
            _in_base=0
            for _b in "${_BASE_SET[@]}"; do [ "${_b}" = "${extra}" ] && _in_base=1 && break; done
            [ "${_in_base}" = "1" ] && continue
            _sorted_spec=$(printf '%s\n' "${_BASE_SET[@]}" "${extra}" | sort | tr '\n' '|' | sed 's/|$//')
            _EXPANDED_COMBOS+=("${_sorted_spec}")
        done

        # Full behavioral combo: best_3 + all non-duplicate extras
        _full_parts=("${_BASE_SET[@]}")
        for extra in "${_EXPANDED_EXTRAS[@]}"; do
            _in_base=0
            for _b in "${_full_parts[@]}"; do [ "${_b}" = "${extra}" ] && _in_base=1 && break; done
            [ "${_in_base}" = "1" ] || _full_parts+=("${extra}")
        done
        if [ "${#_full_parts[@]}" -gt 3 ]; then
            _sorted_spec=$(printf '%s\n' "${_full_parts[@]}" | sort | tr '\n' '|' | sed 's/|$//')
            _EXPANDED_COMBOS+=("${_sorted_spec}")
        fi

        for fusion in "${FUSION_MODES[@]}"; do
            for combo_spec in "${_EXPANDED_COMBOS[@]}"; do
                IFS='|' read -r -a combo_parts <<< "${combo_spec}"
                combo_name="$(combo_name_from_parts "${combo_parts[@]}")"
                manifest="${MANIFESTS}/model_processed_manifest_${combo_name}_multimodal_turns.csv"
                if [ ! -f "${manifest}" ]; then
                    build_multimodal_manifest_if_needed "${combo_name}" "raw_turns" "multimodal_turns" "${combo_parts[@]}" || continue
                fi
                run_if_needed "${combo_name}_turns_multimodal_${TURN_BACKBONE_TRAINER_MODEL}_${fusion}" \
                    "${PYTHON_BIN}" "${SCRIPTS}/train_tcn_multimodal.py" \
                    --manifest "${manifest}" \
                    --backbone "${TURN_BACKBONE_TRAINER_MODEL}" \
                    --fusion-mode "${fusion}" \
                    --fusion-channels "${FUSION_CHANNELS}" \
                    --modality-dropout "${MODALITY_DROPOUT}" \
                    $(attention_args_for_model "${TURN_BACKBONE_TRAINER_MODEL}") \
                    $(turn_common_args)
            done
        done
    fi
fi

# =====================================================================
# Model Ladder Step 5: Legacy fixed-window comparison on winner backbone
# =====================================================================
if should_run_step 5; then
    resolve_turn_backbone_selection
    echo "=== Step 5: Legacy fixed-window comparison (${TURN_BACKBONE_TRAINER_MODEL}) ==="
    for fs in "${FEATURE_SETS[@]}"; do
        manifest="${MANIFESTS}/model_processed_manifest_${fs}_raw_windows.csv"
        if [ ! -f "${manifest}" ]; then
            build_window_manifest_if_needed "${fs}" || continue
        fi
        run_if_needed "${fs}_windows_${TURN_BACKBONE_TRAINER_MODEL}" \
            "${PYTHON_BIN}" "${SCRIPTS}/train_tcn_turns.py" \
            --manifest "${manifest}" \
            --model "${TURN_BACKBONE_TRAINER_MODEL}" \
            $(attention_args_for_model "${TURN_BACKBONE_TRAINER_MODEL}") \
            $(turn_common_args)
    done

    echo "--- Winner-only multimodal legacy windows (${TURN_BACKBONE_TRAINER_MODEL}) ---"
    FUSION_MODES=("${FUSION_MODE}")
    if [ "${INCLUDE_CONCAT_BASELINE}" = "1" ] && [ "${FUSION_MODE}" != "concat" ]; then
        FUSION_MODES=("concat" "${FUSION_MODE}")
    fi

    MULTIMODAL_COMBOS=(
        "${BEST_AUDIO_FEATURE_SET}|${BEST_TEXT_FEATURE_SET}"
        "${BEST_AUDIO_FEATURE_SET}|${BEST_VISUAL_FEATURE_SET}"
        "${BEST_TEXT_FEATURE_SET}|${BEST_VISUAL_FEATURE_SET}"
        "${BEST_AUDIO_FEATURE_SET}|${BEST_TEXT_FEATURE_SET}|${BEST_VISUAL_FEATURE_SET}"
    )

    for fusion in "${FUSION_MODES[@]}"; do
        for combo_spec in "${MULTIMODAL_COMBOS[@]}"; do
            IFS='|' read -r -a combo_parts <<< "${combo_spec}"
            combo_name="$(combo_name_from_parts "${combo_parts[@]}")"
            manifest="${MANIFESTS}/model_processed_manifest_${combo_name}_multimodal_windows.csv"
            if [ ! -f "${manifest}" ]; then
                build_multimodal_manifest_if_needed "${combo_name}" "raw_windows" "multimodal_windows" "${combo_parts[@]}" || continue
            fi
            run_if_needed "${combo_name}_windows_multimodal_${TURN_BACKBONE_TRAINER_MODEL}_${fusion}" \
                "${PYTHON_BIN}" "${SCRIPTS}/train_tcn_multimodal.py" \
                --manifest "${manifest}" \
                --backbone "${TURN_BACKBONE_TRAINER_MODEL}" \
                --fusion-mode "${fusion}" \
                --fusion-channels "${FUSION_CHANNELS}" \
                --modality-dropout "${MODALITY_DROPOUT}" \
                $(attention_args_for_model "${TURN_BACKBONE_TRAINER_MODEL}") \
                $(turn_common_args)
        done
    done
fi

echo "=== Training complete ==="
echo ""
echo "Collect results with:"
echo "  python ${SCRIPTS}/collect_results.py"
