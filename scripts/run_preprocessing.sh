#!/bin/bash

#SBATCH --job-name=gcm_preprocess
#SBATCH --output=/home/mlut/mltac/.garbage/gcm_preprocess.out      # Standard output and error log (%j is job ID)
#SBATCH --error=/home/mlut/mltac/.garbage/gcm_preprocess.err       # Error log
#SBATCH --time=22:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=94GB
#SBATCH --gres=gpu:1
#SBATCH --exclude=cn19

# Full preprocessing pipeline for all modalities.
#
# Runs on HPC from the mltac project root:
#   bash ACM/scripts/run_preprocessing.sh
#
# Steps per modality:
#   1. Align features to 25 Hz target grid -> processed NPZ tensors
#   2. Fit train-only z-score normalizer   -> transformed "raw" tensors
#   3. Build paired turn manifests          -> turn-index CSVs for training
#
# Prerequisites:
#   - Data directories (noxi/, noxij/) present in DATA_ROOT
#   - Manifests built via: python ACM/scripts/build_manifests_from_organizer.py --data-root $DATA_ROOT

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

DATA_ROOT="${DATA_ROOT:-$(pwd)}"
ACM_DIR="${ACM_DIR:-${DATA_ROOT}/ACM}"
SCRIPTS="${ACM_DIR}/scripts"
MANIFESTS="${ACM_DIR}/outputs/manifests"
EXPERIMENTS="${ACM_DIR}/outputs/experiments"

INCLUDE_MULTIMODAL_ABLATION="${INCLUDE_MULTIMODAL_ABLATION:-0}"
INCLUDE_WINDOW_ABLATION="${INCLUDE_WINDOW_ABLATION:-0}"
INCLUDE_EXPANDED_MULTIMODAL="${INCLUDE_EXPANDED_MULTIMODAL:-0}"
AUTO_SELECT_TURN_BACKBONE="${AUTO_SELECT_TURN_BACKBONE:-1}"
ALLOW_PARTIAL_BACKBONE_GRID="${ALLOW_PARTIAL_BACKBONE_GRID:-0}"
TURN_BACKBONE_TRAINER_MODEL="${TURN_BACKBONE_TRAINER_MODEL:-}"
BEST_AUDIO_FEATURE_SET="${BEST_AUDIO_FEATURE_SET:-}"
BEST_TEXT_FEATURE_SET="${BEST_TEXT_FEATURE_SET:-}"
BEST_VISUAL_FEATURE_SET="${BEST_VISUAL_FEATURE_SET:-}"
WINDOW_SIZE="${WINDOW_SIZE:-500}"
WINDOW_STRIDE="${WINDOW_STRIDE:-125}"
PYTHON_BIN="$(resolve_python_bin)"

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
        return 1
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
    done < <("${resolver_cmd[@]}") || return 1

    if [ -z "${TURN_BACKBONE_TRAINER_MODEL}" ] \
        || [ -z "${BEST_AUDIO_FEATURE_SET}" ] \
        || [ -z "${BEST_TEXT_FEATURE_SET}" ] \
        || [ -z "${BEST_VISUAL_FEATURE_SET}" ]; then
        echo "Failed to resolve winner backbone or representative feature sets." >&2
        return 1
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
    shift
    local feature_sets=("$@")
    local output_manifest="${MANIFESTS}/model_processed_manifest_${combo_name}_multimodal_turns.csv"
    local input_manifests=()
    local fs=""

    for fs in "${feature_sets[@]}"; do
        local input_manifest="${MANIFESTS}/model_processed_manifest_${fs}_raw_turns.csv"
        if [ ! -f "${input_manifest}" ]; then
            echo "  [skip] ${combo_name} — missing source turn manifest ${input_manifest}"
            return 0
        fi
        input_manifests+=("${input_manifest}")
    done

    if [ -f "${output_manifest}" ]; then
        echo "  [skip] ${combo_name} — multimodal manifest already built (${output_manifest})"
        return 0
    fi

    echo "  [run]  ${combo_name}"
    "${PYTHON_BIN}" "${SCRIPTS}/noxi_build_multimodal_turn_manifest.py" \
        --input-manifests "${input_manifests[@]}" \
        --output-manifest "${output_manifest}" \
        --combo-name "${combo_name}"
    echo ""
}

echo "=== ACM Preprocessing Pipeline ==="
echo "DATA_ROOT: ${DATA_ROOT}"
echo "ACM_DIR:   ${ACM_DIR}"
echo "INCLUDE_MULTIMODAL_ABLATION: ${INCLUDE_MULTIMODAL_ABLATION}"
echo "INCLUDE_WINDOW_ABLATION:     ${INCLUDE_WINDOW_ABLATION}"
echo "INCLUDE_EXPANDED_MULTIMODAL: ${INCLUDE_EXPANDED_MULTIMODAL}"
echo ""

# ---- Step 0: Build manifests from organizer data (if not already done) ----
MANIFEST="${ACM_DIR}/outputs/model_raw_manifest_train_with_split.csv"
if [ ! -f "${MANIFEST}" ]; then
    echo "--- Step 0: Building manifests from organizer data ---"
    "${PYTHON_BIN}" "${SCRIPTS}/build_manifests_from_organizer.py" --data-root "${DATA_ROOT}"
    echo ""
fi

# ---- All feature sets registered in the pipeline ----
# Stream-based feature sets (processed by Step 1 via the feature registry).
STREAM_FEATURE_SETS=(
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
# Full list including non-stream features (demographic).
# Steps 2-3 iterate over this; Step 1 uses STREAM_FEATURE_SETS only.
FEATURE_SETS=("${STREAM_FEATURE_SETS[@]}" demographic)

# ---- Step 1: Align each stream-based feature set to 25 Hz ----
echo "=== Step 1: Aligning features to 25 Hz ==="
for fs in "${STREAM_FEATURE_SETS[@]}"; do
    PROCESSED_MANIFEST="${MANIFESTS}/model_processed_manifest_${fs}_25hz.csv"
    if [ -f "${PROCESSED_MANIFEST}" ]; then
        echo "  [skip] ${fs} — already aligned (${PROCESSED_MANIFEST})"
        continue
    fi
    echo "  [run]  ${fs}"
    "${PYTHON_BIN}" "${SCRIPTS}/noxi_prepare_feature_tensors_25hz.py" --feature-set "${fs}"
    echo ""
done

# ---- Step 1b: Prepare demographic feature tensors (age, gender) ----
DEMO_25HZ="${MANIFESTS}/model_processed_manifest_demographic_25hz.csv"
if [ -f "${DEMO_25HZ}" ]; then
    echo "  [skip] demographic — already prepared (${DEMO_25HZ})"
else
    # Find any existing 25 Hz manifest as reference for aligned_len.
    _REF_MANIFEST=""
    for fs in "${STREAM_FEATURE_SETS[@]}"; do
        _candidate="${MANIFESTS}/model_processed_manifest_${fs}_25hz.csv"
        if [ -f "${_candidate}" ]; then
            _REF_MANIFEST="${_candidate}"
            break
        fi
    done
    if [ -z "${_REF_MANIFEST}" ]; then
        echo "  [skip] demographic — no reference 25 Hz manifest found (run Step 1 first)"
    else
        echo "  [run]  demographic (reference: $(basename "${_REF_MANIFEST}"))"
        "${PYTHON_BIN}" "${SCRIPTS}/prepare_demographic_tensors.py" \
            --reference-manifest "${_REF_MANIFEST}" \
            --data-root "${DATA_ROOT}" \
            --out-root "${ACM_DIR}/processed/demographic_25hz" \
            --processed-manifest "${DEMO_25HZ}"
        echo ""
    fi
fi

# ---- Step 2: Fit normalizer and produce "raw" (z-score only) tensors ----
echo "=== Step 2: Fitting normalizers (train-only z-score) ==="
for fs in "${FEATURE_SETS[@]}"; do
    INPUT_MANIFEST="${MANIFESTS}/model_processed_manifest_${fs}_25hz.csv"
    OUTPUT_MANIFEST="${MANIFESTS}/model_processed_manifest_${fs}_raw.csv"
    if [ ! -f "${INPUT_MANIFEST}" ]; then
        echo "  [skip] ${fs} — no 25 Hz manifest found"
        continue
    fi
    if [ -f "${OUTPUT_MANIFEST}" ]; then
        echo "  [skip] ${fs} — already transformed (${OUTPUT_MANIFEST})"
        continue
    fi
    echo "  [run]  ${fs}"
    "${PYTHON_BIN}" "${SCRIPTS}/noxi_fit_apply_feature_transform.py" \
        --input-manifest "${INPUT_MANIFEST}" \
        --method raw
    echo ""
done

# ---- Step 3: Build paired turn manifests ----
echo "=== Step 3: Building paired turn manifests ==="
for fs in "${FEATURE_SETS[@]}"; do
    INPUT_MANIFEST="${MANIFESTS}/model_processed_manifest_${fs}_raw.csv"
    OUTPUT_MANIFEST="${MANIFESTS}/model_processed_manifest_${fs}_raw_turns.csv"
    if [ ! -f "${INPUT_MANIFEST}" ]; then
        echo "  [skip] ${fs} — no transformed manifest found"
        continue
    fi
    if [ -f "${OUTPUT_MANIFEST}" ]; then
        echo "  [skip] ${fs} — paired turns already built (${OUTPUT_MANIFEST})"
        continue
    fi
    echo "  [run]  ${fs}"
    "${PYTHON_BIN}" "${SCRIPTS}/noxi_build_turn_manifest.py" \
        --input-manifest "${INPUT_MANIFEST}" \
        --transcript-root "${DATA_ROOT}" \
        --output-manifest "${OUTPUT_MANIFEST}"
    echo ""
done

if [ "${INCLUDE_MULTIMODAL_ABLATION}" = "1" ]; then
    if resolve_turn_backbone_selection; then
        echo "=== Step 4: Building multimodal turn manifests ==="
        echo "BACKBONE: ${TURN_BACKBONE_TRAINER_MODEL}"
        echo "AUDIO:    ${BEST_AUDIO_FEATURE_SET}"
        echo "TEXT:     ${BEST_TEXT_FEATURE_SET}"
        echo "VISUAL:   ${BEST_VISUAL_FEATURE_SET}"

        build_multimodal_manifest_if_needed \
            "$(combo_name_from_parts "${BEST_AUDIO_FEATURE_SET}" "${BEST_TEXT_FEATURE_SET}")" \
            "${BEST_AUDIO_FEATURE_SET}" "${BEST_TEXT_FEATURE_SET}"
        build_multimodal_manifest_if_needed \
            "$(combo_name_from_parts "${BEST_AUDIO_FEATURE_SET}" "${BEST_VISUAL_FEATURE_SET}")" \
            "${BEST_AUDIO_FEATURE_SET}" "${BEST_VISUAL_FEATURE_SET}"
        build_multimodal_manifest_if_needed \
            "$(combo_name_from_parts "${BEST_TEXT_FEATURE_SET}" "${BEST_VISUAL_FEATURE_SET}")" \
            "${BEST_TEXT_FEATURE_SET}" "${BEST_VISUAL_FEATURE_SET}"
        build_multimodal_manifest_if_needed \
            "$(combo_name_from_parts "${BEST_AUDIO_FEATURE_SET}" "${BEST_TEXT_FEATURE_SET}" "${BEST_VISUAL_FEATURE_SET}")" \
            "${BEST_AUDIO_FEATURE_SET}" "${BEST_TEXT_FEATURE_SET}" "${BEST_VISUAL_FEATURE_SET}"

        # --- Expanded combos: add behavioral features to best-3 trio ---
        if [ "${INCLUDE_EXPANDED_MULTIMODAL}" = "1" ]; then
            echo ""
            echo "--- Step 4b: Expanded multimodal combos (behavioral features) ---"
            _EXPANDED_EXTRAS=(audio_egemaps visual_openface visual_openpose demographic)
            _BASE_SET=("${BEST_AUDIO_FEATURE_SET}" "${BEST_TEXT_FEATURE_SET}" "${BEST_VISUAL_FEATURE_SET}")

            # Individual expansions: best_3 + one extra behavioral feature
            for extra in "${_EXPANDED_EXTRAS[@]}"; do
                _in_base=0
                for _b in "${_BASE_SET[@]}"; do [ "${_b}" = "${extra}" ] && _in_base=1 && break; done
                [ "${_in_base}" = "1" ] && continue
                mapfile -t _sorted_parts < <(printf '%s\n' "${_BASE_SET[@]}" "${extra}" | sort)
                build_multimodal_manifest_if_needed \
                    "$(combo_name_from_parts "${_sorted_parts[@]}")" \
                    "${_sorted_parts[@]}"
            done

            # Full behavioral combo: best_3 + all non-duplicate extras
            _full_parts=("${_BASE_SET[@]}")
            for extra in "${_EXPANDED_EXTRAS[@]}"; do
                _in_base=0
                for _b in "${_full_parts[@]}"; do [ "${_b}" = "${extra}" ] && _in_base=1 && break; done
                [ "${_in_base}" = "1" ] || _full_parts+=("${extra}")
            done
            if [ "${#_full_parts[@]}" -gt 3 ]; then
                mapfile -t _sorted_parts < <(printf '%s\n' "${_full_parts[@]}" | sort)
                build_multimodal_manifest_if_needed \
                    "$(combo_name_from_parts "${_sorted_parts[@]}")" \
                    "${_sorted_parts[@]}"
            fi
        fi
    else
        echo "=== Step 4: Skipping multimodal turn-manifest generation ==="
        echo "Winner backbone is not resolvable yet from current experiment outputs."
        echo "Training step 4 will build the required multimodal manifests lazily after unimodal results exist."
        echo ""
    fi
fi

if [ "${INCLUDE_WINDOW_ABLATION}" = "1" ]; then
    echo "=== Step 5: Building paired window manifests ==="
    echo "WINDOW_SIZE:   ${WINDOW_SIZE}"
    echo "WINDOW_STRIDE: ${WINDOW_STRIDE}"
    for fs in "${FEATURE_SETS[@]}"; do
        INPUT_MANIFEST="${MANIFESTS}/model_processed_manifest_${fs}_raw.csv"
        OUTPUT_MANIFEST="${MANIFESTS}/model_processed_manifest_${fs}_raw_windows.csv"
        if [ ! -f "${INPUT_MANIFEST}" ]; then
            echo "  [skip] ${fs} — no transformed manifest found"
            continue
        fi
        if [ -f "${OUTPUT_MANIFEST}" ]; then
            echo "  [skip] ${fs} — window manifest already built (${OUTPUT_MANIFEST})"
            continue
        fi
        echo "  [run]  ${fs}"
        "${PYTHON_BIN}" "${SCRIPTS}/noxi_build_window_manifest.py" \
            --input-manifest "${INPUT_MANIFEST}" \
            --output-manifest "${OUTPUT_MANIFEST}" \
            --window-size "${WINDOW_SIZE}" \
            --stride "${WINDOW_STRIDE}"
        echo ""
    done
fi

echo "=== Preprocessing complete ==="
echo ""
echo "Normalized role-level manifests (source tensors for paired turns):"
for fs in "${FEATURE_SETS[@]}"; do
    m="${MANIFESTS}/model_processed_manifest_${fs}_raw.csv"
    [ -f "$m" ] && echo "  ${m}"
done
echo ""
echo "Paired turn manifests (active turn-level training inputs):"
for fs in "${FEATURE_SETS[@]}"; do
    m="${MANIFESTS}/model_processed_manifest_${fs}_raw_turns.csv"
    [ -f "$m" ] && echo "  ${m}"
done
if [ "${INCLUDE_MULTIMODAL_ABLATION}" = "1" ]; then
    echo ""
    echo "Multimodal turn manifests:"
    for combo in \
        "$(combo_name_from_parts "${BEST_AUDIO_FEATURE_SET}" "${BEST_TEXT_FEATURE_SET}")" \
        "$(combo_name_from_parts "${BEST_AUDIO_FEATURE_SET}" "${BEST_VISUAL_FEATURE_SET}")" \
        "$(combo_name_from_parts "${BEST_TEXT_FEATURE_SET}" "${BEST_VISUAL_FEATURE_SET}")" \
        "$(combo_name_from_parts "${BEST_AUDIO_FEATURE_SET}" "${BEST_TEXT_FEATURE_SET}" "${BEST_VISUAL_FEATURE_SET}")"; do
        m="${MANIFESTS}/model_processed_manifest_${combo}_multimodal_turns.csv"
        [ -f "$m" ] && echo "  ${m}"
    done
fi
if [ "${INCLUDE_WINDOW_ABLATION}" = "1" ]; then
    echo ""
    echo "Paired window manifests:"
    for fs in "${FEATURE_SETS[@]}"; do
        m="${MANIFESTS}/model_processed_manifest_${fs}_raw_windows.csv"
        [ -f "$m" ] && echo "  ${m}"
    done
fi
