#!/bin/bash

set -euo pipefail

ACM_DIR="${ACM_DIR:-$(pwd)/ACM}"
PREPROCESS_SCRIPT="${ACM_DIR}/scripts/run_preprocessing.sh"
TRAIN_SCRIPT="${ACM_DIR}/scripts/run_training.sh"
LOG_DIR="${LOG_DIR:-/home/mlut/mltac/.garbage}"
DRY_RUN="${DRY_RUN:-0}"

ALL_STEPS=(1 2 3 4 5)
declare -A STEP_LABELS=(
    [1]="turns_simple_tcn"
    [2]="turns_dyadic_shared"
    [3]="turns_attention"
    [4]="turns_multimodal_winner"
    [5]="windows_winner"
)
declare -A VALID_STEPS=()
for step in "${ALL_STEPS[@]}"; do
    VALID_STEPS["${step}"]=1
done

usage() {
    cat <<EOF
Usage:
  bash ACM/scripts/submit_training_steps.sh
    bash ACM/scripts/submit_training_steps.sh 1 2 3
    bash ACM/scripts/submit_training_steps.sh --dry-run 2 3
    TRAINING_STEPS=1,2,3 bash ACM/scripts/submit_training_steps.sh

Behavior:
    Submits preprocessing first, then submits the selected training steps with an
    afterok dependency on the preprocessing job.

Environment:
  ACM_DIR          Override ACM root directory.
  LOG_DIR          Override SLURM stdout/stderr directory. Default: ${LOG_DIR}
  TRAINING_STEPS   Comma/space separated step list when no positional args are given.
  DRY_RUN=1        Print sbatch commands without submitting.
EOF
}

REQUESTED_STEPS=()
if [ "$#" -gt 0 ]; then
    for arg in "$@"; do
        case "${arg}" in
            --dry-run)
                DRY_RUN=1
                ;;
            --help|-h)
                usage
                exit 0
                ;;
            --*)
                echo "Unknown option: ${arg}" >&2
                usage >&2
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

if [ "${#REQUESTED_STEPS[@]}" -eq 0 ]; then
    REQUESTED_STEPS=("${ALL_STEPS[@]}")
fi

SELECTED_STEPS=()
declare -A SELECTED_STEP_MAP=()
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

if [ ! -f "${PREPROCESS_SCRIPT}" ]; then
    echo "Preprocessing script not found: ${PREPROCESS_SCRIPT}" >&2
    exit 1
fi

if [ ! -f "${TRAIN_SCRIPT}" ]; then
    echo "Training script not found: ${TRAIN_SCRIPT}" >&2
    exit 1
fi

mkdir -p "${LOG_DIR}"

preprocess_job_name="gcm_preprocess"
preprocess_output_path="${LOG_DIR}/${preprocess_job_name}.%j.out"
preprocess_error_path="${LOG_DIR}/${preprocess_job_name}.%j.err"
preprocess_cmd=(
    sbatch
    --parsable
    --export "ALL,INCLUDE_MULTIMODAL_ABLATION=$([ -n "${SELECTED_STEP_MAP[4]+x}" ] && echo 1 || echo 0),INCLUDE_WINDOW_ABLATION=$([ -n "${SELECTED_STEP_MAP[5]+x}" ] && echo 1 || echo 0)"
    --job-name "${preprocess_job_name}"
    --output "${preprocess_output_path}"
    --error "${preprocess_error_path}"
    "${PREPROCESS_SCRIPT}"
)

echo "=== Submit ACM Pipeline ==="
echo "ACM_DIR:           ${ACM_DIR}"
echo "PREPROCESS_SCRIPT: ${PREPROCESS_SCRIPT}"
echo "TRAIN_SCRIPT:      ${TRAIN_SCRIPT}"
echo "LOG_DIR:           ${LOG_DIR}"
echo "SELECTED_STEPS:    ${SELECTED_STEPS[*]}"
echo "DRY_RUN:           ${DRY_RUN}"
echo ""

if [ "${DRY_RUN}" = "1" ]; then
    echo "[dry] preprocessing"
    printf '      '
    printf '%q ' "${preprocess_cmd[@]}"
    echo ""
    preprocess_dependency="afterok:<preprocess_job_id>"
else
    echo "[submit] preprocessing"
    preprocess_submit="$("${preprocess_cmd[@]}")"
    preprocess_job_id="${preprocess_submit%%;*}"
    if [ -z "${preprocess_job_id}" ]; then
        echo "Failed to parse preprocessing job ID from sbatch output: ${preprocess_submit}" >&2
        exit 1
    fi
    preprocess_dependency="afterok:${preprocess_job_id}"
    echo "         job_id=${preprocess_job_id}"
fi
echo ""

declare -A STEP_JOB_IDS=()

for step in "${ALL_STEPS[@]}"; do
    if [ -z "${SELECTED_STEP_MAP[${step}]+x}" ]; then
        continue
    fi

    label="${STEP_LABELS[${step}]}"
    job_name="gcm_s${step}_${label}"
    output_path="${LOG_DIR}/${job_name}.%j.out"
    error_path="${LOG_DIR}/${job_name}.%j.err"
    dependency_job_ids=()
    if [ "${step}" -ge 4 ]; then
        if [ "${DRY_RUN}" = "1" ]; then
            dependency_job_ids+=("<preprocess_job_id>")
        else
            dependency_job_ids+=("${preprocess_job_id}")
        fi
        for dep_step in 1 2 3; do
            if [ -n "${SELECTED_STEP_MAP[${dep_step}]+x}" ]; then
                if [ "${DRY_RUN}" = "1" ]; then
                    dependency_job_ids+=("<step_${dep_step}_job_id>")
                else
                    dependency_job_ids+=("${STEP_JOB_IDS[${dep_step}]:-missing}")
                fi
            fi
        done
    else
        if [ "${DRY_RUN}" = "1" ]; then
            dependency_job_ids+=("<preprocess_job_id>")
        else
            dependency_job_ids+=("${preprocess_job_id}")
        fi
    fi

    dependency_string="afterok:${dependency_job_ids[0]}"
    dep_id=""
    for dep_id in "${dependency_job_ids[@]:1}"; do
        dependency_string+=":${dep_id}"
    done

    cmd=(
        sbatch
        --parsable
        --dependency "${dependency_string}"
        --export ALL
        --job-name "${job_name}"
        --output "${output_path}"
        --error "${error_path}"
        "${TRAIN_SCRIPT}"
        "${step}"
    )

    if [ "${DRY_RUN}" = "1" ]; then
        echo "[dry] step ${step} (${label})"
        printf '      '
        printf '%q ' "${cmd[@]}"
        echo ""
        continue
    fi

    echo "[submit] step ${step} (${label})"
    submit_output="$("${cmd[@]}")"
    step_job_id="${submit_output%%;*}"
    if [ -z "${step_job_id}" ]; then
        echo "Failed to parse job ID from sbatch output: ${submit_output}" >&2
        exit 1
    fi
    STEP_JOB_IDS["${step}"]="${step_job_id}"
    echo "         job_id=${step_job_id}"
done
