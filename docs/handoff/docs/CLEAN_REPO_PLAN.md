# Clean Repository Plan

Date: 2026-06-14

The clean version should be a single coherent repository that a collaborator can
read, run on RunPod, and cite for hand-in. It should not mirror the messy
workspace layout one-to-one.

## Proposed Structure

```text
ACM-clean/
  README.md
  requirements.txt
  docs/
    experiment_overview.md
    runpod_setup.md
    data_and_artifacts.md
    noxi_gated_fusion.md
    pinsoro_moe1_soft_confidence_hmm.md
    pinsoro_early_fusion_person_interaction.md
    mpii_epoch_selection.md
  src/
    acm_pipeline/
  scripts/
    preprocessing/
    training/
    evaluation/
    submission/
  experiments/
    summaries/
    small_results/
  configs/
  manifests/
    README.md
  artifacts/
    README.md
```

## Source Of Code

Base source:

```text
/work/ACM/mltac-github-clean/ACM/src/acm_pipeline
```

General scripts:

```text
/work/ACM/mltac-github-clean/ACM/scripts
```

MoE and newest PinSoRo scripts:

```text
/work/ACM/mltac-main/ACM/MoE/*.py
/work/ACM/mltac-main/ACM/MoE/pinsoro_noxi_settings/*.py
/work/ACM/mltac-main/ACM/scripts/analyze_pinsoro_*.py
/work/ACM/mltac-main/ACM/scripts/evaluate_pinsoro_checkpoint.py
/work/ACM/mltac-main/ACM/scripts/run_pinsoro_*_4gpu.py
```

Before publishing, the scripts should be grouped by purpose and imports should
be checked from the clean root.

## GitHub-Safe Documentation

The repo should explain four model stories:

1. NOXI early gated fusion regression
2. PinSoRo MoE1 soft-confidence + HMM final line
3. PinSoRo early-fusion/person-interaction next experiment
4. MPII multimodal epoch selection and final model

Each story should include:

```text
task
data inputs
model architecture
training command
evaluation command
expected outputs
artifact paths
known result summary
limitations/caveats
```

## Artifact Boundary

GitHub should contain pointers and manifests, not heavy data. Large files should
be referenced by relative paths under an external artifact root, for example:

```text
ACM_ARTIFACT_ROOT=/work/ACM/artifacts
```

The first artifact package should correspond to the existing 100 GB RunPod plan:

```text
precomputed PinSoRo domain_norm tensors
PinSoRo window manifests
person-interaction checkpoints
NOXI gated-fusion checkpoint/results/manifests
MoE1 soft-confidence + HMM artifacts
MPII transformed tensors/manifests/final model/LOSO outputs
```

## Cleanup Sequence

1. Freeze the current state in documentation.
2. Build a clean repository copy from selected code/docs only.
3. Add `.gitignore` for tensors, checkpoints, caches, virtualenvs, and generated
   outputs.
4. Add artifact README files instead of committing large artifacts.
5. Run import/smoke checks from the clean root.
6. Only then decide whether to remove or archive old workspace folders.

## Do Not Do In The First Pass

- Do not delete `/work/ACM/mltac-main` or `/work/ACM/mltac-github-clean`.
- Do not rely on git clone alone; important code is untracked or dirty.
- Do not commit full `processed/`, raw data, `.venv*`, checkpoints, or caches.
- Do not present validation-prior or `val_*_upper` diagnostics as deployable
  final results.
