# ACM Clean Handoff

Date: 2026-06-14

This repository is a cleaned code/docs handoff for the ACM engagement modelling
workspace. It is intended to be usable by a collaborator, as the basis for a
GitHub hand-in, and as the code side of a RunPod working copy.

Large checkpoints, preprocessed tensors, raw data, caches, and generated bulk
outputs are intentionally not committed here. They are tracked in the external
artifact plan under `docs/handoff/docs/RUNPOD_ARTIFACT_PLAN.md`.

## Main Model Stories

1. NOXI / NOXI-J early gated-fusion regression
   - Description: `noxi settings.txt`
   - Main artifact: `ACM/MoE/noxi_joint_settings` in the original workspace or
     external artifact package.
   - This is the standalone gated multimodal dyadic-shared TCN, not the later
     NOXI MoE-style metadata-head ablation.

2. PinSoRo final MoE1 soft-confidence + HMM line
   - Summary: `MoE/experiment_score_summary_2026-06-10.md`
   - Main scripts: `MoE/train_moe1_metadata_head_tcn.py`,
     `MoE/fit_moe1_combiner.py`, `MoE/ablate_moe1_hmm_decoding.py`,
     `MoE/export_pinsoro_soft_confidence_hmm_submission.py`
   - Main artifacts are external: soft-confidence experts, two-head combiner,
     HMM decoding outputs, and submission export.

3. PinSoRo early-fusion/person-interaction next experiment
   - Summary: `MoE/handoff_modality_fusion_and_experts_2026-06-12.md`
   - Main scripts: `MoE/pinsoro_noxi_settings/train_person_interaction_fusion.py`
     and `MoE/pinsoro_noxi_settings/run_person_interaction_4gpu.py`
   - This is the current architecture direction: early-fused modalities,
     person-level experts, and partner interaction modeled on logits.

4. MPII generalizable-epoch / final multimodal model
   - Main scripts: `scripts/train_tcn_multimodal.py`,
     `scripts/mpii_singlemodality_loso.py`, `scripts/infer_tcn_multimodal.py`
   - Main artifacts are external: MPII transformed tensors, manifests, LOSO
     epoch-selection outputs, and final multimodal checkpoint.

## Repository Layout

```text
src/acm_pipeline/      Shared pipeline/model code
scripts/               General preprocessing/training/evaluation scripts
MoE/                   MoE, PinSoRo, NOXI experiment scripts and summaries
docs/                  Older pipeline docs plus current handoff docs
docs/handoff/          Clean handoff, RunPod, and artifact plan
artifacts/             Placeholder only; large files live outside git
```

## External Artifacts

Use the keep map and RunPod artifact plan before copying data:

```text
docs/handoff/docs/KEEP_MAP.md
docs/handoff/docs/RUNPOD_ARTIFACT_PLAN.md
docs/handoff/docs/OPEN_GAPS.md
```

The first RunPod package should start at 100 GB and should not include the full
192 GB `moe_data_soft_labels` tree. Copy only its small `outputs` folder unless
retraining soft-label experts is required.

## RunPod Starting Point

1. Create 100 GB RunPod Network Storage.
2. Launch a PyTorch/CUDA pod, preferably RTX 4090/24 GB first.
3. Preserve `/work/ACM` paths or create symlinks.
4. Copy this repository plus the external artifact folders listed in the RunPod
   plan.
5. Run path and CUDA smoke checks before full training.

Detailed commands are in:

```text
docs/handoff/EXPERIMENTS_RUNBOOK.md
```

## Cleanup Rule

Do not delete the original `/work/ACM/mltac-main` or
`/work/ACM/mltac-github-clean` workspaces until this clean repository imports,
the artifact manifest exists, and RunPod smoke checks pass.
