# RunPod Artifact Plan

Date: 2026-06-14

This artifact plan is compatible with the clean GitHub repository plan. The
clean repository supplies code/docs; this package supplies the heavy files needed
to run or resume experiments.

## Recommended First Storage

Provision 100 GB RunPod Network Storage for the first pass.

Expected first package size: about 45-55 GB plus code and small summaries.

## Preferred RunPod Layout

Preserve the current absolute path if possible:

```text
/work/ACM
```

If RunPod storage mounts somewhere else, create symlinks so manifests containing
`/work/ACM/...` still resolve.

## First Artifact Package

### Code Snapshot

Copy the clean repository after it is created. Until then, use selective `rsync`
from the working trees rather than `git clone` alone.

### PinSoRo Person-Interaction Continuation

```text
/work/ACM/mltac-main/ACM/MoE/moe_data/outputs/windows_w2400_s1200
/work/ACM/mltac-main/ACM/MoE/moe_data/processed/domain_norm
/work/ACM/mltac-main/ACM/MoE/experiments/pinsoro_person_interaction_early_fusion
```

Purpose: resume newest early-fusion/person-partner interaction experiments.

### PinSoRo Final MoE1 Soft-Confidence + HMM

```text
/work/ACM/mltac-main/ACM/MoE/experiments/moe1_soft_confidence_metadata_head_experts
/work/ACM/mltac-main/ACM/MoE/experiments/moe1_soft_confidence_metadata_head_combiners
/work/ACM/mltac-main/ACM/MoE/experiments/moe1_soft_confidence_hmm_decoding
/work/ACM/mltac-main/ACM/MoE/experiments/soft_confidence_hmm_submission_export
/work/ACM/mltac-main/ACM/MoE/moe_data_soft_labels/outputs
```

Purpose: preserve current chosen PinSoRo submission/evaluation line.

Do not copy the full `moe_data_soft_labels` tree in the first 100 GB package:

```text
/work/ACM/mltac-main/ACM/MoE/moe_data_soft_labels/processed  # 69G
/work/ACM/mltac-main/ACM/MoE/moe_data_soft_labels/cache      # 123G
```

Those folders are preservation/retraining inputs, not needed for the first
RunPod continuation package unless we decide to retrain the soft-label experts.

### Standalone NOXI Gated Fusion

```text
/work/ACM/mltac-main/ACM/MoE/noxi_joint_settings
/work/ACM/mltac-github-clean/ACM/noxi settings.txt
```

Purpose: preserve the exact NOXI early gated-fusion model, checkpoint,
predictions, gate weights, metrics, and manifests.

### MPII Final / Epoch Selection

```text
/work/ACM/mltac-github-clean/ACM/processed/transformed/mpiii_eval
/work/ACM/mltac-github-clean/ACM/outputs/mpiii_eval/manifests
/work/ACM/mltac-github-clean/ACM/outputs/experiments/mpii_final_multimodal
/work/ACM/mltac-github-clean/ACM/outputs/experiments/mpii_loso_multimodal_epoch_selection
/work/ACM/mltac-github-clean/ACM/outputs/experiments/mpii_loso_singlemodality
```

Purpose: preserve transformed MPII tensors, LOSO epoch-selection evidence, and
the final trained MPII model.

## Initial Exclusions

Do not copy initially:

```text
/work/ACM/mltac-main/ACM/MoE/moe_data/processed/domain_norm_mmap
/work/ACM/mltac-main/ACM/MoE/moe_data_soft_labels/processed
/work/ACM/mltac-main/ACM/MoE/moe_data_soft_labels/cache
/work/ACM/mltac-main/ACM/processed
/work/ACM/mltac-github-clean/ACM/processed except processed/transformed/mpiii_eval
raw video/audio trees
virtualenvs
__pycache__
large historical experiment folders not listed in KEEP_MAP.md
```

## Verification On RunPod

Before training:

```bash
python --version
nvidia-smi
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY
```

Check required artifact paths:

```bash
test -f /work/ACM/mltac-main/ACM/MoE/noxi_joint_settings/experiments/noxi_noxij_audio_text_visual_w500_s125_gated_dyadic_shared_seed13/model_best.pt
test -f /work/ACM/mltac-main/ACM/MoE/experiments/pinsoro_person_interaction_early_fusion/pinsoro_cc_audio_text_visual_concat_shared_person_linear_seed13/model_last.pt
test -f /work/ACM/mltac-github-clean/ACM/outputs/experiments/mpii_final_multimodal/mpii_final_visual_videomae_audio_egemaps_text_xlm_roberta_gated_seed13/model_best.pt
```

Then run the smoke checks and resume commands from:

```text
/work/ACM/EXPERIMENTS_RUNBOOK.md
```
