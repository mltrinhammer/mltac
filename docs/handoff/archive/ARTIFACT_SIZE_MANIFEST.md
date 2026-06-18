# Artifact Size Manifest

Date: 2026-06-14

Measured on the university cloud from `/work/ACM`.

## First 100 GB RunPod Candidate

```text
3.0M   mltac-main/ACM/MoE/moe_data/outputs/windows_w2400_s1200
35G    mltac-main/ACM/MoE/moe_data/processed/domain_norm
24M    mltac-main/ACM/MoE/experiments/pinsoro_person_interaction_early_fusion
2.8G   mltac-main/ACM/MoE/experiments/moe1_soft_confidence_metadata_head_experts
12K    mltac-main/ACM/MoE/experiments/moe1_soft_confidence_metadata_head_combiners
176K   mltac-main/ACM/MoE/experiments/moe1_soft_confidence_hmm_decoding
11M    mltac-main/ACM/MoE/experiments/soft_confidence_hmm_submission_export
5.2M   mltac-main/ACM/MoE/moe_data_soft_labels/outputs
192M   mltac-main/ACM/MoE/noxi_joint_settings
6.0G   mltac-github-clean/ACM/processed/transformed/mpiii_eval
209M   mltac-github-clean/ACM/outputs/mpiii_eval/manifests
246M   mltac-github-clean/ACM/outputs/experiments/mpii_final_multimodal
25M    mltac-github-clean/ACM/outputs/experiments/mpii_loso_multimodal_epoch_selection
2.3G   mltac-github-clean/ACM/outputs/experiments/mpii_loso_singlemodality
```

Approximate total: under 50 GB plus code and small docs, leaving margin in a
100 GB RunPod Network Storage volume.

## Preserve Separately, Not First 100 GB Package

```text
69G    mltac-main/ACM/MoE/moe_data_soft_labels/processed
123G   mltac-main/ACM/MoE/moe_data_soft_labels/cache
49G    mltac-main/ACM/MoE/moe_data/processed/domain_norm_mmap
```

These are useful for retraining or alternate loading strategies, but they should
not be copied in the first RunPod package unless specifically needed.

## Clean Repository Size

```text
1.5M   /work/ACM/ACM-clean
```

No files larger than 5 MB were found in the clean repository during verification.
No `*.pt`, `*.npz`, `*.npy`, `*.pkl`, or `*.pyc` files were found in the clean
repository during verification.
