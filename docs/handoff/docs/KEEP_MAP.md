# Keep Map

Date: 2026-06-14

This file defines what to keep from `/work/ACM` and how to classify it for a
clean handoff. It is intentionally conservative: when an item is clearly part of
the current story, keep it first and prune only after verification.

## A. Top-Level Handoff And Decision Notes

Keep in GitHub/shareable docs:

```text
/work/ACM/EXPERIMENTS_RUNBOOK.md
/work/ACM/uni_cloud_agent_runpod_handoff.txt
/work/ACM/hand_off_moe_best_noxi_architecture_2026-06-12.md
/work/ACM/mltac-main/ACM/MoE/moe1_results_overview_2026-06-09.md
/work/ACM/mltac-main/ACM/MoE/experiment_score_summary_2026-06-10.md
/work/ACM/mltac-main/ACM/MoE/handoff_modality_fusion_and_experts_2026-06-12.md
/work/ACM/mltac-main/ACM/MoE/noxi_metadata_head_reproducibility_2026-06-10.md
/work/ACM/mltac-main/ACM/MoE/class0_downweight_experiment_summary_2026-06-10.md
/work/ACM/mltac-main/ACM/MoE/class3_training_ablation_summary_2026-06-10.md
/work/ACM/mltac-main/ACM/MoE/cr_social_class3_problem_note_2026-06-10.md
/work/ACM/mltac-main/ACM/MoE/moe2_experiment_summary_2026-06-10.md
/work/ACM/mltac-main/ACM/MoE/mamba_experiment_summary_2026-06-11.md
/work/ACM/mltac-github-clean/ACM/noxi settings.txt
```

Keep as small result overviews:

```text
/work/ACM/mpii_loso_singlemodality_all_results.csv
/work/ACM/mpii_loso_singlemodality_feature_summary.csv
/work/ACM/mltac-github-clean/ACM/outputs/experiments/mpii_loso_singlemodality/mpii_loso_singlemodality_all_results.csv
/work/ACM/mltac-github-clean/ACM/outputs/experiments/mpii_loso_singlemodality/mpii_loso_singlemodality_feature_summary.csv
/work/ACM/mltac-github-clean/ACM/outputs/experiments/mpii_loso_singlemodality/mpii_loso_singlemodality_summary.csv
```

## B. Code To Keep In The Clean Repository

Base pipeline from:

```text
/work/ACM/mltac-github-clean/ACM/src/acm_pipeline/
/work/ACM/mltac-github-clean/ACM/scripts/
/work/ACM/mltac-github-clean/ACM/docs/
/work/ACM/mltac-github-clean/ACM/README.md
/work/ACM/mltac-github-clean/ACM/requirements.txt
```

MoE / PinSoRo / NOXI experiment code from:

```text
/work/ACM/mltac-main/ACM/MoE/*.py
/work/ACM/mltac-main/ACM/MoE/pinsoro_noxi_settings/*.py
/work/ACM/mltac-main/ACM/scripts/analyze_pinsoro_*.py
/work/ACM/mltac-main/ACM/scripts/evaluate_pinsoro_checkpoint.py
/work/ACM/mltac-main/ACM/scripts/pinsoro_fit_apply_domain_feature_transform.py
/work/ACM/mltac-main/ACM/scripts/pinsoro_prepare_domain_norm_5foldcv.py
/work/ACM/mltac-main/ACM/scripts/pinsoro_split_domain_window_manifests.py
/work/ACM/mltac-main/ACM/scripts/run_pinsoro_*_4gpu.py
```

Important note: several of these files are untracked or dirty in git. The clean
repo should be built from the working tree, not from a fresh clone only.

## C. Final PinSoRo MoE1 Soft-Confidence + HMM Line

Keep as external artifacts for reproducibility and RunPod:

```text
/work/ACM/mltac-main/ACM/MoE/experiments/moe1_soft_confidence_metadata_head_experts
/work/ACM/mltac-main/ACM/MoE/experiments/moe1_soft_confidence_metadata_head_combiners
/work/ACM/mltac-main/ACM/MoE/experiments/moe1_soft_confidence_hmm_decoding
/work/ACM/mltac-main/ACM/MoE/experiments/soft_confidence_hmm_submission_export
/work/ACM/mltac-main/ACM/MoE/moe_data_soft_labels/outputs
```

Preserve, but do not include in the first 100 GB RunPod package unless retraining
the soft-label experts is required:

```text
/work/ACM/mltac-main/ACM/MoE/moe_data_soft_labels/processed  # 69G
/work/ACM/mltac-main/ACM/MoE/moe_data_soft_labels/cache      # 123G
```

Keep small summaries/results in GitHub if size allows:

```text
comparison.json
*_hmm_results.csv
*_class_metrics.csv
combined_hmm_results.csv
export_manifest.json
```

Do not confuse this with older metadata-free or validation-prior upper-bound
diagnostics. Those can stay as history but should not be presented as the final
deployable line.

## D. Standalone NOXI Early Gated Fusion Model

Keep the exact model described by:

```text
/work/ACM/mltac-github-clean/ACM/noxi settings.txt
```

Primary artifact folder:

```text
/work/ACM/mltac-main/ACM/MoE/noxi_joint_settings
```

Important contained files:

```text
experiments/noxi_noxij_audio_text_visual_w500_s125_gated_dyadic_shared_seed13/config.json
experiments/noxi_noxij_audio_text_visual_w500_s125_gated_dyadic_shared_seed13/model_best.pt
experiments/noxi_noxij_audio_text_visual_w500_s125_gated_dyadic_shared_seed13/training_log.csv
experiments/noxi_noxij_audio_text_visual_w500_s125_gated_dyadic_shared_seed13/metrics_overall.csv
experiments/noxi_noxij_audio_text_visual_w500_s125_gated_dyadic_shared_seed13/metrics_by_session.csv
experiments/noxi_noxij_audio_text_visual_w500_s125_gated_dyadic_shared_seed13/metrics_by_dataset.csv
experiments/noxi_noxij_audio_text_visual_w500_s125_gated_dyadic_shared_seed13/val_predictions.csv
experiments/noxi_noxij_audio_text_visual_w500_s125_gated_dyadic_shared_seed13/val_gate_weights.csv
manifests/*.csv
```

Separate ablation/transfer history, not the standalone gated-fusion model:

```text
/work/ACM/mltac-main/ACM/MoE/experiments/noxi_moe1_noxi_metadata_head_experts
/work/ACM/mltac-main/ACM/MoE/experiments/noxi_moe1_noxi_metadata_head_combiners
/work/ACM/mltac-main/ACM/MoE/experiments/noxi_moe1_noxi_j_metadata_head_experts
/work/ACM/mltac-main/ACM/MoE/experiments/noxi_moe1_noxi_j_metadata_head_combiners
```

## E. Newest Early-Fusion / Person-Interaction PinSoRo Work

Keep code in GitHub:

```text
/work/ACM/mltac-main/ACM/MoE/pinsoro_noxi_settings/train_person_interaction_fusion.py
/work/ACM/mltac-main/ACM/MoE/pinsoro_noxi_settings/run_person_interaction_4gpu.py
/work/ACM/mltac-main/ACM/MoE/pinsoro_noxi_settings/train_gated_fusion.py
/work/ACM/mltac-main/ACM/MoE/pinsoro_noxi_settings/run_gated_fusion_4gpu.py
/work/ACM/mltac-main/ACM/MoE/pinsoro_noxi_settings/combine_horizon_experts.py
/work/ACM/mltac-main/ACM/MoE/pinsoro_noxi_settings/export_gated_fusion_checkpoint.py
```

Keep artifacts externally:

```text
/work/ACM/mltac-main/ACM/MoE/experiments/pinsoro_person_interaction_early_fusion
/work/ACM/mltac-main/ACM/MoE/experiments/pinsoro_moe_settings_early_fusion_hmm_concat
/work/ACM/mltac-main/ACM/MoE/experiments/pinsoro_moe_settings_early_fusion_hmm_gated
/work/ACM/mltac-main/ACM/MoE/experiments/pinsoro_noxi_settings_gated_fusion
/work/ACM/mltac-main/ACM/MoE/experiments/pinsoro_noxi_settings_gated_fusion_hmm
/work/ACM/mltac-main/ACM/MoE/experiments/pinsoro_two_horizon_early_concat_hmm_fixed_average
/work/ACM/mltac-main/ACM/MoE/experiments/pinsoro_two_horizon_early_concat_hmm_train_grid
```

Required data for immediate continuation:

```text
/work/ACM/mltac-main/ACM/MoE/moe_data/outputs/windows_w2400_s1200
/work/ACM/mltac-main/ACM/MoE/moe_data/processed/domain_norm
```

Do not copy initially unless deliberately switching to mmap loading:

```text
/work/ACM/mltac-main/ACM/MoE/moe_data/processed/domain_norm_mmap
```

## F. MPII Generalizable Epoch / Final Model Work

Keep code in GitHub:

```text
/work/ACM/mltac-github-clean/ACM/scripts/train_tcn_multimodal.py
/work/ACM/mltac-github-clean/ACM/scripts/infer_tcn_multimodal.py
/work/ACM/mltac-github-clean/ACM/scripts/mpii_singlemodality_loso.py
/work/ACM/mltac-github-clean/ACM/scripts/run_mpiii_test_multimodal_eval.sh
/work/ACM/mltac-github-clean/ACM/scripts/collect_results.py
```

Keep artifacts externally:

```text
/work/ACM/mltac-github-clean/ACM/processed/transformed/mpiii_eval
/work/ACM/mltac-github-clean/ACM/outputs/mpiii_eval/manifests
/work/ACM/mltac-github-clean/ACM/outputs/experiments/mpii_final_multimodal
/work/ACM/mltac-github-clean/ACM/outputs/experiments/mpii_loso_multimodal_epoch_selection
/work/ACM/mltac-github-clean/ACM/outputs/experiments/mpii_loso_singlemodality
```

Known multimodal LOSO best epochs:

```text
008: 9
009: 11
010: 3
026: 2
027: 7
028: 5
```

Known final multimodal checkpoint:

```text
/work/ACM/mltac-github-clean/ACM/outputs/experiments/mpii_final_multimodal/mpii_final_visual_videomae_audio_egemaps_text_xlm_roberta_gated_seed13/model_best.pt
```

## G. Exclude From GitHub

Exclude:

```text
*.pt
*.pth
*.npz
*.npy
*.pkl
__pycache__/
.venv*/
cache/
processed/
outputs/experiments/*/model_*.pt
ACM/MoE/moe_data/
ACM/MoE/moe_data_soft_labels/
ACM/MoE/noxi_data/
ACM/MoE/noxi_j_data/
ACM/MoE/experiments/*/model_*.pt
```

Keep selected small CSV/JSON/MD result summaries in GitHub even when they live
under `outputs/` or `experiments/`; put large artifacts in the external package.
