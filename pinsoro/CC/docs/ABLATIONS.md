# CC Ablations

## Partner Ablation

Goal: compare no explicit partner interaction, late linear partner interaction, and late gated partner interaction.

All rows use audio_w2vbert2, text_xlm_roberta, visual_videomae, w2400_s1200 dyadic manifests, raw validation Cohen's kappa, no logit bias, and no HMM.

| Head | No partner | Late linear partner | Late gated partner |
|---|---:|---:|---:|
| CC task | 0.330583 | 0.376920 | 0.373723 |
| CC social | 0.354842 | 0.346729 | 0.347857 |

Run mapping:

| Row | Run directory |
|---|---|
| CC task, no partner | `artifacts/runs/pinsoro_cc_task_shared_none_shared_tcn_delta010_metadata_seed13` |
| CC task, late linear | `artifacts/runs/pinsoro_cc_task_shared_linear_shared_tcn_delta010_metadata_seed13` |
| CC task, late gated | `artifacts/runs/pinsoro_cc_task_submitted_late_gated_shared_tcn_delta010_metadata_seed13` |
| CC social, no partner | `artifacts/runs/pinsoro_cc_social_submitted_no_partner_head_adapters_delta010_metadata_seed13` |
| CC social, late linear | `artifacts/runs/pinsoro_cc_headarch_head_adapters_delta010_metadata_seed13` |
| CC social, late gated | `artifacts/runs/pinsoro_cc_both_headarch_head_adapters_logit_gated_scale0.1_delta010_metadata_both_seed13` |

## Submitted-Checkpoint Feature/Modality Ablation

Goal: compare modality contribution using the exact submitted checkpoints.

Method:

1. Load the submitted checkpoint `model_best.pt`.
2. Keep weights fixed.
3. Run validation inference seven times while masking excluded modalities to zero.
4. Report raw validation kappa.
5. Do not apply logit bias or HMM.

| Modalities | CC task | CC social |
|---|---:|---:|
| A+T+V | 0.376920 | 0.346729 |
| A+T | 0.140607 | 0.195056 |
| A+V | 0.373855 | 0.351916 |
| T+V | 0.382419 | 0.314679 |
| A | 0.089435 | 0.164105 |
| T | 0.160351 | 0.100059 |
| V | 0.380484 | 0.290584 |

Output CSV:

```text
artifacts/ablation_outputs/pinsoro_cc_submitted_checkpoint_modality_masks_3006/submitted_checkpoint_modality_mask_summary.csv
```

Important: older modality-ablation directories were retrained models. They are not used for the table above because their A+T+V rows are not identical to the submitted checkpoints.
