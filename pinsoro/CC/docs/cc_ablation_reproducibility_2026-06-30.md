# CC Ablation Reproducibility Notes

Date: 2026-06-30

This note documents the CC task/social ablation checks and runs performed on the RunPod at:

```text
ssh root@38.128.233.200 -p 46198 -i /work/.ssh/runpod_ed25519
```

Project root on pod:

```text
/workspace/ACM/ACM-clean
```

All reported ablation numbers below are raw validation Cohen's kappa from `metrics_overall.csv` unless otherwise stated. "Raw" here means no logit-bias postprocessing and no HMM smoothing.

## Submitted Model Checkpoints

### CC Task Submitted Base

Run directory:

```text
/workspace/ACM/ACM-clean/MoE/experiments/pinsoro_cc_core_architecture_delta010_metadata/pinsoro_cc_task_shared_linear_shared_tcn_delta010_metadata_seed13
```

Relevant config:

```text
domain_scope: CC
active_heads: ['task']
encoder_sharing: shared
head_architecture: shared_tcn
interaction_mode: linear
interaction_scale: 0.1
cc_task_weighting: targeted
cc_task_target_class1_weight: 2.0
```

Raw validation kappa:

```text
CC task: 0.37692018200554195
```

Submitted postprocessing for CC task was found in submission summaries:

```text
logit bias: [0.5, -0.25, -0.5, -0.25]
HMM: mix=1.0, strength=12.0
```

Those postprocessing steps were not applied for the ablation tables.

### CC Social Submitted Base

Run directory:

```text
/workspace/ACM/ACM-clean/MoE/experiments/pinsoro_head_architectures_temporal_delta010_metadata/pinsoro_cc_headarch_head_adapters_delta010_metadata_seed13
```

Relevant config:

```text
domain_scope: CC
active_heads: ['task', 'social']
encoder_sharing: shared
head_architecture: head_adapters
head_adapter_levels: 1
interaction_mode: linear
interaction_scale: 0.1
```

Raw validation kappa:

```text
CC social: 0.3467286999282731
```

Submitted social logit bias used elsewhere:

```text
[0.0, 0.0, 1.0, -1.5, 0.75]
```

No HMM was used for the submitted CC social interpretation. Logit bias was not applied for the raw ablation tables.

## Partner Ablation Table

Goal: compare no explicit partner interaction, late linear partner interaction, and late gated partner interaction.

All runs use:

```text
audio_w2vbert2
text_xlm_roberta
visual_videomae
windows_w2400_s1200 dyadic manifests
raw validation kappa
no logit bias
no HMM
```

Final table:

| Head | No partner | Late linear partner | Late gated partner |
|---|---:|---:|---:|
| CC task | 0.330583 | 0.376920 | 0.373723 |
| CC social | 0.354842 | 0.346729 | 0.347857 |

### Partner Run Directories

CC task, no partner:

```text
/workspace/ACM/ACM-clean/MoE/experiments/pinsoro_cc_core_architecture_delta010_metadata/pinsoro_cc_task_shared_none_shared_tcn_delta010_metadata_seed13
```

CC task, late linear partner:

```text
/workspace/ACM/ACM-clean/MoE/experiments/pinsoro_cc_core_architecture_delta010_metadata/pinsoro_cc_task_shared_linear_shared_tcn_delta010_metadata_seed13
```

CC task, late gated partner:

```text
/workspace/ACM/ACM-clean/MoE/experiments/pinsoro_cc_submitted_partner_ablation_3006/pinsoro_cc_task_submitted_late_gated_shared_tcn_delta010_metadata_seed13
```

CC social, no partner:

```text
/workspace/ACM/ACM-clean/MoE/experiments/pinsoro_cc_submitted_partner_ablation_3006/pinsoro_cc_social_submitted_no_partner_head_adapters_delta010_metadata_seed13
```

CC social, late linear partner:

```text
/workspace/ACM/ACM-clean/MoE/experiments/pinsoro_head_architectures_temporal_delta010_metadata/pinsoro_cc_headarch_head_adapters_delta010_metadata_seed13
```

CC social, late gated partner:

```text
/workspace/ACM/ACM-clean/MoE/experiments/pinsoro_logit_interaction_ablation_delta010_metadata/pinsoro_cc_both_headarch_head_adapters_logit_gated_scale0.1_delta010_metadata_both_seed13
```

### New Partner Runs Launched

Output root:

```text
/workspace/ACM/ACM-clean/MoE/experiments/pinsoro_cc_submitted_partner_ablation_3006
```

The following two missing cells were trained:

1. CC task late gated, submitted-style shared TCN:

```bash
cd /workspace/ACM/ACM-clean
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/usr/bin/python3 MoE/pinsoro_noxi_settings/train_person_interaction_fusion_temporal.py \
  --manifest \
    MoE/moe_data/outputs/windows_w2400_s1200/audio_w2vbert2_w2400_s1200_dyadic.csv \
    MoE/moe_data/outputs/windows_w2400_s1200/text_xlm_roberta_w2400_s1200_dyadic.csv \
    MoE/moe_data/outputs/windows_w2400_s1200/visual_videomae_w2400_s1200_dyadic.csv \
  --domain-scope CC \
  --output-root MoE/experiments/pinsoro_cc_submitted_partner_ablation_3006 \
  --run-name pinsoro_cc_task_submitted_late_gated_shared_tcn_delta010_metadata_seed13 \
  --fusion-mode concat \
  --fusion-channels 64 \
  --person-hidden-channels 64 \
  --person-levels 5 \
  --person-kernel-size 11 \
  --dropout 0.2 \
  --modality-dropout 0.1 \
  --causal-tcn \
  --encoder-sharing shared \
  --head-architecture shared_tcn \
  --head-adapter-levels 1 \
  --interaction-mode gated \
  --interaction-hidden-channels 32 \
  --interaction-kernel-size 5 \
  --interaction-scale 0.1 \
  --metadata-mode age_gender_role \
  --metadata-embedding-dim 16 \
  --metadata-dropout 0.2 \
  --cc-task-weighting targeted \
  --cc-task-target-class0-weight 1.0 \
  --cc-task-target-class1-weight 2.0 \
  --cc-task-target-class2-weight 1.0 \
  --cc-task-target-class3-weight 1.0 \
  --temporal-delta-weight 0.1 \
  --soft-label-mode none \
  --active-heads task \
  --batch-size 32 \
  --num-workers 2 \
  --epochs 30 \
  --patience 6 \
  --min-epochs 5 \
  --min-delta 0.001 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --seed 13 \
  --device cuda
```

2. CC social no partner, submitted-style head adapters:

```bash
cd /workspace/ACM/ACM-clean
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/usr/bin/python3 MoE/pinsoro_noxi_settings/train_person_interaction_fusion_temporal.py \
  --manifest \
    MoE/moe_data/outputs/windows_w2400_s1200/audio_w2vbert2_w2400_s1200_dyadic.csv \
    MoE/moe_data/outputs/windows_w2400_s1200/text_xlm_roberta_w2400_s1200_dyadic.csv \
    MoE/moe_data/outputs/windows_w2400_s1200/visual_videomae_w2400_s1200_dyadic.csv \
  --domain-scope CC \
  --output-root MoE/experiments/pinsoro_cc_submitted_partner_ablation_3006 \
  --run-name pinsoro_cc_social_submitted_no_partner_head_adapters_delta010_metadata_seed13 \
  --fusion-mode concat \
  --fusion-channels 64 \
  --person-hidden-channels 64 \
  --person-levels 5 \
  --person-kernel-size 11 \
  --dropout 0.2 \
  --modality-dropout 0.1 \
  --causal-tcn \
  --encoder-sharing shared \
  --head-architecture head_adapters \
  --head-adapter-levels 1 \
  --interaction-mode none \
  --interaction-hidden-channels 32 \
  --interaction-kernel-size 5 \
  --interaction-scale 0.0 \
  --metadata-mode age_gender_role \
  --metadata-embedding-dim 16 \
  --metadata-dropout 0.2 \
  --temporal-delta-weight 0.1 \
  --soft-label-mode none \
  --active-heads task social \
  --batch-size 32 \
  --num-workers 2 \
  --epochs 30 \
  --patience 6 \
  --min-epochs 5 \
  --min-delta 0.001 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --seed 13 \
  --device cuda
```

Both completed via early stopping.

## Submitted-Checkpoint Feature/Modality Ablation

Goal: produce a feature ablation table comparable to the partner table by using the exact submitted checkpoints rather than retraining a separate `modab_atv` model.

Method:

1. Load the submitted checkpoint `model_best.pt`.
2. Keep model weights fixed.
3. Run validation inference seven times while masking excluded input modalities to zero.
4. Report raw validation kappa.
5. Do not apply logit bias or HMM.

Evaluator script created on pod:

```text
/workspace/ACM/ACM-clean/MoE/pinsoro_noxi_settings/evaluate_submitted_modality_masks.py
```

Local copy in workspace:

```text
/work/evaluate_submitted_modality_masks.py
```

Output root:

```text
/workspace/ACM/ACM-clean/MoE/experiments/pinsoro_cc_submitted_checkpoint_modality_masks_3006
```

Summary CSV:

```text
/workspace/ACM/ACM-clean/MoE/experiments/pinsoro_cc_submitted_checkpoint_modality_masks_3006/submitted_checkpoint_modality_mask_summary.csv
```

Command used:

```bash
cd /workspace/ACM/ACM-clean
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/usr/bin/python3 MoE/pinsoro_noxi_settings/evaluate_submitted_modality_masks.py \
  --task-run MoE/experiments/pinsoro_cc_core_architecture_delta010_metadata/pinsoro_cc_task_shared_linear_shared_tcn_delta010_metadata_seed13 \
  --social-run MoE/experiments/pinsoro_head_architectures_temporal_delta010_metadata/pinsoro_cc_headarch_head_adapters_delta010_metadata_seed13 \
  --output-root MoE/experiments/pinsoro_cc_submitted_checkpoint_modality_masks_3006 \
  --device cuda \
  --batch-size 32 \
  --num-workers 2
```

Feature table:

| Modalities | CC task | CC social |
|---|---:|---:|
| A+T+V | 0.376920 | 0.346729 |
| A+T | 0.140607 | 0.195056 |
| A+V | 0.373855 | 0.351916 |
| T+V | 0.382419 | 0.314679 |
| A | 0.089435 | 0.164105 |
| T | 0.160351 | 0.100059 |
| V | 0.380484 | 0.290584 |

Legend:

```text
A = audio_w2vbert2
T = text_xlm_roberta
V = visual_videomae
```

The `A+T+V` rows match the submitted late-linear baseline rows in the partner table.

## Important Interpretation Notes

1. The old modality ablation directories under `pinsoro_cc_modality_ablation_task_social_2906` and `pinsoro_cc_social_submitted_modality_ablation_2906` contain retrained models. Their `A+T+V` rows are not identical to the submitted checkpoints.

2. The new feature table described above is checkpoint-only inference and is therefore comparable to the partner table baseline.

3. The partner table variants are matched architecture training runs. They are not inference-time masks.

4. The CC task sensitivity analysis used the correct submitted task base model:

```text
MoE/experiments/pinsoro_cc_core_architecture_delta010_metadata/pinsoro_cc_task_shared_linear_shared_tcn_delta010_metadata_seed13
```

5. The earlier CC social sensitivity/HMM analysis used an old hidden-attention social model in one script, but CC social was removed from the HMM sensitivity analysis because the submitted CC social interpretation is no-HMM.

## Useful Verification Commands

Partner table extraction:

```bash
cd /workspace/ACM/ACM-clean
python3 - <<'PY'
import csv
from pathlib import Path
runs=[
("CC task","No partner","task",Path("MoE/experiments/pinsoro_cc_core_architecture_delta010_metadata/pinsoro_cc_task_shared_none_shared_tcn_delta010_metadata_seed13")),
("CC task","Late linear","task",Path("MoE/experiments/pinsoro_cc_core_architecture_delta010_metadata/pinsoro_cc_task_shared_linear_shared_tcn_delta010_metadata_seed13")),
("CC task","Late gated","task",Path("MoE/experiments/pinsoro_cc_submitted_partner_ablation_3006/pinsoro_cc_task_submitted_late_gated_shared_tcn_delta010_metadata_seed13")),
("CC social","No partner","social",Path("MoE/experiments/pinsoro_cc_submitted_partner_ablation_3006/pinsoro_cc_social_submitted_no_partner_head_adapters_delta010_metadata_seed13")),
("CC social","Late linear","social",Path("MoE/experiments/pinsoro_head_architectures_temporal_delta010_metadata/pinsoro_cc_headarch_head_adapters_delta010_metadata_seed13")),
("CC social","Late gated","social",Path("MoE/experiments/pinsoro_logit_interaction_ablation_delta010_metadata/pinsoro_cc_both_headarch_head_adapters_logit_gated_scale0.1_delta010_metadata_both_seed13")),
]
for head, mode, metric_head, path in runs:
    with (path/"metrics_overall.csv").open() as f:
        for row in csv.DictReader(f):
            if row["group"]=="overall" and row["head"]==metric_head:
                print(head, mode, row["kappa"], path)
PY
```

Feature table extraction:

```bash
cat /workspace/ACM/ACM-clean/MoE/experiments/pinsoro_cc_submitted_checkpoint_modality_masks_3006/submitted_checkpoint_modality_mask_summary.csv
```
