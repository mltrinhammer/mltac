# NOXI Metadata-Head Baselines - Reproducibility Record - 2026-06-10

Workspace:

```text
/work/ACM/mltac-main
```

## Scripts

The NOXI metadata-head branch is isolated under `ACM/MoE` and does not modify PinSoRo scripts.

```text
ACM/MoE/prepare_noxi_metadata.py
ACM/MoE/train_noxi_metadata_head_tcn.py
ACM/MoE/evaluate_noxi_metadata_head_checkpoint.py
ACM/MoE/run_noxi_metadata_head_experts_4gpu.py
ACM/MoE/fit_noxi_metadata_head_combiner.py
```

## Metadata

Metadata was extracted from raw NOXI / NOXI-J zip archives.

```text
ACM/MoE/noxi_data/outputs/metadata/role_metadata.csv
ACM/MoE/noxi_j_data/outputs/metadata/role_metadata.csv
```

Fields:

```text
age, gender, language
```

Regenerate:

```bash
cd /work/ACM/mltac-main
/work/ACM/mltac-main/ACM/.venv-gpu/bin/python ACM/MoE/prepare_noxi_metadata.py --corpora noxi noxi_j
```

## Architecture

All metadata-head experts use:

```text
model: dyadic metadata-head regression TCN
outputs: novice, expert
features: visual_videomae, audio_w2vbert2, text_xlm_roberta
window / stride: 2000 / 1000
levels: 5
kernel size: 11
hidden channels: 64
dropout: 0.2
causal TCN: yes
metadata mode: age_gender_language
metadata dropout: 0.2
seed: 13
```

## Selected Results

### NOXI

Best clean validation result:

```text
metadata-head shared combiner
overall CCC: 0.856139
novice CCC: 0.675232
expert CCC: 0.837199
```

The selected NOXI combiner depends on all three completed metadata-head experts:

```text
ACM/MoE/experiments/noxi_moe1_noxi_metadata_head_experts/noxi_visual_videomae_dyadic_tcn_k11_metadata_head_seed13
ACM/MoE/experiments/noxi_moe1_noxi_metadata_head_experts/noxi_audio_w2vbert2_dyadic_tcn_k11_metadata_head_seed13
ACM/MoE/experiments/noxi_moe1_noxi_metadata_head_experts/noxi_text_xlm_roberta_dyadic_tcn_k11_metadata_head_seed13
```

Individual expert validation CCCs:

```text
visual_videomae:  0.760422
audio_w2vbert2:   0.818758
text_xlm_roberta: 0.656231
```

Combiner summary:

```text
ACM/MoE/experiments/noxi_moe1_noxi_metadata_head_combiners/summary.csv
ACM/MoE/experiments/noxi_moe1_noxi_metadata_head_combiners/summary.json
ACM/MoE/experiments/noxi_moe1_noxi_metadata_head_combiners/weights.json
```

Reproduce experts:

```bash
cd /work/ACM/mltac-main
/work/ACM/mltac-main/ACM/.venv-gpu/bin/python ACM/MoE/run_noxi_metadata_head_experts_4gpu.py --corpus noxi --gpus 0,1
```

Reproduce combiner:

```bash
cd /work/ACM/mltac-main
/work/ACM/mltac-main/ACM/.venv-gpu/bin/python ACM/MoE/fit_noxi_metadata_head_combiner.py --corpus noxi
```

### NOXI-J

Best clean validation result:

```text
metadata-head audio-only expert
overall CCC: 0.638855
novice CCC: 0.552189
expert CCC: 0.619649
```

Completed metadata-head experts:

```text
ACM/MoE/experiments/noxi_moe1_noxi_j_metadata_head_experts/noxi_j_visual_videomae_dyadic_tcn_k11_metadata_head_seed13
ACM/MoE/experiments/noxi_moe1_noxi_j_metadata_head_experts/noxi_j_audio_w2vbert2_dyadic_tcn_k11_metadata_head_seed13
ACM/MoE/experiments/noxi_moe1_noxi_j_metadata_head_experts/noxi_j_text_xlm_roberta_dyadic_tcn_k11_metadata_head_seed13
```

Individual expert validation CCCs:

```text
visual_videomae:  0.423304
audio_w2vbert2:   0.638855
text_xlm_roberta: 0.434099
```

Combiner results did not beat audio-only:

```text
uniform: 0.607686
shared:  0.588724
role:    0.593064
```

Combiner summary:

```text
ACM/MoE/experiments/noxi_moe1_noxi_j_metadata_head_combiners/summary.csv
ACM/MoE/experiments/noxi_moe1_noxi_j_metadata_head_combiners/summary.json
ACM/MoE/experiments/noxi_moe1_noxi_j_metadata_head_combiners/weights.json
```

Reproduce experts:

```bash
cd /work/ACM/mltac-main
/work/ACM/mltac-main/ACM/.venv-gpu/bin/python ACM/MoE/run_noxi_metadata_head_experts_4gpu.py --corpus noxi_j --gpus 0,1
```

Reproduce combiner:

```bash
cd /work/ACM/mltac-main
/work/ACM/mltac-main/ACM/.venv-gpu/bin/python ACM/MoE/fit_noxi_metadata_head_combiner.py --corpus noxi_j
```

## Files To Keep

For every selected or reported expert, keep:

```text
config.json
model_best.pt
training_log.csv
metrics_overall.csv
metrics_by_dataset.csv
metrics_by_role.csv
metrics_by_session.csv
.complete
```

For combiners, keep:

```text
summary.csv
summary.json
weights.json
```

The large prediction/export files are regenerable from the kept checkpoint and config with `evaluate_noxi_metadata_head_checkpoint.py`.

## GPU Note

Use GPUs `0,1`. GPU id `2` reported CUDA unavailable for this branch.
