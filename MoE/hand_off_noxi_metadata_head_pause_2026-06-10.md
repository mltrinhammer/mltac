# NOXI Metadata-Head MoE Pause Handoff - 2026-06-10

Workspace:

```text
/work/ACM/mltac-main
```

## What Changed

Added NOXI-specific metadata-head MoE scripts. PinSoRo scripts were not changed.

```text
ACM/MoE/prepare_noxi_metadata.py
ACM/MoE/train_noxi_metadata_head_tcn.py
ACM/MoE/evaluate_noxi_metadata_head_checkpoint.py
ACM/MoE/run_noxi_metadata_head_experts_4gpu.py
ACM/MoE/fit_noxi_metadata_head_combiner.py
```

Generated metadata tables:

```text
ACM/MoE/noxi_data/outputs/metadata/role_metadata.csv       # 152 rows
ACM/MoE/noxi_j_data/outputs/metadata/role_metadata.csv     # 102 rows
```

Metadata fields extracted from raw zips:

```text
age, gender, language
```

The model predicts two NOXI role outputs:

```text
novice, expert
```

## Completed NOXI

All three NOXI metadata-head experts completed:

```text
ACM/MoE/experiments/noxi_moe1_noxi_metadata_head_experts/noxi_visual_videomae_dyadic_tcn_k11_metadata_head_seed13/.complete
ACM/MoE/experiments/noxi_moe1_noxi_metadata_head_experts/noxi_audio_w2vbert2_dyadic_tcn_k11_metadata_head_seed13/.complete
ACM/MoE/experiments/noxi_moe1_noxi_metadata_head_experts/noxi_text_xlm_roberta_dyadic_tcn_k11_metadata_head_seed13/.complete
```

Best validation CCCs from logs:

```text
visual_videomae:   0.76042 at epoch 14
audio_w2vbert2:    0.81877 at epoch 28
text_xlm_roberta:  0.65625 at epoch 8
```

NOXI metadata-head combiner completed:

```text
ACM/MoE/experiments/noxi_moe1_noxi_metadata_head_combiners/summary.csv
```

Clean validation summary:

```text
best_single audio: 0.818758
uniform:           0.837678
shared:            0.856139
role:              0.855225
```

Best clean NOXI combiner is currently `shared` with CCC `0.856139`.

## NOXI-J Status At Pause

NOXI-J launcher was paused/stopped at the user's request. No active Python training processes remained after stopping.

Completed:

```text
ACM/MoE/experiments/noxi_moe1_noxi_j_metadata_head_experts/noxi_j_visual_videomae_dyadic_tcn_k11_metadata_head_seed13/.complete
ACM/MoE/experiments/noxi_moe1_noxi_j_metadata_head_experts/noxi_j_text_xlm_roberta_dyadic_tcn_k11_metadata_head_seed13/.complete
```

Best validation CCCs from logs:

```text
noxi_j visual_videomae:   0.42338 at epoch 15
noxi_j text_xlm_roberta:  0.43411 at epoch 52
```

NOXI-J text reached epoch 60, exported test predictions, and completed train diagnostics:

```text
split=train_internal turns=644 ccc=0.937515
```

Incomplete:

```text
noxi_j audio_w2vbert2
```

Audio was stopped during training:

```text
last logged epoch: 33
best logged val CCC: 0.61746 at epoch 29
```

There is no `.complete` marker for NOXI-J audio. Because the new trainer does not currently implement resume, the safest continuation is to rerun only NOXI-J audio from scratch:

```bash
cd /work/ACM/mltac-main
/work/ACM/mltac-main/ACM/.venv-gpu/bin/python ACM/MoE/run_noxi_metadata_head_experts_4gpu.py --corpus noxi_j --features audio_w2vbert2 --gpus 0
```

After NOXI-J audio completes, fit the NOXI-J combiner:

```bash
cd /work/ACM/mltac-main
/work/ACM/mltac-main/ACM/.venv-gpu/bin/python ACM/MoE/fit_noxi_metadata_head_combiner.py --corpus noxi_j
```

## GPU Note

GPU id `2` was not usable for these NOXI metadata-head jobs:

```text
RuntimeError: CUDA requested but not available.
```

Use only:

```text
--gpus 0,1
```

or a single known-good GPU:

```text
--gpus 0
```

## Guardrails

- Do not rerun completed NOXI experts unless explicitly requested.
- Do not touch unrelated running PinSoRo/MoE2 jobs.
- The metadata-head branch is NOXI-specific and isolated under `ACM/MoE`.
- NOXI-J audio is the only missing expert needed before the NOXI-J combiner can run.

## Completion Addendum

Resumed later on 2026-06-10. NOXI-J audio was rerun from scratch on GPU 0 and completed:

```text
ACM/MoE/experiments/noxi_moe1_noxi_j_metadata_head_experts/noxi_j_audio_w2vbert2_dyadic_tcn_k11_metadata_head_seed13/.complete
```

Final NOXI-J audio status:

```text
early_stop epoch=056 best_epoch=040 best_val_ccc=0.63885
split=train_internal turns=644 ccc=0.928422
```

NOXI-J metadata-head combiner was fitted:

```text
ACM/MoE/experiments/noxi_moe1_noxi_j_metadata_head_combiners/summary.csv
```

Clean NOXI-J validation summary:

```text
best_single audio: 0.638855
uniform:           0.607686
shared:            0.588724
role:              0.593064
```

Best clean NOXI-J result is currently the single audio metadata-head expert with CCC `0.638855`.
