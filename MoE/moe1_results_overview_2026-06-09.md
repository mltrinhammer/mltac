# MoE 1 Results Overview - 2026-06-09

Workspace:

```text
/work/ACM/mltac-main
```

## Completed Pipeline

The first MoE pipeline completed for CC and CR with three modality experts:

```text
visual_videomae
audio_w2vbert2
text_xlm_roberta
```

All expert training, train-logit export, validation evaluation, and combiner
summaries are under:

```text
ACM/MoE/experiments/
```

## Main Clean Results

Clean results are train-fitted or fixed combiners evaluated on validation.
Validation-fitted upper-bound diagnostics are excluded from this table.

| Domain | Expert branch | Best clean mode | Val mean kappa | Main read |
|---|---:|---:|---:|---|
| CC | metadata-free | prob_uniform | 0.365412 | Best current CC branch. Probability averaging beat fitted routers. |
| CC | metadata-head | two_head | 0.322331 | Metadata-head underperformed metadata-free CC. |
| CR | metadata-free | shared | 0.301542 | Best metadata-free CR branch. |
| CR | metadata-head | two_head / role_head | 0.351985 | Metadata-head substantially improved CR. |

Recommended current branch by domain:

| Domain | Recommended branch | Recommended mode | Val mean kappa |
|---|---:|---:|---:|
| CC | metadata-free experts | prob_uniform | 0.365412 |
| CR | metadata-head experts | two_head or role_head | 0.351985 |

## Combiner Details

### CC Metadata-Free

Output root:

```text
ACM/MoE/experiments/moe1_cc_combiners/
```

| Mode | Fit split | Optimistic | Val mean kappa |
|---|---:|---:|---:|
| best_single | none | no | 0.295005 |
| uniform | fixed | no | 0.356911 |
| prob_uniform | fixed | no | 0.365412 |
| shared | train_internal | no | 0.351315 |
| two_head | train_internal | no | 0.351315 |
| role_head | train_internal | no | 0.358160 |
| metadata_router | train_internal | no | 0.328429 |
| val_shared_upper | val_internal | yes | 0.372685 |
| val_two_head_upper | val_internal | yes | 0.377526 |
| val_role_head_upper | val_internal | yes | 0.385218 |
| val_metadata_router_upper | val_internal | yes | 0.397706 |

### CC Metadata-Head

Output root:

```text
ACM/MoE/experiments/moe1_cc_metadata_head_combiners/
```

| Mode | Fit split | Optimistic | Val mean kappa |
|---|---:|---:|---:|
| best_single | none | no | 0.308760 |
| uniform | fixed | no | 0.298126 |
| prob_uniform | fixed | no | 0.287101 |
| shared | train_internal | no | 0.316614 |
| two_head | train_internal | no | 0.322331 |
| role_head | train_internal | no | 0.319747 |
| metadata_router | train_internal | no | 0.317805 |
| val_shared_upper | val_internal | yes | 0.309038 |
| val_two_head_upper | val_internal | yes | 0.304463 |
| val_role_head_upper | val_internal | yes | 0.322883 |
| val_metadata_router_upper | val_internal | yes | 0.337344 |

### CR Metadata-Free

Output root:

```text
ACM/MoE/experiments/moe1_cr_combiners/
```

| Mode | Fit split | Optimistic | Val mean kappa |
|---|---:|---:|---:|
| best_single | none | no | 0.284044 |
| uniform | fixed | no | 0.250159 |
| prob_uniform | fixed | no | 0.234642 |
| shared | train_internal | no | 0.301542 |
| two_head | train_internal | no | 0.300239 |
| role_head | train_internal | no | 0.300239 |
| metadata_router | train_internal | no | 0.281705 |
| val_shared_upper | val_internal | yes | 0.284939 |
| val_two_head_upper | val_internal | yes | 0.284259 |
| val_role_head_upper | val_internal | yes | 0.284259 |
| val_metadata_router_upper | val_internal | yes | 0.328270 |

### CR Metadata-Head

Output root:

```text
ACM/MoE/experiments/moe1_cr_metadata_head_combiners/
```

| Mode | Fit split | Optimistic | Val mean kappa |
|---|---:|---:|---:|
| best_single | none | no | 0.335494 |
| uniform | fixed | no | 0.316783 |
| prob_uniform | fixed | no | 0.301707 |
| shared | train_internal | no | 0.351779 |
| two_head | train_internal | no | 0.351985 |
| role_head | train_internal | no | 0.351985 |
| metadata_router | train_internal | no | 0.343002 |
| val_shared_upper | val_internal | yes | 0.345923 |
| val_two_head_upper | val_internal | yes | 0.336558 |
| val_role_head_upper | val_internal | yes | 0.336558 |
| val_metadata_router_upper | val_internal | yes | 0.338227 |

## Temporal Attention Router

Scaffold:

```text
ACM/MoE/fit_moe1_temporal_attention_router.py
```

Purpose:

```text
Train a small causal Transformer router over frozen visual/audio/text expert
logits. The router sees longer temporal context at the expert-output level and
outputs framewise modality weights.
```

Default output roots:

```text
ACM/MoE/experiments/moe1_cc_temporal_attention_router/
ACM/MoE/experiments/moe1_cr_temporal_attention_router/
```

Additional metadata-head roots:

```text
ACM/MoE/experiments/moe1_cc_metadata_head_temporal_attention_router/
ACM/MoE/experiments/moe1_cr_metadata_head_temporal_attention_router/
```

Completed temporal-router runs:

| Domain | Expert branch | Best epoch | Best val mean kappa | Baseline to beat | Result |
|---|---:|---:|---:|---:|---|
| CC | metadata-free | 1 | 0.318569 | 0.365412 | Worse than prob_uniform baseline. |
| CC | metadata-head | 8 | 0.316149 | 0.322331 | Slightly worse than metadata-head two_head baseline. |
| CR | metadata-free | 5 | 0.332041 | 0.301542 | Better than metadata-free shared baseline. |
| CR | metadata-head | 7 | 0.343595 | 0.351985 | Worse than metadata-head two_head / role_head baseline. |

Commands used:

```bash
CUDA_VISIBLE_DEVICES=0 /work/ACM/mltac-main/ACM/.venv-gpu/bin/python ACM/MoE/fit_moe1_temporal_attention_router.py --domain CC --device cuda
CUDA_VISIBLE_DEVICES=1 /work/ACM/mltac-main/ACM/.venv-gpu/bin/python ACM/MoE/fit_moe1_temporal_attention_router.py --domain CC --expert-root ACM/MoE/experiments/moe1_cc_metadata_head_experts --output-root ACM/MoE/experiments/moe1_cc_metadata_head_temporal_attention_router --device cuda
CUDA_VISIBLE_DEVICES=2 /work/ACM/mltac-main/ACM/.venv-gpu/bin/python ACM/MoE/fit_moe1_temporal_attention_router.py --domain CR --device cuda
CUDA_VISIBLE_DEVICES=3 /work/ACM/mltac-main/ACM/.venv-gpu/bin/python ACM/MoE/fit_moe1_temporal_attention_router.py --domain CR --expert-root ACM/MoE/experiments/moe1_cr_metadata_head_experts --output-root ACM/MoE/experiments/moe1_cr_metadata_head_temporal_attention_router --device cuda
```

Main read:

```text
The default temporal attention router helped CR metadata-free experts but did
not improve the best branch for either final domain choice. CC still favors the
simple metadata-free probability average. CR still favors metadata-head
two_head / role_head.
```

Guardrails:

```text
- CC and CR remain separate.
- The router trains only on train_internal expert scores.
- Validation remains held out for evaluation/model selection.
- Participant ID and session ID are not model inputs.
- The attention mask is causal.
- Experts stay frozen; this branch does not retrain modality experts.
```
