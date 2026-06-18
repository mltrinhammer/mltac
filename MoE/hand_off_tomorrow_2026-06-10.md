# Handoff For Tomorrow - PinSoRo MoE 1 - 2026-06-10

Workspace:

```text
/work/ACM/mltac-main
```

## What Completed

The first PinSoRo MoE pipeline completed successfully for CC and CR:

```text
visual_videomae
audio_w2vbert2
text_xlm_roberta
```

The overnight run trained metadata-free experts, fitted combiners, trained
metadata-head experts, and fitted metadata-head combiners. Afterward, a temporal
attention router over frozen expert logits was scaffolded and run.

Main overview file:

```text
ACM/MoE/moe1_results_overview_2026-06-09.md
```

No GPU jobs are currently running.

## Best Current Results

Clean validation results only; validation-fitted upper-bound diagnostics are not
main results.

| Domain | Best branch | Best mode | Val mean kappa |
|---|---:|---:|---:|
| CC | metadata-free experts | prob_uniform | 0.365412 |
| CR | metadata-head experts | two_head / role_head | 0.351985 |

Main interpretation:

```text
CC prefers the simple metadata-free probability average.
CR benefits from metadata-head experts, with two_head/role_head routing best.
```

Metadata is not universally helpful: it hurt CC but helped CR.

## Completed Output Roots

```text
ACM/MoE/experiments/moe1_cc_experts/
ACM/MoE/experiments/moe1_cc_combiners/
ACM/MoE/experiments/moe1_cr_experts/
ACM/MoE/experiments/moe1_cr_combiners/
ACM/MoE/experiments/moe1_cc_metadata_head_experts/
ACM/MoE/experiments/moe1_cc_metadata_head_combiners/
ACM/MoE/experiments/moe1_cr_metadata_head_experts/
ACM/MoE/experiments/moe1_cr_metadata_head_combiners/
```

Temporal attention router roots:

```text
ACM/MoE/experiments/moe1_cc_temporal_attention_router/
ACM/MoE/experiments/moe1_cc_metadata_head_temporal_attention_router/
ACM/MoE/experiments/moe1_cr_temporal_attention_router/
ACM/MoE/experiments/moe1_cr_metadata_head_temporal_attention_router/
```

## Temporal Attention Router Outcome

Script:

```text
ACM/MoE/fit_moe1_temporal_attention_router.py
```

Best validation scores from `training_log.csv`:

| Domain | Expert branch | Best epoch | Best val mean kappa | Baseline to beat | Outcome |
|---|---:|---:|---:|---:|---|
| CC | metadata-free | 1 | 0.318569 | 0.365412 | Worse |
| CC | metadata-head | 8 | 0.316149 | 0.322331 | Slightly worse |
| CR | metadata-free | 5 | 0.332041 | 0.301542 | Better |
| CR | metadata-head | 7 | 0.343595 | 0.351985 | Worse |

The attention router is promising for CR metadata-free, but it does not change
the current best final domain choices.

Important implementation note:

```text
The router script was patched after the CR runs so future runs reload
model_best.pt before writing the final summary. For tonight's CR attention
router results, trust the best rows in training_log.csv, not the final summary
field if it differs.
```

## Related Handoffs

NoXi transfer handoff:

```text
ACM/MoE/hand_off_noxi_from_pinsoro_moe_2026-06-09.md
```

Original MoE 1 handoff:

```text
ACM/MoE/hand_off_moe1_2026-06-09.md
```

## Suggested Next Steps

1. Read `ACM/MoE/moe1_results_overview_2026-06-09.md`.
2. Confirm the best branch per domain from the `comparison.json` files.
3. Decide whether the final submission/evaluation path should use:

```text
CC: metadata-free prob_uniform
CR: metadata-head two_head or role_head
```

4. If continuing router work, prioritize low-risk CR metadata-free temporal
   router variants:

```text
- probability inputs instead of raw logits
- stronger smoothness penalty
- fewer layers / smaller hidden dimension
- early stopping on validation mean kappa
```

5. Do not mix CC and CR normalizers, experts, routers, or metadata statistics.
6. Do not use participant ID or session ID as model features.

## Guardrails

The repository has unrelated dirty/untracked files from earlier work. Do not
revert or delete anything outside the MoE task without explicit instruction.

All new PinSoRo MoE work should stay under:

```text
ACM/MoE/
```
