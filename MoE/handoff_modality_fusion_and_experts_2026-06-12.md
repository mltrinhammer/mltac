# PinSoRo Modality Fusion / Expert Experiments Handoff

Date: 2026-06-12

Workspace:

```text
/work/ACM/mltac-main
```

Main question:

```text
For a generalisable ACM architecture, should PinSoRo use late MoE modality
experts, early multimodal fusion, or a different expert story?
```

## Important Context

PinSoRo has two domains:

```text
CC
CR
```

and two heads:

```text
task
social
```

CR-social validation should be treated cautiously. The relevant valid CR-social
validation support is only:

| Class | Frames |
|---:|---:|
| 0 | 954 |
| 1 | 3077 |
| 2 | 11812 |
| 3 | 1452 |
| 4 | 0 |
| Total | 17295 |

CR-task has 65346 validation frames. So CR-social has only about 26.5 percent as
many valid validation frames as CR-task, and class 4 has no support. We still
report CR-social, but architecture selection should not depend heavily on it.

## Existing Baselines

### PinSoRo MoE1

Current fair clean/deployable line from earlier summaries:

```text
metadata-head + two_head + HMM no-prior
combined mean kappa: 0.3853
```

Head-level fair HMM row:

| Model | CC-task | CC-social | CR-task | CR-social | Mean incl. all | Mean excl. CR-social |
|---|---:|---:|---:|---:|---:|---:|
| MoE1 + HMM no-prior | 0.3938 | 0.3480 | 0.6531 | 0.1463 | 0.3853 | 0.4649 |

Files:

```text
ACM/MoE/experiment_score_summary_2026-06-10.md
ACM/MoE/experiments/moe1_hmm_prior_decoding_ablation/
```

### NOXI / NOXI-J Baselines

Different metric: CCC regression, not PinSoRo kappa.

| Dataset | Best model | Novice CCC | Expert CCC | Overall CCC |
|---|---|---:|---:|---:|
| NOXI | metadata-head shared combiner | 0.6752 | 0.8372 | 0.8561 |
| NOXI-J | metadata-head audio-only expert | 0.5522 | 0.6196 | 0.6389 |

Files:

```text
ACM/MoE/noxi_metadata_head_reproducibility_2026-06-10.md
/work/ACM/mltac-github-clean/ACM/noxi settings.txt
```

NOXI best settings note:

```text
dyadic_shared TCN
gated softmax fusion per modality
fusion channels 64
hidden channels 64
levels 4
kernel 5
dropout 0.2
modality dropout 0.1
window 500 / stride 125 at 25 Hz
CCC-only loss
```

## Experiments Run In This Thread

### 1. Early Fusion With NOXI Settings On PinSoRo

Purpose:

```text
Test whether NOXI-style early gated multimodal fusion works on PinSoRo.
```

Architecture:

```text
audio/text/visual projected per role
softmax-gated early fusion
dyadic shared TCN
NOXI-like levels=4, kernel=5
HMM smoothing
```

Results:

| Model | CC mean HMM | CR mean HMM | Average |
|---|---:|---:|---:|
| NOXI-settings early gated + HMM | 0.4089 | 0.2549 | 0.3319 |

Conclusion:

```text
The early-fusion idea transfers, but NOXI temporal settings are not optimal for
PinSoRo. It underperforms the current PinSoRo MoE1 + HMM line.
```

Files:

```text
ACM/MoE/experiments/pinsoro_noxi_settings_gated_fusion/
ACM/MoE/experiments/pinsoro_noxi_settings_gated_fusion_hmm/
```

### 2. Early Projected Concat vs Early Gated With MoE Temporal Settings

Purpose:

```text
Directly test whether PinSoRo benefits from early multimodal fusion, keeping
the MoE window/TCN settings fixed.
```

Architecture:

```text
audio features  -> projection to 64
text features   -> projection to 64
visual features -> projection to 64

concat mode:
  [audio_64 | text_64 | visual_64] = 192 channels per role

gated mode:
  softmax weighted sum to 64 channels per role

dyadic TCN:
  levels=5
  kernel=11
  hidden=64
  causal
  window=2400, stride=1200
```

Important naming clarification:

```text
"MoE-settings early concat" means MoE hyperparameters/windows/TCN/HMM were used.
It is not a MoE architecture. It is one early-fused multimodal model.
```

Raw results:

| Domain | Fusion | Task | Social | Mean |
|---|---|---:|---:|---:|
| CC | concat | 0.3252 | 0.4631 | 0.3941 |
| CC | gated | 0.3845 | 0.3231 | 0.3538 |
| CR | concat | 0.4054 | 0.0620 | 0.2337 |
| CR | gated | 0.4137 | 0.0842 | 0.2489 |

HMM results:

| Model | CC-task | CC-social | CC mean | CR-task | CR-social | CR mean | Avg incl. all | Avg excl. CR-social |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Early gated + MoE settings + HMM | 0.4353 | 0.3469 | 0.3911 | 0.4496 | 0.0926 | 0.2711 | 0.3311 | 0.4240 |
| Early concat + MoE settings + HMM | 0.3448 | 0.5236 | 0.4342 | 0.5850 | 0.0589 | 0.3219 | 0.3781 | 0.4845 |

Conclusion:

```text
Early projected concat is the strongest early-fusion variant.
When CR-social is included, it is slightly below MoE1 + HMM.
When CR-social is excluded, it beats MoE1 + HMM:

early concat excl. CR-social: 0.4845
MoE1 + HMM excl. CR-social: 0.4649
```

Files:

```text
ACM/MoE/pinsoro_noxi_settings/train_gated_fusion.py
ACM/MoE/experiments/pinsoro_moe_settings_early_fusion/
ACM/MoE/experiments/pinsoro_moe_settings_early_fusion_hmm_concat/
ACM/MoE/experiments/pinsoro_moe_settings_early_fusion_hmm_gated/
```

### 3. Two-Horizon Experts: Short + Long Early-Concat

Purpose:

```text
Try to preserve an "expert" story while keeping early multimodal fusion.
Expert meanings:
  short-horizon early-concat expert
  long-horizon early-concat expert
```

Short expert:

```text
early projected concat
levels=3
kernel=5
hidden=64
window still 2400 / stride 1200
```

Long expert:

```text
existing early projected concat with MoE settings
levels=5
kernel=11
hidden=64
```

Combiner variants:

```text
fixed 50/50 logit average
validation-grid diagnostic weights
train-grid learned weights from frozen train/internal logits
```

Files added:

```text
ACM/MoE/pinsoro_noxi_settings/combine_horizon_experts.py
ACM/MoE/pinsoro_noxi_settings/export_gated_fusion_checkpoint.py
```

Short-horizon standalone behavior:

```text
Short-horizon standalone was weak.
CR best raw validation score was around 0.2074.
CC best raw validation score reached around 0.2832 before export.
```

#### Fixed 50/50

| Model | CC mean HMM | CR mean HMM | Avg incl. all | Avg excl. CR-social |
|---|---:|---:|---:|---:|
| Two-horizon fixed 50/50 + HMM | 0.4337 | 0.3134 | 0.3736 | 0.4744 |

Compared with single long early concat:

```text
Long early concat + HMM avg incl all:     0.3781
Two-horizon fixed + HMM avg incl all:     0.3736

Long early concat + HMM excl CR-social:   0.4845
Two-horizon fixed + HMM excl CR-social:   0.4744
```

Conclusion:

```text
Fixed 50/50 does not improve over single long early concat.
```

Files:

```text
ACM/MoE/experiments/pinsoro_two_horizon_early_concat/
ACM/MoE/experiments/pinsoro_two_horizon_early_concat_combiners/cc_fixed_average/
ACM/MoE/experiments/pinsoro_two_horizon_early_concat_combiners/cr_fixed_average/
ACM/MoE/experiments/pinsoro_two_horizon_early_concat_hmm_fixed_average/
```

#### Validation-Grid Diagnostic

This is not fair/deployable because weights are chosen on validation, but it
measures headroom.

Validation-selected short weights:

| Domain | Task short weight | Social short weight |
|---|---:|---:|
| CC | 0.35 | 0.00 |
| CR | 0.20 | 1.00 |

Results:

| Model | CC mean HMM | CR mean HMM | Avg incl. all |
|---|---:|---:|---:|
| Two-horizon val-grid diagnostic + HMM | 0.4487 | 0.3165 | 0.3826 |

Conclusion:

```text
There is some complementary signal, especially for CC, but the validation-grid
setup is diagnostic only and overfits validation.
```

Files:

```text
ACM/MoE/experiments/pinsoro_two_horizon_early_concat_combiners/cc_val_grid/
ACM/MoE/experiments/pinsoro_two_horizon_early_concat_combiners/cr_val_grid/
ACM/MoE/experiments/pinsoro_two_horizon_early_concat_hmm_val_grid/
```

#### Train-Grid Learned Weights

Train/internal logits were exported from frozen short and long experts, then
weights were grid-searched on train/internal labels and applied to validation.

Learned short weights:

| Domain | Task short weight | Social short weight |
|---|---:|---:|
| CC | 0.30 | 0.40 |
| CR | 0.70 | 0.05 |

Results:

| Model | CC mean HMM | CR mean HMM | Avg incl. all | Avg excl. CR-social |
|---|---:|---:|---:|---:|
| Two-horizon train-grid + HMM | 0.4422 | 0.2772 | 0.3597 | 0.4596 |

Conclusion:

```text
Train-learned weights helped CC relative to fixed 50/50 but hurt CR badly,
especially CR-task. It is worse than single long early concat and worse than
fixed 50/50 when CR-task is considered.

The current short-horizon expert is not useful enough.
```

Files:

```text
ACM/MoE/experiments/pinsoro_two_horizon_early_concat_combiners/cc_train_grid/
ACM/MoE/experiments/pinsoro_two_horizon_early_concat_combiners/cr_train_grid/
ACM/MoE/experiments/pinsoro_two_horizon_early_concat_hmm_train_grid/
```

## Current Best Interpretation

The strongest clean PinSoRo direction is:

```text
single long-horizon early projected concat
+ MoE temporal settings
+ HMM smoothing
```

This is especially true if CR-social validation is downweighted or excluded as a
model-selection target.

Key comparison:

| Model | Mean incl. CR-social | Mean excl. CR-social |
|---|---:|---:|
| MoE1 + HMM | 0.3853 | 0.4649 |
| Early concat + MoE settings + HMM | 0.3781 | 0.4845 |
| Two-horizon fixed 50/50 + HMM | 0.3736 | 0.4744 |
| Two-horizon train-grid + HMM | 0.3597 | 0.4596 |

Main conclusions:

```text
1. Early fusion is strongly supported.
2. Early concat is better than early gated for PinSoRo under MoE settings.
3. Late modality experts are no longer clearly the best story, especially if
   CR-social validation is not trusted.
4. The short/long horizon expert story was attractive, but this first short
   expert did not add robust fair performance.
5. HMM smoothing remains very important.
6. CR-social remains a special diagnostic/problem area, not a stable selection
   target.
```

## Suggested Next Steps

### Best Next Expert Story: Task/Social Specialists

Recommended next experiment:

```text
Expert 1: early-concat task-specialist
Expert 2: early-concat social-specialist
Combiner/HMM after logits
```

Why:

```text
- Keeps early multimodal fusion, which is currently strongest.
- Gives a clean MoE story based on behavioral target, not modality.
- Task and social heads behave very differently.
- CR-task is strong and CR-social is sparse/weird, so a single shared model may
  compromise between incompatible supervision signals.
```

Possible implementation:

```text
Train two early-concat models with same architecture/windows:

task-specialist:
  loss mainly or only task head
  social head optional but low/zero loss weight

social-specialist:
  loss mainly or only social head
  task head optional but low/zero loss weight

Then combine:
  use task logits from task specialist
  use social logits from social specialist
  optionally compare with learned/shared combiner
  apply HMM
```

This is more defensible than modality experts because modalities appear to want
early integration, while target heads appear genuinely different.

### Other Possible Expert Ideas

#### Domain Specialists

```text
CC specialist early-concat
CR specialist early-concat
```

This is less novel because most PinSoRo runs already train CC and CR separately,
but it is valid if the final architecture is trained separately per dataset/domain.

#### Role-Separated Directed Interaction Expert

Only meaningful if role streams stay separate until cross-role interaction:

```text
child multimodal encoder -> child states
adult multimodal encoder -> adult states

child queries adult history -> adult-to-child influence
adult queries child history -> child-to-adult influence
```

This could answer "who responds to whom" better than generic dyadic hidden
states. A generic dyadic fused hidden state is not interpretable enough for this.

#### Uncertainty / Disagreement Expert

This fits the PinSoRo annotator-disagreement story:

```text
standard early-concat expert
uncertainty/disagreement-sensitive expert
```

Potentially train with soft labels or confidence-weighted loss. This may be
useful because current MoE1 is also being tested with uncertainty due annotator
disagreement.

#### Temporal Stability Expert

Since HMM gives large gains, another story is:

```text
framewise early-concat expert
transition/stability-aware expert
```

But be careful: if this is just another way of doing HMM, it may not add enough
architectural clarity.

## Practical Notes For Next Agent

Modified/added files in this thread:

```text
ACM/MoE/pinsoro_noxi_settings/train_gated_fusion.py
ACM/MoE/ablate_gated_fusion_hmm_decoding.py
ACM/MoE/pinsoro_noxi_settings/combine_horizon_experts.py
ACM/MoE/pinsoro_noxi_settings/export_gated_fusion_checkpoint.py
```

Useful result directories:

```text
ACM/MoE/experiments/pinsoro_moe_settings_early_fusion/
ACM/MoE/experiments/pinsoro_moe_settings_early_fusion_hmm_concat/
ACM/MoE/experiments/pinsoro_moe_settings_early_fusion_hmm_gated/
ACM/MoE/experiments/pinsoro_two_horizon_early_concat/
ACM/MoE/experiments/pinsoro_two_horizon_early_concat_combiners/
ACM/MoE/experiments/pinsoro_two_horizon_early_concat_hmm_fixed_average/
ACM/MoE/experiments/pinsoro_two_horizon_early_concat_hmm_val_grid/
ACM/MoE/experiments/pinsoro_two_horizon_early_concat_hmm_train_grid/
```

All training/export/HMM processes from this thread were confirmed stopped before
this handoff was written.

