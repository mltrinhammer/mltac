# Handoff - PinSoRo MoE Class-3 Follow-up - 2026-06-10

Workspace:

```text
/work/ACM/mltac-main
```

## Current Best Overall Direction

Selected architecture before today:

```text
metadata-head modality experts + logit-space two_head combiner
```

Best same-setting baseline before smoothing:

| Domain | Branch/mode | Val mean kappa |
|---|---|---:|
| CC | metadata-head two_head | 0.322331 |
| CR | metadata-head two_head | 0.351985 |
| mean | same setting | 0.337158 |

Temporal smoothing/post-processing improved this substantially. Best original-model post-processing:

| Post-processing | CC mean | CR mean | Combined mean |
|---|---:|---:|---:|
| hysteresis 1.0 | 0.345265 | 0.376340 | 0.360802 |
| prob_mean window 121 | 0.342076 | 0.375392 | 0.358734 |
| logit_mean window 121 | 0.341311 | 0.375719 | 0.358515 |

Smoothing output root:

```text
ACM/MoE/experiments/moe1_temporal_smoothing_ablation/
```

## Error Analysis Finding

Error analysis script:

```text
ACM/MoE/error_analysis_moe1_metadata_head_two_head.py
```

Output root:

```text
ACM/MoE/experiments/moe1_error_analysis_metadata_head_two_head/
```

Main weakness:

```text
CR social is much weaker than CR task.
CR social class 3 is never predicted by the selected model.
```

CR social validation labels:

| Class | Support |
|---:|---:|
| 0 | 954 |
| 1 | 3077 |
| 2 | 11812 |
| 3 | 1452 |
| 4 | 0 |

Selected model CR social class 3:

```text
predicted count = 0
recall = 0.0
```

CC class 3 status:

```text
CC task class 3 is predicted reasonably.
CC social class 3 is weak but not fully collapsed.
CR social class 3 is fully collapsed.
```

## Train/Validation Class-3 Exposure

For CR social:

| Split | Windows | Windows with class 3 | Unique social frames | Unique class-3 frames |
|---|---:|---:|---:|---:|
| train_internal | 333 | 7 | 384143 | 712 |
| val_internal | 109 | 10 | 17295 | 1452 |

Train class-3 windows are only in:

```text
032 purple: 2 windows
045 purple: 2 windows
047 purple: 3 windows
```

Validation class-3 windows are in:

```text
018 purple: 2 windows
059 purple: 8 windows
```

This is a severe train/validation mismatch.

## Class-3 Attempts Run Today

A concise result summary is saved here:

```text
ACM/MoE/class3_training_ablation_summary_2026-06-10.md
```

### 1. Targeted CR Social Class3x10 Weighting

Output roots:

```text
ACM/MoE/experiments/moe1_cr_metadata_head_class3x10_experts/
ACM/MoE/experiments/moe1_cr_metadata_head_class3x10_combiners/
ACM/MoE/experiments/moe1_temporal_smoothing_ablation_cc_base_cr_class3x10/
```

Combiner result before smoothing:

| Mode | CR val mean kappa |
|---|---:|
| best_single | 0.331480 |
| shared | 0.386912 |
| two_head / role_head | 0.414482 |
| metadata_router | 0.396105 |

Best smoothing with original CC + class3x10 CR:

| Post-processing | CC mean | CR mean | Combined mean |
|---|---:|---:|---:|
| prob_mean window 121 | 0.342076 | 0.466485 | 0.404281 |
| logit_mean window 121 | 0.341311 | 0.465119 | 0.403215 |
| hysteresis 1.0 | 0.345265 | 0.459439 | 0.402352 |

But CR social class 3 remained collapsed:

```text
class3 predicted = 0
class3 recall = 0.0
```

Interpretation: this is a strong overall CR improvement, but it mostly improves class 2. Do not treat it as a class-3 fix.

### 2. Social Focal Loss Gamma=2

Output roots:

```text
ACM/MoE/experiments/moe1_cr_metadata_head_social_focal2_experts/
ACM/MoE/experiments/moe1_cr_metadata_head_social_focal2_combiners/
ACM/MoE/experiments/moe1_temporal_smoothing_ablation_cc_base_cr_social_focal2/
```

Combiner result before smoothing:

| Mode | CR val mean kappa |
|---|---:|
| best_single | 0.323645 |
| shared | 0.321139 |
| two_head / role_head | 0.362359 |
| metadata_router | 0.318496 |

Best smoothing with original CC + focal2 CR:

| Post-processing | CC mean | CR mean | Combined mean |
|---|---:|---:|---:|
| hysteresis 1.0 | 0.345265 | 0.384520 | 0.364893 |
| prob_mean window 121 | 0.342076 | 0.379767 | 0.360922 |
| logit_mean window 121 | 0.341311 | 0.380060 | 0.360685 |

CR social class 3 remained collapsed:

```text
class3 predicted = 0
class3 recall = 0.0
```

### 3. Simple CR Social Class-3 Window Oversampling 5x

Output roots:

```text
ACM/MoE/experiments/moe1_cr_metadata_head_class3_oversample5_experts/
ACM/MoE/experiments/moe1_cr_metadata_head_class3_oversample5_combiners/
ACM/MoE/experiments/moe1_temporal_smoothing_ablation_cc_base_cr_class3_oversample5/
```

Implementation:

```text
CR train windows containing social class 3 were duplicated to 5x exposure.
base_windows=333
class3_windows=7
expanded_windows=361
```

Combiner result before smoothing:

| Mode | CR val mean kappa |
|---|---:|
| best_single | 0.330342 |
| shared | 0.293122 |
| two_head / role_head | 0.302781 |
| metadata_router | 0.284470 |

Best smoothing with original CC + oversample5 CR:

| Post-processing | CC mean | CR mean | Combined mean |
|---|---:|---:|---:|
| prob_mean window 121 | 0.342076 | 0.317057 | 0.329567 |
| logit_mean window 121 | 0.341311 | 0.317499 | 0.329405 |
| hysteresis 1.0 | 0.345265 | 0.308615 | 0.326940 |

CR social class 3 remained collapsed:

```text
class3 predicted = 0
class3 recall = 0.0
```

Interpretation: simple 5x oversampling hurt CR and did not recover class 3.

## Code State / Cleanup

Temporary focal-loss and smoothing-root override hooks were removed after saving results. Scripts compile.

One small oversampling flag remains in the metadata-head trainer/runner because it was used for the oversample5 experiment:

```text
--cr-social-class3-oversample
```

Files touched/created during this thread include:

```text
ACM/MoE/error_analysis_moe1_metadata_head_two_head.py
ACM/MoE/ablate_moe1_temporal_smoothing.py
ACM/MoE/ablate_moe1_class_calibration.py
ACM/MoE/class3_training_ablation_summary_2026-06-10.md
ACM/MoE/hand_off_class3_next_2026-06-10.md
```

`model_last.pt` files were deleted from the new retrain roots. `model_best.pt`, logs, validation outputs, train-internal score exports, combiner summaries, and smoothing outputs were kept.

## Recommendation For Next Agent

The weighting, focal-loss, and oversampling experiments were targeted imbalance interventions. They were useful because they tested plausible explanations for the class-3 collapse. Since all three failed to recover class 3, the next step should be diagnostic separability analysis rather than another training-side imbalance variant.

Recommended next step: diagnostic separability analysis.

### Diagnostic 1: Class-3 Logit/Probability Separability

For CR social frames, using frozen selected experts or class3x10 experts:

```text
- distribution of class-3 logits/probabilities on true class-3 frames
- same distribution on true class-2 frames and other classes
- margin: logit_3 - max(other logits)
- rank of class 3 among logits
```

Key question:

```text
Does class 3 ever get close to winning?
```

### Diagnostic 2: Binary Class-3 Detector From Frozen Expert Logits

Train a tiny diagnostic classifier:

```text
Input: visual/audio/text logits or probabilities for CR social
Target: class3 vs not-class3
Fit split: train_internal
Eval split: val_internal
Models: logistic regression first, maybe tiny MLP second
```

If this works, class-3 signal exists but is suppressed by the 5-class head/combiner. Then consider a class-3 auxiliary/rescue head.

If this fails, class 3 may not be learnable from current frozen representations/features, or train/val class-3 sessions may be too different.

### Diagnostic 3: Class-3 Context/Session Inspection

Compare train class-3 sessions `032/045/047` with validation class-3 sessions `018/059`:

```text
- duration of class-3 segments
- neighboring labels before/after class 3
- whether class 3 is transitional between 2 and something else
- role/metadata/session differences
- visual/audio/text expert disagreement around class-3 frames
```

## Current Practical Best For Score

If the objective is score, not class-3 recovery, the best seen today is:

```text
CC original metadata-head two_head + CR class3x10 metadata-head two_head + prob_mean smoothing window 121
combined mean kappa = 0.404281
```

But this should be reported carefully because it does not solve class 3; it mainly improves CR class 2.
