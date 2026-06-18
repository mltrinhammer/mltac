# CR social class 3: problem note

Date: 2026-06-10

## Headline

CR social class 3 is the most problematic remaining class-specific failure in
the current PinSoRo MoE setup. The selected model never predicts it on
validation, even though the validation set contains 1452 CR social class-3
frames. Several targeted imbalance interventions did not recover class 3. The
strongest interpretation is that CR social class 3 is not represented
consistently across sessions by the current frozen experts, rather than being a
simple class-weighting issue.

Important clarification: class IDs are categorical. Class 2 and class 3 are not
"close" because of their numeric values. Any possible relation between them is
empirical/contextual: class-3 segments often occur next to class-2 segments in
the label timeline, and some interventions improved class-2-heavy behavior
without improving class 3.

## Current best model context

The current best architecture before the class-3 investigation was:

- metadata-head experts
- logit-space `two_head` combiner

Clean validation scores:

| Domain | Score |
|---|---:|
| CC | 0.322331 |
| CR | 0.351985 |
| Mean | 0.337158 |

Best temporal smoothing on the original model:

| Smoothing | CC | CR | Mean |
|---|---:|---:|---:|
| hysteresis 1.0 | 0.345265 | 0.376340 | 0.360802 |
| prob_mean 121 | 0.342076 | 0.375392 | 0.358734 |
| logit_mean 121 | 0.341311 | 0.375719 | 0.358515 |

## General class-3 picture

Class 3 is not uniformly broken across all settings.

- CC task class 3 is predicted reasonably.
- CC social class 3 is weak, but not fully collapsed.
- CR social class 3 is fully collapsed in the selected model: predicted count
  is zero on validation.

This matters because it argues against a global "class 3 is impossible"
conclusion. The failure is concentrated in CR social.

## CR social class-3 support

CR social validation support:

| Class | Frames |
|---:|---:|
| 0 | 954 |
| 1 | 3077 |
| 2 | 11812 |
| 3 | 1452 |
| 4 | 0 |

Selected model on CR social validation:

| Class | True frames | Predicted frames | Recall |
|---:|---:|---:|---:|
| 3 | 1452 | 0 | 0.0 |

Train/validation exposure is highly mismatched:

| Split | Windows | Class-3 windows | Class-3 sessions |
|---|---:|---:|---|
| train_internal | 333 | 7 | 032, 045, 047 |
| val_internal | 109 | 10 | 018, 059 |

Unique frame-level support:

| Split | Unique social frames | Unique class-3 frames | Class-3 share |
|---|---:|---:|---:|
| train_internal | 384143 | 712 | 0.19% |
| val_internal | 17295 | 1452 | 8.40% |

This is a severe mismatch: validation contains proportionally far more class 3
than training, and the class-3 validation sessions are different from the
training class-3 sessions.

## Targeted imbalance interventions tried

### 1. CR social class3x10 weighting

Output roots:

- `ACM/MoE/experiments/moe1_cr_metadata_head_class3x10_experts/`
- `ACM/MoE/experiments/moe1_cr_metadata_head_class3x10_combiners/`
- `ACM/MoE/experiments/moe1_temporal_smoothing_ablation_cc_base_cr_class3x10/`

Before smoothing:

| Combiner | CR score |
|---|---:|
| best_single | 0.331480 |
| shared | 0.386912 |
| two_head / role_head | 0.414482 |
| metadata_router | 0.396105 |

Best smoothing with original CC plus class3x10 CR:

| Smoothing | CC | CR | Mean |
|---|---:|---:|---:|
| prob_mean 121 | 0.342076 | 0.466485 | 0.404281 |
| logit_mean 121 | 0.341311 | 0.465119 | 0.403215 |
| hysteresis 1.0 | 0.345265 | 0.459439 | 0.402352 |

Outcome:

- Overall CR improved substantially.
- CR social class 3 still had predicted count 0 and recall 0.0.

Interpretation:

- The gain is not a class-3 recovery.
- The likely benefit is to other CR social behavior, especially the dominant
  class-2-heavy part of the problem.
- This is useful as a performance result but should not be presented as fixing
  class 3.

### 2. CR social focal loss, gamma=2

Output roots:

- `ACM/MoE/experiments/moe1_cr_metadata_head_social_focal2_experts/`
- `ACM/MoE/experiments/moe1_cr_metadata_head_social_focal2_combiners/`
- `ACM/MoE/experiments/moe1_temporal_smoothing_ablation_cc_base_cr_social_focal2/`

Before smoothing:

| Combiner | CR score |
|---|---:|
| best_single | 0.323645 |
| shared | 0.321139 |
| two_head / role_head | 0.362359 |
| metadata_router | 0.318496 |

Best smoothing:

| Smoothing | CC | CR | Mean |
|---|---:|---:|---:|
| hysteresis 1.0 | 0.345265 | 0.384520 | 0.364893 |
| prob_mean 121 | 0.342076 | 0.379769 | 0.360922 |
| logit_mean 121 | 0.341311 | 0.380058 | 0.360685 |

Outcome:

- Some smoothed CR improvement over the original clean model.
- CR social class 3 still had predicted count 0.

Interpretation:

- Focal loss did not expose a usable class-3 decision boundary.
- This makes a pure loss-shaping fix less promising.

### 3. CR social class-3 window oversampling, 5x

Output roots:

- `ACM/MoE/experiments/moe1_cr_metadata_head_class3_oversample5_experts/`
- `ACM/MoE/experiments/moe1_cr_metadata_head_class3_oversample5_combiners/`
- `ACM/MoE/experiments/moe1_temporal_smoothing_ablation_cc_base_cr_class3_oversample5/`

Implementation:

- Duplicate training windows containing CR social class 3.
- Base windows: 333
- Class-3 windows: 7
- Expanded windows: 361

Before smoothing:

| Combiner | CR score |
|---|---:|
| best_single | 0.330342 |
| shared | 0.293122 |
| two_head / role_head | 0.302781 |
| metadata_router | 0.284470 |

Best smoothing:

| Smoothing | CC | CR | Mean |
|---|---:|---:|---:|
| prob_mean 121 | 0.342076 | 0.317057 | 0.329567 |
| logit_mean 121 | 0.341311 | 0.317499 | 0.329405 |
| hysteresis 1.0 | 0.345265 | 0.308615 | 0.326940 |

Outcome:

- Oversampling hurt CR.
- CR social class 3 still had predicted count 0.

Interpretation:

- Simply showing the same seven class-3 training windows more often does not
  create validation-transferable class-3 behavior.
- This supports the session-mismatch hypothesis.

## Frozen-output binary diagnostic

Script:

- `ACM/MoE/diagnose_cr_social_class3_binary.py`

Output roots:

- `ACM/MoE/experiments/moe1_cr_social_class3_binary_diagnostic/`
- `ACM/MoE/experiments/moe1_cr_social_class3_binary_diagnostic_class3x10/`

This trained simple binary classifiers on frozen CR social expert logits/probs:
class 3 vs not class 3.

Best original-logits detector:

| Metric | Value |
|---|---:|
| Train rows | 384143 |
| Train positives | 712 |
| Validation rows | 17295 |
| Validation positives | 1452 |
| Average precision | 0.105649 |
| ROC-AUC | 0.603000 |
| Best F1 | 0.191076 |
| Best precision | 0.109441 |
| Best recall | 0.752066 |
| Predicted positives | 9978 |

Validation prevalence is 1452 / 17295 = 0.08395, so AP is only modestly above
prevalence.

Session behavior:

| Session | True class 3 | Predicted class 3 | Precision | Recall | Interpretation |
|---|---:|---:|---:|---:|---|
| 018 | 436 | 5204 | 0.0746 | 0.8899 | many false positives |
| 019 | 0 | 3771 | 0.0 | n/a | false-positive flood |
| 026 | 0 | 106 | 0.0 | n/a | some false positives |
| 059 | 1016 | 897 | 0.7848 | 0.6929 | works better |

Interpretation:

- Frozen expert outputs contain only weak class-3 signal.
- The signal is strongly session-dependent.
- A simple post-hoc class-3 rescue model would be unreliable: it works somewhat
  on session 059, but overfires badly on session 018 and on sessions with no
  class 3.

Class3x10 frozen logits were worse:

| Metric | Value |
|---|---:|
| Average precision | 0.070443 |
| ROC-AUC | 0.444924 |
| Best F1 | 0.157916 |

Interpretation:

- The class3x10 training run improved overall CR but made frozen class-3
  separability worse.
- Again, the class3x10 gain should not be described as class-3 recovery.

## Context diagnostic: segments, margins, and transitions

Script:

- `ACM/MoE/diagnose_cr_social_class3_context.py`

Output root:

- `ACM/MoE/experiments/moe1_cr_social_class3_context_diagnostic/`

Files:

- `session_context_summary.csv`
- `class3_segments.csv`
- `label_run_transitions.csv`

### Session-level result

Using the original CR metadata-head experts plus logit-space `two_head`
combiner:

| Split | Session | True class 3 | Pred class 3 | Class-3 recall | Mean class-3 rank | Mean class-3 probability |
|---|---:|---:|---:|---:|---:|---:|
| train | 032 | 51 | 458 | 1.0 | 1.0 | 0.930 |
| train | 045 | 100 | 286 | 1.0 | 1.0 | 0.993 |
| train | 047 | 561 | 1540 | 1.0 | 1.0 | 0.856 |
| val | 018 | 436 | 0 | 0.0 | 4.0 | 0.0033 |
| val | 059 | 1016 | 0 | 0.0 | 4.0 | 0.0064 |

Interpretation:

- Training class-3 frames are very easy for the current combiner.
- Validation class-3 frames are ranked around fourth on average and receive
  tiny class-3 probability.
- This is not just under-weighting. The validation class-3 examples look
  different to the model.

### Label-transition story

Class-3 segments:

| Split | Session | Segment length | Previous -> next label | Predicted as |
|---|---:|---:|---|---|
| train | 032 | 12 | 0 -> 2 | all class 3 |
| train | 032 | 39 | 2 -> 2 | all class 3 |
| train | 045 | 100 | 2 -> 2 | all class 3 |
| train | 047 | 561 | 2 -> 2 | all class 3 |
| val | 018 | 436 | 1 -> 2 | mostly 0/1 |
| val | 059 | 912 | 0 -> 2 | mostly 0, some 1/2 |
| val | 059 | 104 | 2 -> end | all 0 |

Transitions involving class 3:

| Split | Transition | Count |
|---|---|---:|
| train | 3 -> 2 | 4 |
| train | 2 -> 3 | 3 |
| train | 0 -> 3 | 1 |
| val | 3 -> 2 | 2 |
| val | 0 -> 3 | 1 |
| val | 1 -> 3 | 1 |
| val | 2 -> 3 | 1 |

Interpretation:

- Class 3 often appears next to class 2 in the label timeline.
- In training, most class-3 segments are islands inside or adjacent to class 2,
  and the model predicts them confidently.
- In validation, class-3 segments also often transition into class 2, but the
  model does not treat them like the training class-3 examples.
- Therefore, "class 2 vs class 3" is not a numeric similarity claim. It is a
  contextual observation: class 3 often appears beside class 2, and some
  performance changes may affect the class-2-heavy region without recovering
  class 3.

Consequence:

- It would be misleading to say the model simply confuses class 3 with class 2.
  In validation, session 018 class 3 is mostly predicted as 0/1, and session 059
  mostly as 0.
- The stronger conclusion is session mismatch or label-context mismatch.

## Leave-one-class3-session-out diagnostic

Script:

- `ACM/MoE/diagnose_cr_social_class3_loso.py`

Output roots:

- `ACM/MoE/experiments/moe1_cr_social_class3_loso_diagnostic/`
- `ACM/MoE/experiments/moe1_cr_social_class3_loso_diagnostic_probs/`
- `ACM/MoE/experiments/moe1_cr_social_class3_loso_diagnostic_logits_probs/`

This diagnostic trains a simple class-3-vs-not detector on all class-3 sessions
except one and evaluates on the held-out session. It tests whether class 3
transfers across sessions in the current frozen representation.

Logit-input results:

| Held-out session | Split | Class-3 frames | AP | ROC-AUC | Best F1 | Interpretation |
|---|---|---:|---:|---:|---:|---|
| 032 | train | 51 | 0.076 | 0.982 | 0.203 | separable ranking, poor precision |
| 045 | train | 100 | 0.002 | 0.340 | 0.006 | does not transfer |
| 047 | train | 561 | 0.116 | 0.914 | 0.238 | partly transfers |
| 018 | val | 436 | 0.052 | 0.562 | 0.111 | weak/noisy |
| 059 | val | 1016 | 0.768 | 0.426 | 0.812 | high F1 is misleading |

The 059 result needs caution:

- 1016 of 1488 frames are class 3.
- The best-F1 threshold predicts all 1488 frames as positive.
- This gives high F1 because class 3 dominates the session, but it is not a
  useful frame-level detector.
- ROC-AUC is poor.

The same broad story held for probability inputs and logits+probabilities.

Interpretation:

- Class-3 transfer is inconsistent even across training class-3 sessions.
- Validation session 018 does not transfer well.
- Validation session 059 is not cleanly detected; the apparent F1 gain can be
  achieved by predicting nearly everything in the session as class 3.

## Overall interpretation

The evidence points away from simple fixes:

- Not just class imbalance: weighting, focal loss, and oversampling did not
  recover class 3.
- Not just lack of any class-3 signal: the model predicts training class-3
  frames confidently.
- Not a simple class-2/class-3 ordinal confusion: class IDs are categorical, and
  validation class 3 is often predicted as 0/1 or 0, not mostly as 2.
- Strongest explanation: CR social class 3 is session-dependent in the current
  representation, and the train class-3 examples do not cover the validation
  class-3 variants.

Possible causes:

- Annotation inconsistency across sessions.
- Class 3 may encode context-dependent behavior not captured by the current
  features.
- The class may be transitional or local in the label timeline, but its visual,
  audio, or text signature differs by session.
- The validation class-3 sessions may contain a different subtype than the
  training class-3 sessions.

## Consequences for next steps

Recommended consequence:

- Do not spend more time on simple class-3 weighting, focal loss, or naive
  oversampling for CR social class 3 right now.
- Return to the broader original error-analysis plan and look for higher-value
  gains across CC/CR, task/social, and temporal stability.

If we later revisit CR social class 3:

1. Consider session-level or leave-one-session-out retraining only if we want to
   explicitly test whether class 3 is learnable when validation-style sessions
   are included in training.
2. Inspect raw annotations for sessions 018, 059, 032, 045, and 047 to check
   whether class 3 means the same thing across sessions.
3. Compare raw feature/behavior segments for class-3 regions, not just model
   scores.
4. Treat any class-3-specific post-hoc rescue as high risk unless it is
   validated session-wise, because the binary diagnostic overfires badly on
   some sessions.

Current practical stance:

- For overall model quality, the original metadata-head + logit `two_head`
  setup with temporal smoothing remains the clean baseline.
- The class3x10 run is interesting for overall CR performance, but should be
  described carefully because it does not solve class 3.
- CR social class 3 should be documented as an unresolved class-specific,
  session-dependent failure.
