# Regime/Head Error Map - 2026-06-10

Class names:

```text
task:   0 goaloriented, 1 aimless, 2 adultseeking, 3 noplay
social: 0 solitary, 1 onlooker, 2 parallel, 3 associative, 4 cooperative
```

Evidence status:

```text
CC evidence: MoE2 horizon model, prob_shared + train-tuned hysteresis margin 0.4
CR evidence: MoE1 metadata-head model; MoE2 CR horizon run is not present yet
```

## Summary

The errors should be treated as four separate buckets, not one generic model problem:

| Bucket | Current evidence | Main failure | Priority |
|---|---:|---|---:|
| CC-task | MoE2 | `adultseeking` is mostly predicted as `noplay` | 3 |
| CC-social | MoE2 | `onlooker` and `associative` are barely recovered | 2 |
| CR-task | MoE1 | `aimless` recall is almost zero; `adultseeking` is over-called | 4 |
| CR-social | MoE1 | `parallel` and `associative` collapse into `solitary/onlooker` | 1 |

## CC-Task

Model: MoE2 `prob_shared` + hysteresis.

Headline:

```text
task kappa 0.45530
```

Class recalls:

```text
goaloriented  0.7564
aimless       0.6042
adultseeking  0.2266
noplay        0.6532
```

Confusion matrix:

```text
true goaloriented: 94464 -> goaloriented, 13991 -> aimless,  3963 -> adultseeking, 12475 -> noplay
true aimless:       4264 -> goaloriented, 11662 -> aimless,    38 -> adultseeking,  3338 -> noplay
true adultseeking:  3914 -> goaloriented,  2500 -> aimless,  6036 -> adultseeking, 14192 -> noplay
true noplay:        4677 -> goaloriented,  4675 -> aimless,  6202 -> adultseeking, 29301 -> noplay
```

Interpretation:

The dominant CC-task error is:

```text
adultseeking -> noplay: 14192 / 26642 = 53.3%
```

This looks like a recognition problem for adult-oriented behavior, not mainly temporal jitter. Hysteresis already helped temporal stability, but it does not make the visual-only model understand whether the child is seeking an adult.

Likely next levers:

- inspect adultseeking segments to see whether they are visually subtle, short-lived, or annotation-transition-heavy
- consider whether partner/adult orientation needs stronger features than the current partner-attention expert provides
- consider class-aware calibration/training for task class 2 only after checking whether the issue is session-local

## CC-Social

Model: MoE2 `prob_shared` + hysteresis.

Headline:

```text
social kappa 0.41117
```

Class recalls:

```text
solitary     0.5321
onlooker     0.0146
parallel     0.6013
associative  0.0556
cooperative  0.7667
```

Confusion matrix:

```text
true solitary:       7120 -> solitary, 102 -> onlooker, 2123 -> parallel, 117 -> associative,  3919 -> cooperative
true onlooker:       8756 -> solitary, 200 -> onlooker,  570 -> parallel, 1891 -> associative, 2318 -> cooperative
true parallel:       8137 -> solitary, 477 -> onlooker, 36557 -> parallel, 106 -> associative, 15517 -> cooperative
true associative:       0 -> solitary,   0 -> onlooker,   92 -> parallel, 150 -> associative, 2458 -> cooperative
true cooperative:    9749 -> solitary, 148 -> onlooker, 11292 -> parallel, 739 -> associative, 72078 -> cooperative
```

Interpretation:

The dominant CC-social failures are:

```text
onlooker -> solitary:       8756 / 13735 = 63.7%
associative -> cooperative: 2458 / 2700  = 91.0%
parallel -> cooperative:   15517 / 60794 = 25.5%
solitary -> cooperative:    3919 / 13381 = 29.3%
```

Hysteresis improves overall kappa and reduces over-switching, but it also commits to dominant social states. It does not recover rare/subtle states. `onlooker` and `associative` are the clearest CC-social targets.

Likely next levers:

- class-specific diagnostics for `onlooker` and `associative`
- check whether these errors are concentrated in session `005` or a small number of validation sessions
- try social-head-only class calibration or class-aware training before broad architecture changes
- be careful with stronger smoothing: it may further suppress short `onlooker`/`associative` runs

## CR-Task

Model: MoE1 metadata-head selected model. MoE2 CR is pending.

Headline:

```text
task kappa 0.56118
```

Class recalls:

```text
goaloriented  0.8351
aimless       0.0385
adultseeking  0.9723
noplay        0.8868
```

Confusion matrix:

```text
true goaloriented: 39590 -> goaloriented, 891 -> aimless, 1721 -> adultseeking,  5205 -> noplay
true aimless:       3418 -> goaloriented, 169 -> aimless,    0 -> adultseeking,   797 -> noplay
true adultseeking:     0 -> goaloriented,   0 -> aimless, 1262 -> adultseeking,    36 -> noplay
true noplay:         213 -> goaloriented,  27 -> aimless, 1148 -> adultseeking, 10869 -> noplay
```

Interpretation:

CR-task is not the highest-risk bucket because the overall kappa is comparatively strong. The main issue is that `aimless` is almost not recovered:

```text
aimless recall: 169 / 4384 = 3.85%
```

But `adultseeking` and `noplay` are strong. This looks like a class-boundary or class-prior problem rather than a general CR-task failure.

Likely next levers:

- defer until CR-social and CC-social are addressed
- if revisited, test task-head calibration or class weighting for `aimless`, not broad retraining

## CR-Social

Model: MoE1 metadata-head selected model. MoE2 CR is pending.

Headline:

```text
social kappa 0.14279
```

Class recalls:

```text
solitary     0.6184
onlooker     0.9834
parallel     0.0653
associative  0.0000
cooperative  no validation support
```

Confusion matrix:

```text
true solitary:      590 -> solitary,  364 -> onlooker, 0 -> parallel, 0 -> associative, 0 -> cooperative
true onlooker:       51 -> solitary, 3026 -> onlooker, 0 -> parallel, 0 -> associative, 0 -> cooperative
true parallel:     8667 -> solitary, 2374 -> onlooker, 771 -> parallel, 0 -> associative, 0 -> cooperative
true associative:  1145 -> solitary,  201 -> onlooker, 106 -> parallel, 0 -> associative, 0 -> cooperative
```

Interpretation:

This is the most severe bucket. CR-social has a structural collapse:

```text
parallel -> solitary:     8667 / 11812 = 73.4%
associative -> solitary:  1145 / 1452  = 78.9%
associative predicted:    0 frames
```

This is not just over-smoothing or temporal instability. The model does not learn a usable CR-social decision boundary for `parallel` and `associative`.

Likely next levers:

- run MoE2 horizon experts for CR before assuming MoE1 conclusions hold
- if MoE2 CR still collapses, focus on CR-social-specific class handling
- avoid using CC-social fixes blindly on CR-social; CR has a different label distribution and no cooperative validation support

## Recommended Next Experiment

Run the MoE2 horizon setup for CR, then repeat the same error analysis for CR.

Reason:

CR-social is the highest-risk bucket, but current CR evidence is still from MoE1. Before changing training, we need to know whether MoE2's long/partner horizon structure fixes any of the CR-social collapse.

Suggested order:

1. Train/evaluate MoE2 CR horizon experts using the same design as CC.
2. Fit the CR horizon combiner.
3. Tune hysteresis on CR `train_internal`, evaluate on CR `val_internal`.
4. Generate the same confusion/error map for CR.
5. Only then decide between class calibration, class weighting, or data/window changes.

Do not touch final test while doing this.
