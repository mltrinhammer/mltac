# MoE2 Horizon Expert Experiment Summary

Date: 2026-06-10

## Purpose

MoE2 tested whether the expert axis should represent **time horizons** rather than modalities. All experts used only `visual_videomae`; the three experts were:

| Expert | Model | Intended role |
|---|---|---|
| short | dyadic TCN, kernel 3, levels 3 | local/short context |
| long | dyadic TCN, kernel 15, levels 6 | long context, about 1 minute receptive field |
| partner | attention model, kernel 7, levels 4 | partner-interaction context |

This was compared against the modality-expert direction. Final conclusion: **MoE2 is useful as an ablation, but should not be the main path. Modality experts are the better direction.**

## Selected Combiner Weights

| Domain | Selected combiner | Combine space | short | long | partner | Val mean kappa |
|---|---|---:|---:|---:|---:|---:|
| CC | `prob_shared` | probability | 0.00 | 0.50 | 0.50 | 0.39762 |
| CR | `shared` | logit | 0.10 | 0.90 | 0.00 | 0.25432 |

Interpretation:

- CC used both long context and partner context; the short expert was ignored.
- CR almost entirely collapsed to the long expert; partner was ignored.
- This is the central reason MoE2 is not compelling as the final direction: the horizon split helps CC but does not solve CR.

## Domain/Head Results

Kappa by domain and head for the selected combiner, before and after train-tuned hysteresis:

| Domain | Variant | Task kappa | Social kappa | Mean kappa |
|---|---|---:|---:|---:|
| CC | selected combiner | 0.41368 | 0.38157 | 0.39762 |
| CC | + hysteresis margin 0.4 | 0.45530 | 0.41117 | 0.43323 |
| CR | selected combiner | 0.37836 | 0.13027 | 0.25432 |
| CR | + hysteresis margin 0.5 | 0.38699 | 0.13101 | 0.25900 |

Interpretation:

- Hysteresis is valuable for CC because it reduces excessive temporal switching and improves both heads.
- Hysteresis barely helps CR, which indicates CR is not mainly a temporal-jitter problem.
- CR-social remains the hard failure.

## Expert Ablations By Head

### CC

| Model | Task kappa | Social kappa | Mean |
|---|---:|---:|---:|
| short expert | 0.27466 | 0.18778 | 0.23122 |
| long expert | 0.36740 | 0.32604 | 0.34672 |
| partner expert | 0.30780 | 0.34567 | 0.32673 |
| selected combiner | 0.41368 | 0.38157 | 0.39762 |
| selected + hysteresis | 0.45530 | 0.41117 | 0.43323 |

CC takeaway: horizon ensembling worked. The long expert was strongest for task, partner was strongest for social, and combining them improved both.

### CR

| Model | Task kappa | Social kappa | Mean |
|---|---:|---:|---:|
| short expert | 0.13971 | 0.19533 | 0.16752 |
| long expert | 0.38202 | 0.12468 | 0.25335 |
| partner expert | 0.32009 | 0.06158 | 0.19084 |
| selected combiner | 0.37836 | 0.13027 | 0.25432 |
| selected + hysteresis | 0.38699 | 0.13101 | 0.25900 |

CR takeaway: the horizon MoE did not create useful complementarity. The selected combiner mostly reverted to the long expert.

## Notable Combiner Ablations

| Domain | Mode | Mean kappa | Notes |
|---|---|---:|---|
| CC | best single expert | 0.34672 | long expert |
| CC | uniform logits | 0.36195 | simple ensemble helps |
| CC | shared logits | 0.39459 | learned shared weights |
| CC | prob_shared | 0.39762 | best clean combiner |
| CC | prob_shared + hysteresis | 0.43323 | best MoE2 CC result |
| CR | best single expert | 0.25335 | long expert |
| CR | uniform logits | 0.24204 | hurts |
| CR | shared logits | 0.25432 | best clean combiner, only tiny gain |
| CR | prob_shared | 0.25334 | essentially same as long |
| CR | shared + hysteresis | 0.25900 | tiny gain |

## Class-Level Error Map

Class names:

| Head | Class IDs |
|---|---|
| task | 0 goaloriented, 1 aimless, 2 adultseeking, 3 noplay |
| social | 0 solitary, 1 onlooker, 2 parallel, 3 associative, 4 cooperative |

### CC-Task

MoE2 + hysteresis class recalls:

| Class | Recall |
|---|---:|
| goaloriented | 0.75636 |
| aimless | 0.60419 |
| adultseeking | 0.22656 |
| noplay | 0.65324 |

Main failure:

```text
adultseeking -> noplay: 14192 / 26642 = 53.3%
```

Interpretation: CC-task still struggles to recognize adult-oriented behavior from VideoMAE horizon context.

### CC-Social

MoE2 + hysteresis class recalls:

| Class | Recall |
|---|---:|
| solitary | 0.53210 |
| onlooker | 0.01456 |
| parallel | 0.60133 |
| associative | 0.05556 |
| cooperative | 0.76674 |

Main failures:

```text
onlooker -> solitary:       8756 / 13735 = 63.7%
associative -> cooperative: 2458 / 2700  = 91.0%
```

Interpretation: CC-social rare/subtle states are not recovered. Hysteresis improves temporal stability but does not fix rare-class recognition.

### CR-Task

MoE2 + hysteresis class recalls:

| Class | Recall |
|---|---:|
| goaloriented | 0.65225 |
| aimless | 0.11291 |
| adultseeking | 0.95532 |
| noplay | 0.86326 |

Main failure:

```text
aimless recall: 495 / 4384 = 11.3%
```

Interpretation: CR-task is weaker than MoE1 task and mainly misses `aimless`, but CR-task is not the main blocker.

### CR-Social

MoE2 + hysteresis class recalls:

| Class | Recall |
|---|---:|
| solitary | 0.57652 |
| onlooker | 0.65161 |
| parallel | 0.18888 |
| associative | 0.00000 |
| cooperative | no validation support |

Main failures:

```text
parallel -> solitary/onlooker:     9581 / 11812 = 81.1%
associative -> solitary/onlooker:  1340 / 1452  = 92.3%
associative predicted:            0 frames
```

Interpretation: CR-social is a structural class discrimination failure. The horizon-only VideoMAE setup does not solve it.

## Temporal Stability

MoE2 baseline still over-switched relative to true labels, but this differed by domain.

| Domain | Head | Variant | True flips | Pred flips | Pred/true ratio |
|---|---|---|---:|---:|---:|
| CC | task | baseline | 148 | 2629 | 17.76x |
| CC | task | hysteresis | 148 | 349 | 2.36x |
| CC | social | baseline | 88 | 1770 | 20.11x |
| CC | social | hysteresis | 88 | 197 | 2.24x |
| CR | task | baseline | 25 | 653 | 26.12x |
| CR | task | hysteresis | 25 | 354 | 14.16x |
| CR | social | baseline | 6 | 167 | 27.83x |
| CR | social | hysteresis | 6 | 90 | 15.00x |

Interpretation:

- CC benefits strongly from hysteresis.
- CR remains poor even after reducing switches, so its main issue is not only temporal noise.

## Regularization Ablation

An exploratory CC long-expert regularization ablation was run because the original long expert peaked late. These runs were **not adopted as final MoE2**, because we decided the final direction should return to modality experts.

| Variant | LR | Weight decay | Dropout | Best epoch | Best val mean kappa |
|---|---:|---:|---:|---:|---:|
| original long expert | 0.001 | 0.0001 | 0.2 | 35 | 0.34672 |
| lower_lr | 0.0005 | 0.0001 | 0.2 | 43 | 0.37168 |
| more_weight_decay | 0.001 | 0.0003 | 0.2 | 47 | 0.37239 |
| more_dropout | 0.001 | 0.0001 | 0.3 | 57 | 0.38794 |
| combined_regularized | 0.0005 | 0.0003 | 0.3 | 51 | 0.38020 |

Interpretation:

- Regularization helped the CC long expert.
- The best peak was late, which raised reasonable validation-overfitting concerns.
- This was not pursued further because CR remained unsolved and the horizon-only design was not the right final direction.

## Final Conclusion

MoE2 answered the design question:

```text
Time-horizon experts are useful for CC, especially with hysteresis.
Time-horizon experts do not solve CR, especially CR-social.
```

The final modeling focus should return to **modality experts**, likely with:

1. modality diversity retained for CR,
2. hysteresis or another temporal-stability layer after prediction,
3. CR-social-specific treatment for `parallel` and `associative`,
4. no final-test use while iterating.

MoE2 should be treated as a completed ablation, not the main final model.
