# CR Social Class-3 Training Ablations - 2026-06-10

Selected baseline before these tests:

```text
CC: original metadata-head two_head
CR: original metadata-head two_head
Best smoothing/post-processing: hysteresis 1.0
Combined mean kappa: 0.360802
```

## Experiments Run

### 1. CR metadata-head targeted class3x10

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

Best smoothing result with original CC + class3x10 CR:

| Post-processing | CC mean | CR mean | Combined mean |
|---|---:|---:|---:|
| prob_mean window 121 | 0.342076 | 0.466485 | 0.404281 |
| logit_mean window 121 | 0.341311 | 0.465119 | 0.403215 |
| hysteresis 1.0 | 0.345265 | 0.459439 | 0.402352 |

CR social class 3 remained collapsed:

```text
class3 predicted = 0
class3 recall = 0.0
```

The gain mainly came from better CR social class 2 behavior, not class 3.

### 2. CR metadata-head social focal loss gamma=2

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

Best smoothing result with original CC + focal2 CR:

| Post-processing | CC mean | CR mean | Combined mean |
|---|---:|---:|---:|
| hysteresis 1.0 | 0.345265 | 0.384520 | 0.364893 |
| prob_mean window 121 | 0.342076 | 0.379767 | 0.360922 |
| logit_mean window 121 | 0.341311 | 0.380060 | 0.360685 |

CR social class 3 also remained collapsed:

```text
class3 predicted = 0
class3 recall = 0.0
```

## Interpretation

The class3x10 targeted-weight run is a strong overall CR improvement, but it does not solve the intended class-3 failure. Since it mainly improves class 2, do not treat it as a class-3 fix. The focal-loss run is only a small improvement over the original and also does not recover class 3.

Next class-3 discussion should focus on approaches that force actual class-3 representation or sampling, such as class-3 window oversampling, binary class-3 auxiliary/rescue objective, or inspecting whether class-3 labels are concentrated in specific sessions/contexts.


### 3. CR metadata-head social class-3 window oversampling 5x

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

Best smoothing result with original CC + oversample5 CR:

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

This simple 5x oversampling hurt CR performance and did not recover class 3.
