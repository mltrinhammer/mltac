# CR social class-0 downweight experiment summary

Date: 2026-06-10

## Purpose

Test whether CR social class-0 dominance in training was suppressing class 3 or causing validation overprediction of class 0.

## Setup

MoE1 PinSoRo metadata-head experts were trained separately for CC and CR. CC used the clean metadata-head setup. CR used targeted CR-social loss weighting with:

- class 0 raw multiplier: 0.5
- class 2 raw multiplier: 1.0
- class 3 raw multiplier: 1.0
- focal gamma: 0.0
- class-3 oversampling: 1

The normalized CR-social effective weights were:

| Class | Effective weight |
|---:|---:|
| 0 | 0.7708297 |
| 1 | 1.5416595 |
| 2 | 1.5416595 |
| 3 | 1.5416595 |
| 4 | 0.0 |

## Outputs kept

- Expert configs, logs, training logs, and metric CSVs under:
  - `ACM/MoE/experiments/moe1_cc_metadata_head_class0down05_experts/`
  - `ACM/MoE/experiments/moe1_cr_metadata_head_class0down05_experts/`
- Combiner results under:
  - `ACM/MoE/experiments/moe1_cc_metadata_head_class0down05_combiners/`
  - `ACM/MoE/experiments/moe1_cr_metadata_head_class0down05_combiners/`
- Smoothing results under:
  - `ACM/MoE/experiments/moe1_temporal_smoothing_ablation_class0down05/`

Bulky checkpoints and per-frame exports were deleted after results were summarized because the experiment was not promising.

## Combiner results

| Domain | Mode | Validation mean kappa |
|---|---|---:|
| CC | two_head | 0.322331 |
| CR | two_head | 0.162511 |
| CR | best_single visual | 0.355419 |

The planned architecture was metadata-head experts plus logit-space `two_head`. Under that architecture, CR collapsed badly compared with the previous clean CR metadata-head `two_head` baseline of 0.351985.

## Smoothing results for two_head roots

Best combined smoothing settings:

| Smoothing | CC | CR | Combined |
|---|---:|---:|---:|
| prob_mean 121 | 0.342076 | 0.167568 | 0.254822 |
| hysteresis 1.0 | 0.345265 | 0.162179 | 0.253722 |
| logit_mean 121 | 0.341311 | 0.164713 | 0.253012 |

Smoothing did not rescue the CR two_head collapse.

## CR social class behavior

For CR social baseline/two_head:

| Class | Support | Predicted | Precision | Recall |
|---:|---:|---:|---:|---:|
| 0 | 954 | 9894 | 0.0395 | 0.4099 |
| 1 | 3077 | 1165 | 0.5983 | 0.2265 |
| 2 | 11812 | 6236 | 0.8063 | 0.4257 |
| 3 | 1452 | 0 | n/a | 0.0 |

CR social class 3 still did not appear. Class-0 downweighting did not solve the class-3 collapse and, under `two_head`, class 0 remained heavily overpredicted.

For the CR visual-only best single expert:

| Class | Support | Predicted |
|---:|---:|---:|
| 0 | 954 | 841 |
| 1 | 3077 | 3589 |
| 2 | 11812 | 12865 |
| 3 | 1452 | 0 |

The visual-only expert reduced class-0 overprediction, but still predicted zero class-3 frames.

## Interpretation

This experiment is another negative result for simple imbalance shaping. Lowering class-0 loss pressure did not recover CR social class 3. It also made the intended `two_head` architecture much worse for CR. The result supports returning to broader error analysis rather than continuing with simple class-weight/oversampling variants for CR social class 3.
