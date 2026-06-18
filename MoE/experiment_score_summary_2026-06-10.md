# MoE Experiment Score Summary

Date: 2026-06-10

Scores below are mean combined kappa across CC/CR and task/social where available. The current clean/deployable line is **metadata-head + two_head + HMM no-prior**.

| Family | Setup | Mean combined kappa | Note |
| --- | --- | ---: | --- |
| HMM/prior | HMM + val-prior 0.25 | **0.3909** | Oracle diagnostic; uses validation label prior. |
| HMM | HMM no-prior strength 8 | **0.3853** | Current fair best. |
| HMM/prior | HMM + train-prior 0.25 | 0.3853 | Same as no-prior because train-to-train correction is zero. |
| HMM | HMM no-prior strength 12 | 0.3778 | Stronger smoothing; closer flip rate, lower kappa. |
| Temporal smoothing | Hysteresis 1.0 | 0.3608 | Previous fair best. |
| Temporal smoothing | Probability mean window 121 | 0.3587 |  |
| Temporal smoothing | Logit mean window 121 | 0.3585 |  |
| Architecture | Metadata-head + two_head | 0.3372 | Current raw base architecture. |
| Architecture | Metadata-head + shared | 0.3342 |  |
| Architecture | Metadata-head + metadata_router | 0.3304 |  |
| Architecture | Metadata-free + two_head | 0.3258 |  |
| Architecture | Metadata-head + prob_uniform | 0.2944 |  |
| Training imbalance | CR social focal + best smoothing | 0.3649 | Did not solve CR social class 3. |
| Training imbalance | CR class3 oversample 5x + best smoothing | 0.3296 | Poor; oversampling hurt. |
| Training imbalance | Class0 downweight 0.5 + best smoothing | 0.2548 | Poor. |
| Class calibration | Balanced CE + hysteresis | 0.3173 | Poor. |
| Class calibration | Uniform prior + hysteresis | 0.1441 | Poor. |

## Important Caveats

- `val-prior` and `val_*_upper` results are diagnostic/oracle-style results. They are useful for understanding headroom, but not final deployable settings.
- The fair current best is **metadata-head + two_head + HMM no-prior**, mean combined kappa **0.3853**.
- A saved `CR class3 weight 10x + prob_mean 121` row showed a higher headline score (`0.4043`), but it belongs to a targeted imbalance branch that should not be treated as a clean candidate without rechecking class behavior. We had already judged that line problematic because headline score could hide undesirable class tradeoffs.

## Current Interpretation

The strongest robust improvement came from temporal decoding. HMM/Viterbi uses the MoE framewise logits as emissions and training-label transition probabilities as a temporal prior. This reduced excessive prediction switching and improved the combined score substantially.

CR social class 2 appears to have weak but recoverable signal: the validation-prior diagnostic increased class-2 prediction/recall. CR social class 3 remains unresolved; the model almost never emits enough class-3 evidence for postprocessing to recover it.
