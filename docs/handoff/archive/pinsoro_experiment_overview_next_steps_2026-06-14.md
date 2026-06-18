# PinSoRo Experiment Overview And Next Steps

Date: 2026-06-14

## Current Score Reference

Scores are validation kappa means across CC/CR and task/social unless stated
otherwise.

| Family | Setting | CC task | CC social | CR task | CR social | Mean |
|---|---:|---:|---:|---:|---:|---:|
| MoE1 metadata-head + two_head + HMM | fair clean baseline | 0.3938 | 0.3480 | 0.6531 | 0.1463 | 0.3853 |
| Early concat + MoE temporal settings + HMM | single early-fusion model | 0.3448 | 0.5236 | 0.5850 | 0.0589 | 0.3781 |
| Two-horizon early concat fixed 50/50 + HMM | short+long experts | - | - | - | - | 0.3736 |
| Two-horizon early concat train-grid + HMM | train-learned short/long weights | - | - | - | - | 0.3597 |
| Person interaction linear + HMM | early concat, shared person, post-logit partner linear | 0.4780 | 0.3580 | 0.4913 | 0.1180 | 0.3613 |

Mean excluding CR-social:

| Family | Mean excl. CR-social |
|---|---:|
| MoE1 metadata-head + two_head + HMM | 0.4649 |
| Early concat + MoE temporal settings + HMM | 0.4845 |
| Two-horizon fixed 50/50 + HMM | 0.4744 |
| Two-horizon train-grid + HMM | 0.4596 |

## What Worked

1. HMM temporal decoding is consistently important. It remains the strongest
   post-processing improvement across model families.
2. Early multimodal fusion is competitive with, and sometimes stronger than,
   late modality-expert MoE when CR-social is not used as the dominant selection
   target.
3. Early projected concat is better than early gated fusion for PinSoRo under
   MoE temporal settings.
4. Linear post-logit partner interaction improves over the shared-person
   no-interaction baseline on CC, and HMM lifts it further.

## What Did Not Work Robustly

1. NOXI temporal settings transferred to PinSoRo but underperformed the better
   PinSoRo settings.
2. Short/long horizon experts did not beat the single long early-concat model in
   fair train-selected form.
3. CR-social remains unstable and low-support. It has only 17,295 validation
   frames and no class-4 support, so it should be reported but treated
   cautiously for model selection.
4. Class calibration, focal/oversampling, and prior tricks did not cleanly solve
   CR-social class behavior.

## Most Promising Next Experiments

1. Task/social specialists on early-concat features:
   - train a task-specialist model mostly or only on the task head;
   - train a social-specialist model mostly or only on the social head;
   - use task logits from the task specialist and social logits from the social
     specialist;
   - apply HMM after combining.

2. Compare no-interaction vs linear interaction with HMM for both CC and CR:
   - CC no-interaction raw is already worse than linear raw;
   - run CR no-interaction and HMM both no-interaction runs to isolate how much
     the partner module contributes beyond architecture and HMM.

3. Small post-logit TCN interaction:
   - same model as linear interaction, but replace the linear residual with a
     tiny causal TCN over self/partner logits;
   - keep it small: 1-2 blocks, hidden 16 or 32, residual scale 0.05-0.1;
   - apply HMM after.

4. Add soft-confidence labels to the best early-concat/person-interaction
   branches:
   - current person-interaction runs used soft_label_mode=none;
   - compare soft_confidence against none with identical HMM decoding.

5. HMM transition variants:
   - current person-interaction best used mix=1, strength=12, alpha=1;
   - test role-specific or domain/head-specific transition matrices only if
     there is enough support.

## Current Interpretation

For a clean hand-in story, the safest baseline remains MoE1 metadata-head +
two_head + HMM. The strongest architecture direction is early multimodal concat
with PinSoRo temporal settings, especially when CR-social is treated as a
low-support diagnostic head rather than the main selector. The most defensible
next expert story is task/social specialization, not modality experts or
short/long horizon experts.
