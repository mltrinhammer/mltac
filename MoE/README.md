# PinSoRo MoE Plan

Date: 2026-06-09

This folder is the working area for the PinSoRo mixture-of-experts direction.
The old five-fold model-improvement folders are historical evidence only.

## Selected Expert Modalities

Use one modality family representative first:

| Family | Feature set | Reason |
| --- | --- | --- |
| Visual | `visual_videomae` | Best visual and best overall baseline. |
| Audio | `audio_w2vbert2` | Best audio baseline. |
| Text | `text_xlm_roberta` | Only text feature set and best text baseline. |

Earlier organizer-score references from `outputs/pinsoro/results_report.md`:

| Feature set | Best baseline model | Organizer score |
| --- | --- | ---: |
| `visual_videomae` | attention | 0.4149 |
| `audio_w2vbert2` | simple | 0.2479 |
| `text_xlm_roberta` | attention | 0.1563 |

## Hard Rule: No CC/CR Mixing

CC and CR must stay separate at every data-dependent step:

- separate CC and CR feature normalizers
- separate CC and CR train/evaluation manifests
- separate CC and CR modality experts
- separate CC and CR calibration or decoding parameters, if used

Do not fit a normalizer, class-weight vector, decoder, calibration model, or
gate using combined CC+CR rows.

## Preprocessing

Every selected modality is aligned to the 30 Hz label timeline by
`scripts/pinsoro_prepare_feature_tensors_30hz.py`.

That script:

- reads the raw feature stream rate from each `.stream` header
- interpolates features onto the 30 Hz label grid when needed
- keeps labels and masks on the same 30 Hz timeline
- writes one session-role `.npz` tensor per modality

This applies to audio, text, and VideoMAE. The text feature set is stored as
`audio.xlm_roberta_embeddings` in the raw stream manifest, but it is still
prepared by the same 30 Hz alignment code.

After 30 Hz alignment, fit z-score normalization separately for:

```text
domain in {CC, CR}
feature_set in {visual_videomae, audio_w2vbert2, text_xlm_roberta}
```

Apply the matching `(domain, feature_set)` normalizer to train/validation/test
rows for that domain and feature set.

## Evaluation Strategy

Use the existing labeled split for the first MoE development loop:

```text
train-cc -> train the three CC modality experts
val-cc   -> validate experts and choose/early-stop the tiny MoE combiner
```

After the CC recipe is fixed, repeat the same protocol for CR:

```text
train-cr -> train the three CR modality experts
val-cr   -> validate experts and choose/early-stop the tiny MoE combiner
```

Leave-one-session-out is too expensive as the default because it retrains deep
experts many times. Keep it as a later robustness audit, not the first training
protocol.

Validation rows must not influence normalization, class weights, calibration,
decoding, or combiner fitting. Use validation only for selection and stopping.

## Final Training

For final submission models, include all labeled organizer train and validation
sessions for the relevant domain.

For example:

- final CC experts: fit CC normalizers on all labeled CC train+validation rows,
  then train CC experts on all labeled CC rows
- final CR experts: fit CR normalizers on all labeled CR train+validation rows,
  then train CR experts on all labeled CR rows

This is the only place where the old validation split should be merged into
training. Even then, CC and CR remain separate.

## Initial Expert Count

The first MoE version should train three CC modality experts:

```text
CC visual_videomae
CC audio_w2vbert2
CC text_xlm_roberta
```

After the CC MoE recipe is chosen, train the same architecture and procedure on
CR with CR-only data. Use one architecture family across modalities where
practical, then let the input dimensionality vary by feature set.

## MoE Weight Learning

Start with trained, frozen modality experts. Export logits on `train-cc` and
`val-cc`, then learn a tiny combiner on `train-cc` logits:

```text
task weights:   video, audio, text
social weights: video, audio, text
```

Use `val-cc` only to choose or early-stop the combiner. This uses more labeled
data than validation-only grid search while keeping the learned MoE parameters
small enough to debug.

## Current Regeneration Status

Raw streams still exist under:

```text
cache/pinsoro
```

The processed audio/text tensors were deleted during cleanup and must be
regenerated from the raw cache before MoE experiments.

## MoE 1 Scripts

The prepared overnight runner is:

```bash
cd /work/ACM/mltac-main
/work/ACM/mltac-main/ACM/.venv-gpu/bin/python ACM/MoE/run_moe1_overnight.py --gpus 0,1,2
```

It extracts age/gender metadata, runs CC first, then CR. The metadata-free and metadata-router outputs are separated by domain under:

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

Use `--domains CC` or `--domains CR` to run one domain only. Use
`--skip-experts` to rerun only the lightweight combiner ablations after expert
training has finished.


Participant ID is intentionally not used as metadata. The only metadata features prepared for MoE are age and gender.

The metadata-head branch uses separate trainer/evaluator scripts under `ACM/MoE` and does not modify the base PinSoRo trainer.
