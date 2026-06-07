# PinSoRo Handoff

## Goal

Run an ACM-equivalent modality and architecture analysis on PinSoRo while
keeping the existing NoXi/NoXi-J pipeline unchanged.

Selected architectures:

- simple TCN as the individual-role baseline
- dyadic shared TCN
- attention TCN

PinSoRo is a two-head multiclass classification task:

- task engagement: 4 classes
- social engagement: 5 classes

## Agreed Modeling Contract

- Use one combined CC/CR model with explicit domain metadata.
- CC: purple and yellow are both supervised participants.
- CR: purple is supervised; yellow is retained as robot/context input.
- Use one shared encoder with separate task and social classification heads.
- Ignore blank, `nan`, and unknown labels through per-head masks.
- Primary metric: Cohen's kappa.
- Additional metrics: macro-F1 and accuracy.
- Fit feature normalization using training splits only.
- Use fixed 10-second windows: 250 frames at 25 Hz.
- Use stride 62 frames, approximately 2.5 seconds.
- Individual and dyadic models must use the same canonical window boundaries.
- Report combined results and separate CC/CR results.

The organizer baseline was inspected from:

```text
/work/ACM/MultiMediate26-main.zip
```

The organizer baseline predicts individual frames with a dense model. The ACM
PinSoRo pipeline intentionally extends this with temporal windows and TCNs.

## Stage 1 Status

Stage 1 is complete.

Results:

- 56 sessions shared across all 9 modalities
- 1,008 normalized session-role tensors
- 0 tensor integrity errors
- 27,026 shared canonical windows

Storage:

- extracted cache: approximately 123 GB
- aligned and normalized tensors: approximately 133 GB
- manifests and reports: approximately 137 MB

## Stage 1 Implementation

Canonical runner:

```text
ACM/scripts/run_pinsoro_stage1.sh
```

PinSoRo-specific scripts:

```text
ACM/scripts/pinsoro_build_raw_manifests.py
ACM/scripts/pinsoro_prepare_feature_tensors_25hz.py
ACM/scripts/pinsoro_fit_apply_feature_transform.py
ACM/scripts/pinsoro_build_shared_window_manifests.py
ACM/scripts/pinsoro_build_window_manifests.py
ACM/scripts/pinsoro_report_stream_coverage.py
ACM/scripts/pinsoro_validate_preprocessing.py
```

Shared PinSoRo conventions:

```text
ACM/src/acm_pipeline/pinsoro.py
```

Stage 1 documentation:

```text
ACM/docs/04_pinsoro_stage1.md
```

No existing NoXi scripts were modified.

## Stage 1 Outputs

Original archives:

```text
/work/ACM/PinSoRo/
```

Extracted data:

```text
/work/ACM/mltac-main/ACM/cache/pinsoro/
```

Aligned and normalized tensors:

```text
/work/ACM/mltac-main/ACM/processed/pinsoro/
```

Raw and processed manifests:

```text
/work/ACM/mltac-main/ACM/outputs/pinsoro/manifests/
```

Training-ready window manifests:

```text
/work/ACM/mltac-main/ACM/outputs/pinsoro/windows/
```

Important files:

```text
outputs/pinsoro/windows/shared_w250_s62_canonical.csv
outputs/pinsoro/windows/<feature_set>_w250_s62_individual.csv
outputs/pinsoro/windows/<feature_set>_w250_s62_dyadic.csv
```

Validation reports:

```text
/work/ACM/mltac-main/ACM/outputs/pinsoro/validation/
```

Reports include:

- `tensor_integrity.csv`
- `label_class_counts.csv`
- `window_counts.csv`
- `stream_coverage.csv`

## Stage 2 Starting Point

Stage 2 should adapt the three selected TCN architectures for the PinSoRo
classification contract. Do not modify or break existing NoXi training.

Recommended implementation order:

1. Add PinSoRo individual and dyadic window dataset loaders.
2. Add domain encoding for CC/CR.
3. Adapt simple TCN for one participant and two classification heads.
4. Adapt dyadic shared and attention TCNs for synchronized purple/yellow input.
5. Implement masked multitask cross-entropy.
6. Implement kappa, macro-F1, and accuracy overall and by domain/head/role.
7. Add CPU smoke tests using a small subset of existing Stage 1 outputs.
8. Add a PinSoRo-specific training runner for the 9 modality ablations and 3 architectures.

For test inference:

- CC should produce purple and yellow predictions.
- CR should produce purple predictions only.
- Domain is known from `source_split` and explicit `domain` fields.

GPU guidance:

- Stage 2 development and smoke tests can run on CPU.
- Full training should use one GPU per experiment.
- Independent modality/architecture experiments can be parallelized across GPUs.
