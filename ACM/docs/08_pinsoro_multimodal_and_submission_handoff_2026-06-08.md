# PinSoRo Multimodal and Submission Handoff - 2026-06-08

## Decisions

Multimodal capability is implemented now, but modality selection must happen
only after the full unimodal grid finishes. The selector refuses to proceed
while any unimodal feature/model result is incomplete.

The bounded default multimodal experiment design is:

- select the architecture with the highest mean organizer score across all
  completed unimodal feature sets
- rank modalities within family using that selected architecture
- retain the top 2 audio modalities
- retain the only available text modality
- retain the top 3 visual modalities
- train one audio + one text + one visual modality per combination

This produces 2 x 1 x 3 = 6 multimodal runs, rather than all possible feature
subsets or all architecture/combination pairs. The counts and architecture can
be explicitly overridden if a later experimental decision requires it.

## Fusion Implementation

Selected feature manifests are passed together to `train_pinsoro_tcn.py`.
The loader verifies that their canonical windows match, memory-maps each
selected modality, and concatenates aligned feature windows per role. This is
early fusion and reuses the existing simple, dyadic-shared, and attention TCNs.

No additional combined tensor cache is created. The existing 95 GiB mmap cache
is reused.

Relevant files:

```text
src/acm_pipeline/pinsoro_data.py
scripts/train_pinsoro_tcn.py
scripts/run_pinsoro_selected_multimodal_4gpu.py
```

## Required Workflow on the GPU Host

First finish the 27-run unimodal seed-13 grid. Then collect only completed
results:

```bash
python scripts/collect_pinsoro_results.py
```

Inspect the bounded multimodal plan:

```bash
python scripts/run_pinsoro_selected_multimodal_4gpu.py \
  --python /path/to/cuda-enabled/python \
  --gpus 0,1,2,3 \
  --dry-run
```

The dry run writes:

```text
outputs/pinsoro/selected_multimodal_plan.csv
```

Review that plan before removing `--dry-run`. By default the selector requires
all 27 unimodal feature/model results and will fail if any are missing.

Multimodal outputs are written under:

```text
outputs/pinsoro/multimodal_experiments/
```

Because fused inputs can be much wider than unimodal inputs, monitor GPU memory
on the first runs. Do not reduce batch size unless required by GPU memory.

## Prediction Output Contract

Every completed new run writes both:

```text
test_predictions.csv                 long-form diagnostic output
test_submission_format/              organizer-style submission tree
```

The submission tree uses the requested layout:

```text
test_submission_format/
  pinsoro-cc/
    007/
      purple.social_engagement.prediction.csv
      purple.task_engagement.prediction.csv
      yellow.social_engagement.prediction.csv
      yellow.task_engagement.prediction.csv
  pinsoro-cr/
    <session>/
      purple.social_engagement.prediction.csv
      purple.task_engagement.prediction.csv
```

CC exports purple and yellow. CR exports purple only. Each CSV contains exactly
one string class label per line and no header, matching the organizer's
classification serialization. The requested filename form is used without an
extra duplicated `engagement` component.

The four already-completed unimodal runs were converted without retraining.
The standalone converter is:

```bash
python scripts/export_pinsoro_submission.py \
  --predictions <run>/test_predictions.csv \
  --output-dir <run>/test_submission_format
```

## Verification Completed

- Real three-family individual fused batch: passed
- Real three-family dyadic fused batch: passed
- End-to-end three-family CPU training smoke test: passed
- New trainer submission-tree export: passed
- Existing completed-run conversion: 4 runs, 36 files per run
- Synthetic complete-grid selector test: exactly 6 default multimodal runs
- Incomplete real-grid selector test: correctly refused selection
