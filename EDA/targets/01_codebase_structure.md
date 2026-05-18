# PinSoRo EDA Codebase Structure

## Current project layout

- `scripts/label_eda.py`
- `scripts/stream_inventory.py`
- `scripts/temporal_label_eda.py`
- `scripts/annotator_disagreement_eda.py`
- `outputs/label_summary_by_session.csv`
- `outputs/label_class_counts_by_session.csv`
- `outputs/label_numeric_annotation_files.csv`
- `outputs/label_aggregate_main_counts.csv`
- `outputs/label_scenario_counts.csv`
- `outputs/label_scenario_aggregate_counts.csv`
- `outputs/label_scenario_class_weights.csv`
- `outputs/purple_cc_vs_cr_comparison.csv`
- `outputs/disagreement_blank_summary_by_session.csv`
- `outputs/disagreement_blank_neighbor_labels.csv`
- `outputs/nan_summary_by_session.csv`
- `outputs/stream_inventory_by_session.csv`
- `outputs/stream_coverage_by_split_feature.csv`
- `outputs/label_runs_by_session.csv`
- `outputs/label_run_summary_by_label.csv`
- `outputs/label_run_summary_by_session.csv`
- `outputs/label_transition_counts.csv`
- `outputs/label_transition_probs.csv`
- `outputs/label_transition_with_gap_context.csv`
- `outputs/label_persistence_curve.csv`
- `outputs/label_temporal_heterogeneity.csv`
- `outputs/annotator_disagreement_frame_votes.csv`
- `outputs/annotator_disagreement_session_summary.csv`
- `outputs/annotator_disagreement_signature_counts.csv`
- `outputs/annotator_disagreement_label_counts.csv`
- `outputs/annotator_disagreement_entropy_summary.csv`

## Purpose of each script

### `scripts/label_eda.py`

First-pass label EDA over mounted dataset (`X:`) for `train-cc` and `train-cr`.

What it does:

1. Iterates session folders in `X:\train-cc\*` and `X:\train-cr\*`.
2. Reads main annotation files:
   - `purple.task_engagement.annotation.csv`
   - `purple.social_engagement.annotation.csv`
   - `yellow.task_engagement.annotation.csv`
   - `yellow.social_engagement.annotation.csv`
3. Counts total rows, nonblank rows, blank rows, and agreement rate.
4. Counts per-label frequencies per session.
5. Detects numbered annotator files like `*.1.annotation.csv` and summarizes their presence.
6. Writes compact CSV outputs for downstream analysis and plotting.
7. Builds scenario-based outputs for four filtering settings:
   - `all`
   - `drop_blank`
   - `drop_nan`
   - `drop_blank_and_nan`
8. Computes split/color/task class weights and a dedicated `purple` CC-vs-CR comparison table.
9. Computes disagreement-focused outputs from blank rows:
   - per-session blank rate
   - neighboring-label context around blank positions.

### `scripts/stream_inventory.py`

Per-session stream availability inventory over all splits.

What it does:

1. Iterates sessions in `train-*`, `val-*`, `test-*`.
2. Lists every `*.stream` header and checks if matching `*.stream~` exists.
3. Parses `sr`, `dim`, and `num` from stream headers.
4. Writes both detailed inventory and split-level feature coverage.

### `scripts/temporal_label_eda.py`

Temporal structure EDA on `purple` labels (train splits), using `drop_blank_and_nan`.

What it does:

1. Builds consecutive label runs per session/task.
2. Computes run durations in frames and seconds.
3. Computes run-level transition counts/probabilities.
4. Captures transition context across removed gaps (blank/`nan`).
5. Computes persistence curves `P(run >= t)`.
6. Computes session-level temporal heterogeneity metrics.

### `scripts/annotator_disagreement_eda.py`

Annotator-focused disagreement EDA on frames where main consensus files are blank.

What it does:

1. Finds blank indices in main annotation files.
2. Reads numbered annotator files (for example `.1.annotation.csv`, `.2.annotation.csv`).
3. Extracts per-frame vote signatures at those blank indices.
4. Aggregates disagreement signatures, label counts, and entropy.
5. Summarizes disagreement availability and coverage per session.

## Purpose of each output file

### `outputs/label_summary_by_session.csv`

Session-level row counts and agreement rates for each split/color/task combination.

### `outputs/label_class_counts_by_session.csv`

Session-level class distributions.

### `outputs/label_numeric_annotation_files.csv`

Inventory of numbered annotator files (`*.1.annotation.csv`, etc.) and basic row counts.

### `outputs/label_aggregate_main_counts.csv`

Aggregate class counts across all sessions for each split/color/task in the main (non-numbered) files.

### `outputs/label_scenario_counts.csv`

Session-level class counts under each filter scenario.

### `outputs/label_scenario_aggregate_counts.csv`

Aggregate class counts and proportions under each filter scenario.

### `outputs/label_scenario_class_weights.csv`

Class weights by scenario, split, color, and task.
Weight formula used: `N / (K * n_c)`, where `N` is total rows, `K` number of classes, and `n_c` class count.

### `outputs/purple_cc_vs_cr_comparison.csv`

Direct comparison of `purple` label distributions between `train-cc` and `train-cr`, including proportion differences.

### `outputs/disagreement_blank_summary_by_session.csv`

Per-session blank-row counts/rates for each split/color/task in main annotation files.

### `outputs/disagreement_blank_neighbor_labels.csv`

Counts of label context around blank positions (`prev_label` and `next_label`).

### `outputs/nan_summary_by_session.csv`

Per-session `nan` counts/rates for each split/color/task in main annotation files.

### `outputs/stream_inventory_by_session.csv`

Long-format stream presence table per session/entity/feature with parsed header metadata.

### `outputs/stream_coverage_by_split_feature.csv`

Split-level counts showing how many session-entity streams exist for each feature.

### `outputs/label_runs_by_session.csv`

One row per run segment with `start/end`, duration, and neighboring run labels.

### `outputs/label_run_summary_by_label.csv`

Per split/task/label duration statistics (`mean`, `median`, quantiles, max) and run counts.

### `outputs/label_run_summary_by_session.csv`

Per split/session/task run-level summary statistics.

### `outputs/label_transition_counts.csv`

Run-level transition counts (`from_label -> to_label`).

### `outputs/label_transition_probs.csv`

Row-normalized transition probabilities per `from_label`.

### `outputs/label_transition_with_gap_context.csv`

Context of transitions separated by removed gaps (`blank`/`nan`) with gap lengths.

### `outputs/label_persistence_curve.csv`

Persistence/survival values per label across time thresholds.

### `outputs/label_temporal_heterogeneity.csv`

Session-level temporal complexity metrics: transition rate, run duration, entropy.

### `outputs/annotator_disagreement_frame_votes.csv`

Frame-level vote signatures from numbered annotator files at blank-consensus indices.

### `outputs/annotator_disagreement_session_summary.csv`

Session-level coverage of blank frames with annotator votes.

### `outputs/annotator_disagreement_signature_counts.csv`

Counts of unique disagreement signatures (`label_a | label_b | ...`) by split/color/task.

### `outputs/annotator_disagreement_label_counts.csv`

Label frequencies observed inside disagreement votes.

### `outputs/annotator_disagreement_entropy_summary.csv`

Mean vote entropy (bits) by split/color/task on blank-consensus frames.
