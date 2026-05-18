# Output Figures (Planned from Current Outputs)

## Figure 1: Session count by split

- Source: folder counts from `train-cc` and `train-cr`.
- Purpose: establish dataset balance at session level.
- Expected pattern: more sessions in `train-cc` than `train-cr`.
- Key takeaway: the training set is session-imbalanced (`train-cc` > `train-cr`), so pooled metrics can over-reflect CC behavior.

## Figure 2: Agreement rate by split/color/task

- Source: `outputs/label_summary_by_session.csv`.
- Plot: grouped bar chart of agreement rate.
- Purpose: show where disagreement/blank rows are concentrated.
- Expected pattern: `train-cc` lower agreement than `train-cr`.
- Key takeaway: disagreement is mostly a CC issue; CR appears near-consensus in the main files.

## Figure 3: Class distribution for task engagement

- Source: `outputs/label_aggregate_main_counts.csv`.
- Plot: stacked bars or side-by-side bars, split by `train-cc` vs `train-cr`, and color (`purple`, `yellow`).
- Purpose: identify class imbalance and domain differences.
- Key takeaway: `goaloriented` dominates most groups, while some classes (for example `adultseeking`) are sparse and likely harder to learn.

## Figure 4: Class distribution for social engagement

- Source: `outputs/label_aggregate_main_counts.csv`.
- Plot: same style as Figure 3.
- Purpose: compare social-behavior composition across domains.
- Key takeaway: social labels shift strongly by domain, especially `cooperative` (higher in CC) versus `solitary` (higher in CR).

## Figure 5: Session-level variability heatmap

- Source: `outputs/label_class_counts_by_session.csv`.
- Plot: session x label heatmap (normalized).
- Purpose: detect whether class imbalance is global or driven by specific sessions.
- Key takeaway: label mix is not uniform across sessions; a few sessions can disproportionately drive class prevalence.

## Figure 6: Numbered annotator file coverage

- Source: `outputs/label_numeric_annotation_files.csv`.
- Plot: count of sessions with numbered files by split/color/task.
- Purpose: quantify where optional annotator-specific labels are available.
- Key takeaway: annotator-specific files are available but uneven; use them as optional uncertainty signals, not mandatory inputs.

## Figure 7: Scenario-wise class weights

- Source: `outputs/label_scenario_class_weights.csv`.
- Plot: bar chart of class weights per scenario for each split/task/color.
- Purpose: make class-imbalance handling explicit before training.
- Key takeaway: weights change materially when filtering blanks/`nan`; training config must lock the exact filtering scenario.

## Figure 8: Purple CC vs CR label shift

- Source: `outputs/purple_cc_vs_cr_comparison.csv`.
- Plot: paired bars for `cc_proportion` and `cr_proportion` by class, faceted by task and scenario.
- Purpose: isolate domain shift on the comparable actor (`purple`).
- Key takeaway: `purple` distributions differ clearly between CC and CR, so domain-aware evaluation (and possibly domain-aware training) is warranted.

## Figure 9: Blank-rate by session and target

- Source: `outputs/disagreement_blank_summary_by_session.csv`.
- Plot: grouped bars or heatmap over `session x (color, task)` using `blank_rate`.
- Purpose: localize disagreement intensity by session and behavior target.
- Key takeaway: disagreement is concentrated in CC sessions; CR is nearly blank-free.

## Figure 10: Stream coverage matrix

- Source: `outputs/stream_inventory_by_session.csv` and `outputs/stream_coverage_by_split_feature.csv`.
- Plot: `session x feature` presence heatmap (optionally faceted by split/entity).
- Purpose: verify feature-stream availability before multimodal model assembly.
- Key takeaway: current inventory indicates complete stream presence with matching binary payloads.

## Figure 11: NaN-rate by session and target

- Source: `outputs/nan_summary_by_session.csv`.
- Plot: heatmap over `session x (split,color,task)` using `nan_rate`.
- Purpose: localize where undefined labels (`nan`) concentrate.
- Key takeaway: CR sessions show a consistently higher `nan` rate than CC.

## Figure 12: NaN-rate by split/color/task

- Source: `outputs/nan_summary_by_session.csv`.
- Plot: grouped bar chart of aggregated `nan_rate`.
- Purpose: summarize undefined-label prevalence by regime and target.
- Key takeaway: blank disagreement and `nan` prevalence capture different annotation-quality dimensions.

## Figure 13: Blank-vote coverage heatmap

- Source: `outputs/annotator_disagreement_session_summary.csv`.
- Plot: session x group heatmap of `blank_vote_rate`.
- Purpose: show where blank consensus frames have usable numbered-annotator votes.
- Key takeaway: useful annotator-vote uncertainty evidence is concentrated in selected CC sessions.

## Figure 14: Top disagreement signatures

- Source: `outputs/annotator_disagreement_signature_counts.csv`.
- Plot: top vote signatures by count.
- Purpose: identify which class pairs/sets are most frequently disputed.
- Key takeaway: disagreement is structured (specific confusing label pairs), not random noise.


## Temporal Figure 01: Run-duration distributions

- Source: `outputs/label_runs_by_session.csv`.
- Plot: violin distributions of run length (seconds) by split/task/label.
- Purpose: compare label persistence across domains and tasks.
- Key takeaway: dominant social states in CR tend to have longer runs than CC.

## Temporal Figure 02: Run count vs total label time

- Source: `outputs/label_run_summary_by_label.csv`.
- Plot: scatter of `n_runs` versus `total_seconds`.
- Purpose: separate frequent-short labels from rare-long labels.
- Key takeaway: labels with similar total time can have very different temporal structure.

## Temporal Figure 03: Task-transition heatmap

- Source: `outputs/label_transition_probs.csv`.
- Plot: `from -> to` probability heatmap for `task_engagement` (CC and CR panels).
- Purpose: reveal dominant task-state pathways.
- Key takeaway: transition behavior differs by domain beyond static class balance.

## Temporal Figure 04: Social-transition heatmap

- Source: `outputs/label_transition_probs.csv`.
- Plot: `from -> to` probability heatmap for `social_engagement` (CC and CR panels).
- Purpose: compare social dynamics in child-child and child-robot settings.
- Key takeaway: CC shows richer social state switching; CR is more concentrated.

## Temporal Figure 05: Gap-context transition pairs

- Source: `outputs/label_transition_with_gap_context.csv`.
- Plot: most common label pairs across removed gaps (`blank`/`nan`).
- Purpose: localize where uncertainty/removals disrupt transition paths.
- Key takeaway: disagreement/missing segments are concentrated around specific boundaries.

## Temporal Figure 06: Persistence curves

- Source: `outputs/label_persistence_curve.csv`.
- Plot: `P(run >= t)` by label over time thresholds.
- Purpose: compare temporal stability per class.
- Key takeaway: clearly distinguishes transient labels from stable-state labels.

X-axis: time threshold t (seconds).
  - Y-axis: P(run >= t) = fraction of runs for a label that last
    at least t seconds.
  - Each line: one label in one split/task.
    How to read it:
  - Higher line = more persistent/stable label.
  - Faster drop = more short/transient runs.
    What to use it for:
  - Decide smoothing/window sizes (short-lived labels need
    shorter windows).
  - Identify labels that may need temporal priors or post-
    processing.

## Temporal Figure 07: Session temporal heterogeneity

- Source: `outputs/label_temporal_heterogeneity.csv`.
- Plot: transition rate per minute versus mean run duration.
- Purpose: identify sessions with volatile versus stable dynamics.
- Key takeaway: sessions cluster by temporal regime and can inform split strategy or curriculum.

For Temporal 07 (session heterogeneity):

  - Each point: one session (for one task).
  - X-axis: transition rate per minute (how often label
    changes).
  - Y-axis: mean run duration in seconds (average stability of
    states).
    How to read it:
  - Right + low = volatile session (many short states).
  - Left + high = stable session (few long states).
    What to use it for:
  - Detect outlier sessions.
  - Build balanced train/val splits that include both volatile
    and stable sessions.
  - Consider curriculum/domain strategy (train on stable first,
    then volatile, or stratify batches).

## Implementation status

- CSV outputs are generated and ready.
- Figure image files are generated, including temporal figures.
