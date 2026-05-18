# First Results and Interpretation

## Session counts

- `train-cc`: 20 sessions
- `train-cr`: 12 sessions

## Agreement-rate summary from main files

Aggregated over sessions:

- `train-cc`:
  - `social_engagement`: ~0.918 agreement (`purple` and `yellow`)
  - `task_engagement`: ~0.929 agreement (`purple` and `yellow`)
- `train-cr`:
  - all four combinations are ~1.000 agreement (only 6 blank rows total per combination over all sessions)

## Disagreement pattern highlights (blank rows)

From `outputs/disagreement_blank_summary_by_session.csv`:

- `train-cc` blank rates are substantial:
  - ~8.15% (`purple social`)
  - ~7.12% (`purple task`)
  - ~8.16% (`yellow social`)
  - ~7.09% (`yellow task`)
- `train-cr` blank rates are effectively zero (6 blank rows over 406,902 per combination).

From `outputs/disagreement_blank_neighbor_labels.csv`:

- Many `train-cc` blank rows occur near `cooperative` contexts in social engagement.
- This suggests disagreement is concentrated around specific social-label regions/transitions, not uniformly random.

## `nan` pattern highlights

From `outputs/nan_summary_by_session.csv`:

- `train-cc` `nan` rates are around 2.5% to 2.8% depending on color/task.
- `train-cr` `nan` rates are consistently higher at about 5.59% across all color/task combinations.

Consequence:

- Even though blank disagreement is near-zero in CR, CR still has substantial `nan`-labeled rows.
- Filtering policy should always state both blank handling and `nan` handling explicitly.

## Annotator-disagreement highlights from numbered files

From `outputs/annotator_disagreement_session_summary.csv` and related outputs:

- Main blank disagreement with annotator vote traces is concentrated in `train-cc`.
- In `train-cr`, only a few blank rows exist and they mostly have no numbered-vote coverage.
- For the covered `train-cc` groups, about 37.5% of blank frames have annotator vote signatures available in numbered files.

From `outputs/annotator_disagreement_signature_counts.csv`:

- Frequent disagreement signatures include:
  - `associative | parallel` (social)
  - `associative | cooperative` (social)
  - `aimless | goaloriented` (task)
  - `goaloriented | nan` and `nan | noplay` (task)

Consequence:

- Uncertainty modeling can be developed primarily from `train-cc` where disagreement evidence is richer.
- Label boundaries and `nan`-adjacent regions are key uncertainty hotspots and should be treated explicitly in loss design.

## Label distribution highlights

From `outputs/label_aggregate_main_counts.csv`:

- `train-cc`:
  - `social_engagement` is dominated by `cooperative` and `parallel`.
  - `task_engagement` is dominated by `goaloriented`, with substantial `aimless` and `noplay`.
- `train-cr` (`purple`):
  - `social_engagement` is dominated by `solitary` and `parallel`.
  - `task_engagement` is dominated by `goaloriented`, then `noplay`.
- `train-cr` (`yellow`, likely robot side):
  - `task_engagement` is highly concentrated in `noplay`.
  - very low `goaloriented`.

## Purple CC vs CR comparison highlights

From `outputs/purple_cc_vs_cr_comparison.csv`:

- `task_engagement` (`all` scenario):
  - `goaloriented` proportion is higher in `train-cr` than `train-cc`.
  - `aimless` proportion is higher in `train-cc` than `train-cr`.
- `social_engagement` (`all` scenario):
  - `cooperative` is common in `train-cc` and absent in `train-cr` `purple`.
  - `solitary` is much higher in `train-cr` `purple` than `train-cc` `purple`.

This confirms a strong domain shift between child-child and child-robot interaction settings.

## Scenario-based weighting outputs

Generated in `outputs/label_scenario_class_weights.csv` for:

- `all`
- `drop_blank`
- `drop_nan`
- `drop_blank_and_nan`

These are ready for weighted loss or weighted sampling in training.

## Stream availability highlights

From `outputs/stream_inventory_by_session.csv` and `outputs/stream_coverage_by_split_feature.csv`:

- All inspected streams have matching binary files (`has_binary_stream = 1` throughout current inventory).
- Coverage is complete by expected entity multiplicity:
  - visual features (`clip`, `dino`, `openface2/3`, `openpose`, `swin`, `videomae`) appear for `env`, `purple`, `yellow`.
  - audio features (`audio.egemapsv2`, `audio.w2vbert2_embeddings`, `audio.xlm_roberta_embeddings`) appear for `purple`, `yellow` only.

Consequence:

- Stream existence is not the immediate bottleneck; label policy/disagreement handling is the primary modeling-risk axis right now.

## Temporal structure highlights (`purple`, `drop_blank_and_nan`)

From `outputs/label_run_summary_by_label.csv`:

- `train-cr` social labels are more persistent for dominant classes:
  - `solitary` and `parallel` runs are very long on average compared with CC.
- `train-cc` has broader social transition activity:
  - more frequent switching among `associative`, `cooperative`, and `parallel`.
- For task engagement in both domains:
  - `goaloriented` has the largest total time and long average run durations.
  - `adultseeking` remains relatively short and sparse.

From `outputs/label_transition_counts.csv`:

- In `train-cc` social engagement, top transitions include:
  - `associative -> cooperative`
  - `parallel -> cooperative`
  - `cooperative -> associative`
- This supports the interpretation that CC interactions contain richer short-horizon social state changes than CR.

From `outputs/label_temporal_heterogeneity.csv`:

- `train-cc` social sessions show higher transition activity than `train-cr` social.
- Task transition rates are high in both domains, with `train-cr` task slightly higher on average.

Consequence:

- Duration-aware modeling is likely helpful (for example temporal smoothing or sequence models with persistence bias).
- Domain-aware modeling remains important because CC/CR differ not only in class balance but also in temporal dynamics.

## Consequences for modeling

1. Strong class imbalance is expected and should be handled explicitly.
   - Use class weighting and/or balanced sampling.
2. `cc` and `cr` are behaviorally different domains.
   - Evaluate per-domain, not only pooled.
3. `yellow` in `train-cr` should be treated carefully.
   - It likely represents robot behavior and may not be directly comparable to child labels.
4. Agreement filtering is critical.
   - Primary training/evaluation should use consensus rows from main files.
5. Scenario-dependent weights should be tracked explicitly.
   - Weights change materially when removing blanks and/or `nan`.

## Caveat to resolve next

- The token `nan` appears as a frequent label value in multiple files.
- Need to verify whether this is:
  - a valid dataset class token, or
  - a missing/undefined label encoding that should be excluded/remapped.
