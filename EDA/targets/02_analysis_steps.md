# Label EDA Steps (Current)

## Scope of this first pass

Goal: understand annotation structure and class balance before feature modeling.

Dataset scope:

- `train-cc` sessions
- `train-cr` sessions
- Colors: `purple`, `yellow`
- Targets: `task_engagement`, `social_engagement`

## Step-by-step process

1. Counted sessions per split.
   - `train-cc = 20`
   - `train-cr = 12`

2. Read main annotation files (`*.annotation.csv`, non-numbered).
   - Counted `total_rows`, `nonblank_rows`, `blank_rows`.
   - Computed `agreement_rate = nonblank_rows / total_rows`.

3. Parsed label values from nonblank rows.
   - Computed per-session class counts.
   - Aggregated counts over all sessions per split/color/task.

4. Indexed numbered annotator files (`*.1.annotation.csv`, etc.).
   - Recorded file presence and nonblank row counts.
   - Kept this separate from the main consensus-file analysis.

5. Wrote outputs to `outputs/` for reproducible downstream plotting and checks.

6. Added four filtering scenarios for label accounting:
   - `all`
   - `drop_blank`
   - `drop_nan`
   - `drop_blank_and_nan`

7. Computed class weights per scenario/split/color/task.
   - Formula: `weight_c = N / (K * n_c)`.

8. Added explicit `purple` domain-shift comparison (`train-cc` vs `train-cr`) with both absolute counts and proportions.

9. Added disagreement-focused analysis from blank rows in main labels.
   - Per-session blank rates.
   - Neighbor-label context around blank indices (`prev_label`, `next_label`).

10. Added stream inventory analysis across sessions.
   - Stream presence (`*.stream` and matching `*.stream~`).
   - Header metadata extraction (`sr`, `dim`, `num`).

11. Added temporal-structure EDA (`purple`, `train-cc/train-cr`, `drop_blank_and_nan`).
   - Run segmentation and duration statistics.
   - Transition counts/probabilities.
   - Gap-context transitions across removed rows.
   - Persistence curves and session heterogeneity metrics.

12. Added annotator-disagreement EDA on blank consensus frames.
   - Joined main blank indices with numbered annotator files.
   - Extracted frame-level vote signatures and disagreement frequencies.
   - Computed session-level blank-vote coverage and vote entropy.

## Notes on interpretation

- Main non-numbered annotation files are treated as the primary training/evaluation target.
- Blank rows in main files are treated as no-consensus/disagreement rows.
- Numbered files are treated as additional annotator-specific views, not primary labels in this first pass.

## Next analysis steps (not run yet)

1. Add split-aware temporal baselines (for example, Markov or run-duration conditioned baseline).
2. Decide whether to model `cc` and `cr` separately or with domain adaptation, based on transition and duration divergence.
3. Add feature-quality checks per stream (missing values, constant dimensions) now that stream presence is mapped.
4. Integrate temporal windowing strategy with class weighting into first supervised model pipeline.
