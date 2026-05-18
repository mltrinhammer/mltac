# PinSoRo EDA (Targets)

This folder contains exploratory data analysis for the MultiMediate PInSoRo challenge target labels, focused on `train-cc` and `train-cr` with precomputed multimodal features.

## What was analyzed

- Label balance and agreement (`blank` rows) by split, color, and task.
- `nan` prevalence by split, color, and task.
- Temporal structure of labels:
  - run durations
  - label transitions
  - persistence curves
  - session heterogeneity
- Annotator disagreement from numbered files (`*.1.annotation.csv`, `*.2.annotation.csv`) at blank-consensus frames.
- Stream availability and metadata (`sr`, `dim`, `num`) per session.

## Key insights

- Strong domain differences between child-child (`cc`) and child-robot (`cr`) in both class mix and temporal dynamics.
- `cc` has substantial blank/disagreement rates; `cr` has near-zero blanks but higher `nan` rates.
- Disagreement is structured (specific label-pair confusion), not random.
- Temporal persistence and transition patterns differ across domains, suggesting domain-aware and uncertainty-aware modeling is relevant.

## Folder guide

- `scripts/`: EDA scripts used to generate outputs and figures.
- `outputs/`: CSV tables from the analyses.
- `figures/`: generated visual summaries.
- `01_*.md` to `04_*.md`: narrative documentation of methods and findings.
