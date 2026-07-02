# NoXi / NoXi-J Reproducibility Track

This package covers the clean W750 NoXi family used for article reproducibility:

- submitted/forward NoXi models and prediction folders;
- final VideoMAE-family feature ablations from June 29, 2026;
- submitted-profile partner/self ablations from June 30, 2026.

Older W500/S125 historical leaderboard artifacts are intentionally out of scope. The goal is to reproduce the submitted-style W750 family and the ablations reported in the paper.

## Layout

- `scripts/`: NoXi training/preprocessing utilities.
- `src/acm_pipeline/`: local pipeline modules required by the scripts.
- `data/manifests/`: W750 manifests used by the submitted-style models and ablations.
- `data/metadata/`: role metadata used for domain-role output calibration.
- `artifacts/training_pipeline/`: end-to-end retraining notes.
- `artifacts/inference_only/`: selected checkpoints, configs, compact metrics, submitted predictions, and paper-ablation artifacts.

See `artifacts/inference_only/ARTIFACT_MANIFEST.csv` for the run-level mapping.
