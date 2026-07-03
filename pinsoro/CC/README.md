# PinSoRo CC Reproducibility Track

This package covers the PinSoRo CC results used for article reproducibility:

- submitted CC models yielding the best test scores;
- final CC feature ablations reported in the paper;
- final CC partner ablations reported in the paper;
- CC sensitivity artifacts used in the paper analysis;
- final submitted CC prediction folders.

Development-only and non-final exploratory runs are intentionally out of scope.

## Layout

- `scripts/`: CC training, inference, post-processing, and verification wrappers.
- `scripts/upstream/`: training/evaluation scripts used by the packaged CC runs.
- `src/acm_pipeline/`: local pipeline modules required by the scripts.
- `configs/`: named configs for submitted and ablation runs.
- `docs/`: detailed reproducibility and ablation notes.
- `artifacts/training_pipeline/`: end-to-end retraining notes.
- `artifacts/inference_only/`: selected checkpoints, configs, compact metrics,
  submitted-model artifacts, feature-ablation outputs, partner-ablation
  artifacts, and sensitivity outputs.
- `submissions/`: final submitted CC prediction folder.

See `artifacts/inference_only/ARTIFACT_MANIFEST.csv` for the run-level mapping.
