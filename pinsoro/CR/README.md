# PinSoRo CR Reproducibility Track

This package covers the PinSoRo CR results used for article reproducibility:

- submitted CR models yielding the best test scores;
- final CR feature ablations reported in the paper;
- final CR partner / encoder ablations reported in the paper;
- CR sensitivity artifacts used in the paper analysis.

Development-only and non-final exploratory runs are intentionally out of scope.
The goal is to reproduce the submitted CR predictions and the ablations reported
in the paper.

## Layout

- `MoE/pinsoro_noxi_settings/`: CR training and post-processing scripts.
- `scripts/`: PinSoRo preprocessing and manifest utilities.
- `src/acm_pipeline/`: local pipeline modules required by the scripts.
- `data/manifests/`: CR window manifests used by the submitted models and ablations.
- `data/metadata/`: participant metadata used by the packaged CR runs.
- `artifacts/training_pipeline/`: end-to-end retraining notes.
- `artifacts/inference_only/`: selected checkpoints, run-local configs, compact
  metrics, submitted prediction outputs, paper-ablation artifacts, and
  sensitivity outputs.

CR does not use a separate top-level `configs/` folder in this package. The
configs needed for inference-only reproduction are stored with their
corresponding run artifacts, for example under
`artifacts/inference_only/submitted_models/` and
`artifacts/inference_only/feature_ablations/`.

See `artifacts/inference_only/ARTIFACT_MANIFEST.csv` for the run-level mapping.
