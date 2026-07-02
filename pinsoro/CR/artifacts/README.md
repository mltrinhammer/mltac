# CR Reproducibility Artifacts

This directory follows a two-track layout.

- `training_pipeline/`: commands and manifests for end-to-end retraining from organizer-provided PinSoRo embedding streams. Tensors are not included.
- `inference_only/`: checkpoints, configs, HMM settings, metrics, and exported submission-format predictions for reproducing selected CR results without retraining. Large prepared tensors and per-frame score dumps are not included.

The inference-only track is split into submitted-model artifacts, paper feature ablations, paper partner/encoder ablations, and the soft-kappa sensitivity analysis.
