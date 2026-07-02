# CC Reproducibility Artifacts

This directory now follows the same two-track layout used by the CR bundle.

- `training_pipeline/`: notes for end-to-end retraining from organizer-provided PinSoRo embedding streams. Processed tensors are not included.
- `inference_only/`: checkpoints, configs, compact metrics, submission outputs, and ablation outputs needed for inference-only reproduction. Large score dumps and `model_last.pt` files are intentionally excluded from the push-ready bundle.

The inference-only track is split into submitted-model artifacts, submitted-checkpoint feature ablations, partner ablations, and a sensitivity placeholder. The CC sensitivity artifacts still need to be added by the CC pipeline owner.
