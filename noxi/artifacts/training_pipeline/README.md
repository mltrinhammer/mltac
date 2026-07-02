# Training Pipeline Track

The NoXi W750 models use `scripts/train_tcn_multimodal.py` with manifests under `data/manifests/` and role metadata under `data/metadata/`.

The source note `artifacts/inference_only/PAPER_ABLATIONS_NOXI.txt` records the exact command families and hyperparameters for the June 29 feature ablations and June 30 partner/self ablations.

Processed feature tensors are not included. Regenerate or place them at the paths expected by the manifests before retraining or checkpoint inference.
