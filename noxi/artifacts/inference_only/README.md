# Inference-Only Track

This track contains the selected artifacts needed to reproduce the clean W750 NoXi submitted/forward models and paper ablations without retraining, assuming the processed NoXi tensors are regenerated from the organizer-provided feature streams.

Each run directory keeps `config.json`, `model_best.pt`, compact metrics/logs, `val_gate_weights.csv` when present, and submission-format outputs when present. It intentionally omits `val_predictions.csv`, prepared `.npz` tensors, and `model_last.pt`.

## Submitted / Forward Models

- `submitted_models/noxi_base_meta_causal_w750_s375_seed13`: clean W750/S375 causal metadata model used for NOXI-base forward submission.
- `submitted_models/noxij_clean_forward_w750_s563_seed13`: clean W750/S563 metadata model used for the NOXI-J / NOXI-additional forward article family.

Submitted prediction folders are preserved under `submitted_predictions/` for `noxi-base`, `noxi-j`, and `noxi-additional`.


## Inference Command Pattern

After placing regenerated processed tensors at the relative paths referenced by the manifests, use `--eval-only` to load an included checkpoint and regenerate validation/test exports without retraining. Example for the NOXI-base W750/S375 submitted model:

```bash
python scripts/train_tcn_multimodal.py \
  --eval-only \
  --manifest data/manifests/window_stride_2026_06_22/audio_w2vbert2__text_xlm_roberta__visual_videomae_noxi_noxij_w750_s375_multimodal.csv \
  --output-root artifacts/inference_only/submitted_models \
  --run-name noxi_base_meta_causal_w750_s375_seed13 \
  --backbone dyadic_shared \
  --fusion-mode gated \
  --fusion-channels 96 \
  --modality-dropout 0.10 \
  --metadata data/metadata/noxi_metadata/role_metadata.csv data/metadata/noxi_j_metadata/role_metadata.csv \
  --metadata-set domain_role \
  --metadata-injection output_calibration \
  --metadata-embedding-dim 16 \
  --metadata-dropout 0.10 \
  --hidden-channels 96 \
  --levels 5 \
  --kernel-size 5 \
  --dropout 0.20 \
  --causal-tcn \
  --batch-size 8 \
  --ccc-weight 1.0 \
  --mse-weight 0.0 \
  --seed 13 \
  --device cuda
```

The matching NOXI-J/NOXI-additional clean forward model uses the W750/S563 manifest and run name `noxij_clean_forward_w750_s563_seed13`.

## Feature Ablations

`feature_ablations/` contains the 14 final VideoMAE-family runs from June 29, 2026: NOXI-facing S375 and NOXI-J-facing S563 profiles crossed with audio, text, VideoMAE, pairwise combinations, and all three modalities.

## Partner / Self Ablations

`partner_self_ablations/` contains the clean W750 submitted-profile rows from the June 30 note: self-only, dyadic shared, gated partner, and attention separate-encoder variants for single-corpus and joint training regimes.

## Scope Note

Older W500/S125 historical leaderboard artifacts are excluded deliberately. This package targets the clean W750 submitted-style family and the paper ablations.
