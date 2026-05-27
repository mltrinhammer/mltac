# Preprocessing Progress

This document tracks what has been built so far, why it was done, and which scripts/outputs correspond to each step. It is intended to be updated as UCloud runs are added.

## Current Goal

Build a clean preprocessing pipeline that can be pushed to GitHub and reused on UCloud. The pipeline should prepare consistent model inputs for audio and visual feature-set experiments.

## Feature-Set Decision

We decided to work with full feature sets rather than subselecting individual OpenFace/OpenPose dimensions, because the stream files available here do not provide reliable per-dimension feature names.

Current feature sets:

```text
audio_egemaps
audio_w2vbert2
visual_swin
visual_openface
visual_openpose
```

The feature-set definitions live in:

```text
src/acm_pipeline/feature_registry.py
```

## Alignment Decision

The common aligned representation is 25 Hz.

Reason:

```text
the engagement target is effectively on a 25 Hz grid
most available streams are listed as 25 Hz
the existing eGeMAPS baseline used 25 Hz successfully
```

The alignment logic also handles cases where metadata is suspicious. If the feature frame count already closely matches the target length, the pipeline trusts the frame count rather than blindly resampling by the declared sample rate.

Implemented in:

```text
src/acm_pipeline/alignment.py
scripts/noxi_prepare_feature_tensors_25hz.py
```

Primary outputs:

```text
processed/<feature_set>_25hz/
outputs/manifests/model_processed_manifest_<feature_set>_25hz.csv
outputs/manifests/feature_status_<feature_set>_25hz.csv
```

## Normalization Decision

All model-input branches use train-fitted z-score normalization.

This applies to:

```text
raw
pca
random_projection
```

Meaning of `raw`:

```text
aligned + normalized, no dimensionality reduction
```

It does not mean unnormalized.

Reason:

```text
PCA is scale-sensitive
random projection behaves better with comparable feature scales
TCN/Transformer optimization is more stable with normalized inputs
different streams have different numeric ranges and meanings
```

Implemented in:

```text
src/acm_pipeline/transforms.py
scripts/noxi_fit_apply_feature_transform.py
```

## Dimensionality-Reduction Branches

The first supported branches are:

```text
raw
pca
random_projection
```

PCA was included first because it is common, interpretable, and produces an explained-variance table for choosing component counts.

Random projection was included as a simple additional baseline. It is useful as a comparison against PCA because it compresses dimensions without learning variance structure from the data.

Deferred for later:

```text
learned projection inside the model
autoencoder
other supervised or nonlinear reducers
```

## Smoke Tests Completed

The clean pipeline was smoke-tested using the existing eGeMAPS cache from the exploratory directory.

Alignment command:

```powershell
python scripts\noxi_prepare_feature_tensors_25hz.py --feature-set audio_egemaps --cache-root C:\Users\anec\projects\mltac\Noxi_Noxij\cache --manifest C:\Users\anec\projects\mltac\Noxi_Noxij\outputs\model_raw_manifest_train_with_split.csv --streams C:\Users\anec\projects\mltac\Noxi_Noxij\outputs\model_raw_manifest_streams_train.csv
```

Observed output:

```text
Wrote processed rows: 138
outputs/manifests/model_processed_manifest_audio_egemaps_25hz.csv
outputs/manifests/feature_status_audio_egemaps_25hz.csv
```

Raw transform command:

```powershell
python scripts\noxi_fit_apply_feature_transform.py --input-manifest outputs\manifests\model_processed_manifest_audio_egemaps_25hz.csv --method raw
```

PCA smoke command:

```powershell
python scripts\noxi_fit_apply_feature_transform.py --input-manifest outputs\manifests\model_processed_manifest_audio_egemaps_25hz.csv --method pca --n-components 8 --max-fit-frames 2000
```

Random projection smoke command:

```powershell
python scripts\noxi_fit_apply_feature_transform.py --input-manifest outputs\manifests\model_processed_manifest_audio_egemaps_25hz.csv --method random_projection --n-components 8 --max-fit-frames 2000
```

The PCA smoke run produced:

```text
outputs/transforms/audio_egemaps_pca8/pca_explained_variance.csv
```

The first 8 PCA components explained approximately 74.2% cumulative variance in the small smoke sample. This number is only a smoke-test diagnostic, not a final modelling decision.

## UCloud Plan

Once persistent UCloud storage is available:

1. Clone the GitHub repo.
2. Install requirements.
3. Populate `cache/` with the required `.stream` and `.stream~` files.
4. Run `noxi_prepare_feature_tensors_25hz.py` for each feature set.
5. Run `noxi_fit_apply_feature_transform.py` for raw, PCA, and random projection branches.
6. Sync `outputs/` and selected `processed/` artifacts back to persistent storage.

Future documents will track:

```text
analysis steps
training runs
validation results
interpretation
```

## Baseline Modelling Step Added

The first baseline model scripts have been added. They are designed to consume any transformed manifest:

```text
raw
pca
random_projection
```

This works because all transform branches write the same tensor contract:

```text
x             [time, features]
y             [time]
target_mask   [time]
```

### TCN Baseline

Script:

```text
scripts/train_tcn.py
```

Design:

```text
create sequence windows lazily during training
predict engagement for every frame in each window
average overlapping validation predictions back to full sessions
report CCC/MAE/RMSE/Pearson overall and by dataset/role/session
```

Reason for lazy windows:

```text
avoids duplicating overlapping windows on disk
allows different models to choose different window sizes
keeps UCloud storage lower
```

### XGBoost Baseline

Script:

```text
scripts/train_xgboost.py
```

Design:

```text
create 20s/5s windows
summarize each window into mean/std feature descriptors
optionally add min/max descriptors
predict one scalar mean engagement value per window
average overlapping window predictions back to full-session frame predictions
```

Reason:

```text
XGBoost is a strong tabular baseline
it provides a useful comparison against sequence models
it can train on raw, PCA, or random-projection feature branches
```

### Example Commands

```powershell
python scripts\train_tcn.py --manifest outputs\manifests\model_processed_manifest_audio_egemaps_raw.csv --run-name egemaps_raw_tcn
python scripts\train_tcn.py --manifest outputs\manifests\model_processed_manifest_audio_egemaps_pca8.csv --run-name egemaps_pca8_tcn
python scripts\train_xgboost.py --manifest outputs\manifests\model_processed_manifest_audio_egemaps_raw.csv --run-name egemaps_raw_xgb
python scripts\train_xgboost.py --manifest outputs\manifests\model_processed_manifest_audio_egemaps_rp8.csv --run-name egemaps_rp8_xgb
```
