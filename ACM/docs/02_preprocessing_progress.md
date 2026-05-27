# Preprocessing Progress

This document tracks the data-preparation pipeline: what has been built, why it was done, and which scripts/outputs correspond to each step. Model architecture details and experiment logs are kept in separate modelling documents.

## Current Purpose

Prepare consistent NOXI / NOXI-J inputs that can be reused locally and on UCloud across audio and visual feature-set experiments.

The preprocessing pipeline should produce:

```text
role-level tensors for single-person baselines
dyadic tensors for interaction-aware models
raw, PCA, and random-projection transform branches
shared-transform and role-specific-transform variants
```

## Feature Sets

We decided to work with full feature sets rather than selecting individual OpenFace/OpenPose dimensions, because the available stream files do not provide reliable per-dimension feature names.

Current feature sets:

```text
audio_egemaps
audio_w2vbert2
visual_swin
visual_openface
visual_openpose
```

Defined in:

```text
src/acm_pipeline/feature_registry.py
```

## Alignment

The common aligned representation is 25 Hz.

Reason:

```text
the engagement target is effectively on a 25 Hz grid
most available streams are listed as 25 Hz
the previous eGeMAPS baseline used 25 Hz successfully
```

The alignment logic first checks whether the feature frame count already closely matches the target length. If it does, the pipeline trusts frame count and truncates to the shared length. Otherwise, it uses the declared sample rate and linear interpolation.

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

Role-level tensor contract:

```text
x             [time, features]
y             [time]
target_mask   [time]
```

## Normalization

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

Reason:

```text
PCA is scale-sensitive
random projection behaves better with comparable feature scales
TCN and Transformer optimization are more stable with normalized inputs
different streams have different numeric ranges and meanings
```

Implemented in:

```text
src/acm_pipeline/transforms.py
scripts/noxi_fit_apply_feature_transform.py
scripts/noxi_fit_apply_feature_transform_by_role.py
```

## Transform Branches

Supported branches:

```text
raw
pca
random_projection
```

PCA was included because it is common, interpretable, and writes an explained-variance table for deciding component counts.

Random projection was included as a simple additional compression baseline. It reduces dimensionality without learning variance structure from the data.

Deferred for later:

```text
learned projection inside the model
autoencoder
other supervised or nonlinear reducers
```

## Shared vs Role-Specific Transforms

Shared transform script:

```text
scripts/noxi_fit_apply_feature_transform.py
```

This fits one normalizer/reducer on all training frames from novice and expert roles together, then applies the same transform to both roles.

Role-specific transform script:

```text
scripts/noxi_fit_apply_feature_transform_by_role.py
```

This fits separate transform objects for:

```text
novice
expert
```

Reason for keeping both:

```text
shared transforms keep novice/expert components on the same axes
role-specific transforms may preserve role-specific variance patterns better
the comparison is an empirical modelling question
```

## Dyadic Tensor Creation

We decided not to replace the role-level preprocessing pipeline. Instead, dyadic tensors are built as an additional branch after role-level transforms.

Script:

```text
scripts/noxi_build_dyadic_tensors.py
```

Reason:

```text
role-level models remain useful baselines
dyadic models avoid artificial temporal transitions between people
shared PCA/RP and role-specific PCA/RP can be compared cleanly
```

Dyadic tensor contract:

```text
x           [time, 2 * feature_dim]
y           [time, 2]
target_mask [time, 2]
role_order  ["novice", "expert"]
```

The dyadic builder groups by dataset/session, pairs novice and expert by frame index, concatenates features at each aligned time step, and writes one session tensor per dyad.

Detailed dyadic notes:

```text
docs/05_dyadic_representation.md
```

## Smoke Checks Completed

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

Shared raw transform:

```powershell
python scripts\noxi_fit_apply_feature_transform.py --input-manifest outputs\manifests\model_processed_manifest_audio_egemaps_25hz.csv --method raw
```

Shared PCA smoke:

```powershell
python scripts\noxi_fit_apply_feature_transform.py --input-manifest outputs\manifests\model_processed_manifest_audio_egemaps_25hz.csv --method pca --n-components 8 --max-fit-frames 2000
```

Shared random projection smoke:

```powershell
python scripts\noxi_fit_apply_feature_transform.py --input-manifest outputs\manifests\model_processed_manifest_audio_egemaps_25hz.csv --method random_projection --n-components 8 --max-fit-frames 2000
```

The shared PCA smoke run produced:

```text
outputs/transforms/audio_egemaps_pca8/pca_explained_variance.csv
```

The first 8 PCA components explained approximately 74.2% cumulative variance in the small smoke sample. This is only a smoke-test diagnostic, not a final modelling decision.

Dyadic smoke checks:

```powershell
python scripts\noxi_build_dyadic_tensors.py --input-manifest outputs\manifests\model_processed_manifest_audio_egemaps_raw.csv

python scripts\noxi_fit_apply_feature_transform_by_role.py --input-manifest outputs\manifests\model_processed_manifest_audio_egemaps_25hz.csv --method pca --n-components 8 --max-fit-frames 2000

python scripts\noxi_build_dyadic_tensors.py --input-manifest outputs\manifests\model_processed_manifest_audio_egemaps_pca8_by_role.csv
```

Observed dyadic outputs:

```text
audio_egemaps_raw_dyadic:
  output manifest: outputs/manifests/model_processed_manifest_audio_egemaps_raw_dyadic.csv
  dyadic rows: 69

audio_egemaps_pca8_by_role:
  output manifest: outputs/manifests/model_processed_manifest_audio_egemaps_pca8_by_role.csv
  transformed rows: 138

audio_egemaps_pca8_by_role_dyadic:
  output manifest: outputs/manifests/model_processed_manifest_audio_egemaps_pca8_by_role_dyadic.csv
  dyadic rows: 69
```

One inspected role-specific PCA dyadic tensor had:

```text
x:           (26642, 16)
y:           (26642, 2)
target_mask: (26642, 2)
role_order:  ["novice", "expert"]
```

## UCloud Preprocessing Plan

Once persistent UCloud storage is available:

1. Clone the GitHub repo.
2. Install requirements.
3. Populate `cache/` from persistent storage with the required `.stream` and `.stream~` files.
4. Run `noxi_prepare_feature_tensors_25hz.py` for each feature set.
5. Run shared transform branches with `noxi_fit_apply_feature_transform.py`.
6. Run role-specific transform branches with `noxi_fit_apply_feature_transform_by_role.py`.
7. Build dyadic manifests with `noxi_build_dyadic_tensors.py`.
8. Sync `outputs/`, selected `processed/`, and selected `models/` artifacts back to persistent storage.

## Modelling Documents

Preprocessing stops at model-ready manifests and tensors. Modelling setup and experiment tracking are documented separately:

```text
docs/03_tcn_architecture.md
docs/04_transformer_architecture.md
docs/05_dyadic_representation.md
docs/tcn_modelling.md
```
