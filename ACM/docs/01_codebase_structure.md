# Codebase Structure

This repository contains the clean, GitHub-ready pipeline for preparing NOXI / NOXI-J features and training continuous engagement models. Large data artifacts are intentionally excluded from git.

## Repository Layout

```text
ACM/
  README.md
  requirements.txt
  scripts/
  src/acm_pipeline/
  docs/
```

Generated folders are ignored by git:

```text
cache/
processed/
outputs/
models/
```

## Source Package

### `src/acm_pipeline/feature_registry.py`

Defines the supported feature sets and their underlying stream names.

Current feature sets:

```text
audio_egemaps      -> audio.egemapsv2
audio_w2vbert2     -> audio.w2vbert2_embeddings
visual_swin        -> swin
visual_openface    -> openface2 + openface3
visual_openpose    -> openpose
```

The registry is used by preprocessing scripts so stream combinations are defined in one place.

### `src/acm_pipeline/io.py`

Shared input/output helpers:

```text
CSV manifest reading/writing
cache path resolution
engagement target loading
SSI .stream header parsing
binary stream loading
```

The stream loader expects SSI-style files:

```text
example.stream   # text metadata, including sr and dim
example.stream~  # raw float32 feature matrix
```

### `src/acm_pipeline/alignment.py`

Aligns stream matrices to the 25 Hz engagement target grid.

Main rule:

```text
if frame count nearly matches target length: trust frame count and truncate
else: use declared sample rate and linear interpolation
```

This handles known cases where stream sample-rate metadata can be misleading while frame counts are already aligned to the target.

### `src/acm_pipeline/transforms.py`

Reusable transform utilities:

```text
train-only z-score normalization
frame sampling for reducer fitting
PCA fitting
Gaussian random projection fitting
transform serialization
```

Normalization is fitted on the training split only, then applied unchanged to validation/test rows.

### `src/acm_pipeline/data.py`

Shared model-input utilities:

```text
processed/transformed manifest loading
NPZ tensor loading
lazy sequence-window indexing
XGBoost window-summary table construction
```

The key design is that models consume the same tensor contract regardless of whether the input branch is raw, PCA, or random projection:

```text
x             [time, features]
y             [time]
target_mask   [time]
```

### `src/acm_pipeline/metrics.py`

Shared regression metrics and losses:

```text
CCC
MAE
RMSE
Pearson
masked MSE loss
CCC loss
```

CCC is the primary validation metric.

### `src/acm_pipeline/models_tcn.py`

Defines a small residual TCN for frame-level sequence regression.

Input/output:

```text
input:  [batch, features, time]
output: [batch, time]
```

### `src/acm_pipeline/train_utils.py`

Shared training/evaluation output helpers:

```text
grouped metrics by overall/dataset/role/session
frame-level validation prediction export
CSV writing
```

## Scripts

### `scripts/noxi_prepare_feature_tensors_25hz.py`

Purpose: create aligned 25 Hz tensors for one feature set.

Inputs:

```text
raw modelling manifest with split labels
stream manifest with paths, stream names, sr, dim
local cache containing .stream and .stream~ files
```

Outputs:

```text
processed/<feature_set>_25hz/<dataset>/<session>/<role>.<feature_set>.25hz.npz
outputs/manifests/model_processed_manifest_<feature_set>_25hz.csv
outputs/manifests/feature_status_<feature_set>_25hz.csv
```

Each NPZ contains:

```text
x
y
target_mask
stream_names
stream_dims
stream_source_rates
stream_alignment_methods
sample_rate_hz
feature_set
```

### `scripts/noxi_fit_apply_feature_transform.py`

Purpose: turn aligned tensors into model-input tensors for one transform branch.

Supported methods:

```text
raw                 # z-score normalization only
pca                 # z-score normalization + PCA
random_projection   # z-score normalization + Gaussian random projection
```

Outputs:

```text
processed/transformed/<feature_set>_<method>/
outputs/manifests/model_processed_manifest_<feature_set>_<method>.csv
outputs/transforms/<feature_set>_<method>/normalizer.npz
outputs/transforms/<feature_set>_<method>/transform_config.json
```

Additional PCA output:

```text
outputs/transforms/<feature_set>_pca<n>/pca.pkl
outputs/transforms/<feature_set>_pca<n>/pca_explained_variance.csv
```

Additional random projection output:

```text
outputs/transforms/<feature_set>_rp<n>/random_projection.pkl
```

### `scripts/train_tcn.py`

Purpose: train a frame-level TCN from any transformed manifest.

Windowing is done lazily inside the data loader:

```text
aligned/transformed session tensor
-> window indices in memory
-> fixed-length batches
-> frame-level predictions
-> overlapping validation windows averaged back to full sessions
```

Primary outputs:

```text
outputs/experiments/<run_name>/model_best.pt
outputs/experiments/<run_name>/config.json
outputs/experiments/<run_name>/training_log.csv
outputs/experiments/<run_name>/val_predictions.csv
outputs/experiments/<run_name>/metrics_*.csv
```

### `scripts/train_xgboost.py`

Purpose: train a tabular XGBoost baseline from any transformed manifest.

Because XGBoost is not sequence-native, each window is summarized into fixed descriptors:

```text
mean per feature
std per feature
optional min/max per feature
```

The model predicts one scalar engagement value per window. Validation predictions are assigned back over the covered frames and averaged across overlapping windows, so reported metrics are still frame-level and comparable with the TCN.

Primary outputs:

```text
outputs/experiments/<run_name>/model.pkl
outputs/experiments/<run_name>/config.json
outputs/experiments/<run_name>/val_predictions.csv
outputs/experiments/<run_name>/metrics_*.csv
```

## Current Modelling Code

The first TCN baseline was developed in the exploratory `Noxi_Noxij` directory. The clean ACM repo currently focuses on preprocessing and transform branches. The next step is to port the modelling modules into this repo once the preprocessing interface is stable.
