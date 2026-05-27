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

Generated folders are currently visible to git for this project hand-off:

```text
processed/
outputs/
models/
```

Raw external cache folders should still be treated carefully because full feature caches can become too large for normal GitHub pushes.

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

### `src/acm_pipeline/dyadic_data.py`

Shared dyadic model-input utilities:

```text
dyadic manifest loading
dyadic NPZ tensor validation
lazy dyadic sequence-window indexing
dyadic XGBoost window-summary table construction
```

The dyadic tensor contract is:

```text
x             [time, 2 * features]
y             [time, 2]
target_mask   [time, 2]
role_order    ["novice", "expert"]
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
output: [batch, time] or [batch, time, output_dim]
```

### `src/acm_pipeline/models_transformer.py`

Defines a small encoder-only Transformer for frame-level sequence regression.

Input/output:

```text
input:  [batch, features, time]
output: [batch, time] or [batch, time, output_dim]
```

The model projects input features to `d_model`, adds positional encoding, applies Transformer encoder layers, and predicts one engagement value per frame.

### `src/acm_pipeline/train_utils.py`

Shared training/evaluation output helpers:

```text
grouped metrics by overall/dataset/role/session
frame-level validation prediction export
CSV writing
```

### `src/acm_pipeline/dyadic_train_utils.py`

Shared dyadic training/evaluation output helpers:

```text
metrics by overall/role-channel/dataset/session
long-format dyadic validation prediction export
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

### `scripts/noxi_fit_apply_feature_transform_by_role.py`

Purpose: fit role-specific normalization and optional dimensionality reduction.

This script is parallel to `noxi_fit_apply_feature_transform.py`, but it fits separate transform objects for:

```text
novice
expert
```

This supports dyadic experiments where novice and expert features are compressed separately before being concatenated at each time step.

Primary outputs:

```text
processed/transformed/<feature_set>_<method>_by_role/
outputs/manifests/model_processed_manifest_<feature_set>_<method>_by_role.csv
outputs/transforms/<feature_set>_<method>_by_role/novice/
outputs/transforms/<feature_set>_<method>_by_role/expert/
```

### `scripts/noxi_build_dyadic_tensors.py`

Purpose: fuse role-level transformed tensors into time-aligned dyadic session tensors.

Input:

```text
any role-level transformed manifest
```

Output tensor contract:

```text
x           [time, 2 * feature_dim]
y           [time, 2]
target_mask [time, 2]
role_order  ["novice", "expert"]
```

Further details live in:

```text
docs/05_dyadic_representation.md
```

### `scripts/train_tcn.py`

Purpose: train a frame-level TCN from any transformed manifest.

Architecture notes and experiment tracking live in:

```text
docs/03_tcn_architecture.md
```

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

### `scripts/train_tcn_dyadic.py`

Purpose: train a frame-level TCN from any dyadic manifest.

Input/output:

```text
input x:  [batch, 2 * features, time]
output:   [batch, time, 2]
targets:  [batch, time, 2]
```

The two output channels are novice and expert engagement. Validation windows are averaged back to full dyadic sessions and metrics are reported overall, by role channel, by dataset, and by session.

Head variants:

```text
--head-type shared         # one 2-channel prediction head
--head-type role_specific  # one 1-channel prediction head per role
```

Both variants use the same dyadic TCN encoder, so this is a focused test of the output mapping.

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

### `scripts/train_xgboost_dyadic.py`

Purpose: train a tabular XGBoost baseline from any dyadic manifest.

Each dyadic window is summarized into fixed descriptors, and the model predicts two scalar window targets:

```text
[novice_mean_engagement, expert_mean_engagement]
```

The predictions are expanded back over covered frames and averaged across overlapping windows.

### `scripts/train_transformer.py`

Purpose: train an encoder-only Transformer from any transformed manifest.

Architecture notes and experiment tracking live in:

```text
docs/04_transformer_architecture.md
```

Like the TCN, the Transformer uses lazy sequence windows and reconstructs validation predictions by averaging overlapping window outputs back onto full sessions.

Primary outputs:

```text
outputs/experiments/<run_name>/model_best.pt
outputs/experiments/<run_name>/config.json
outputs/experiments/<run_name>/training_log.csv
outputs/experiments/<run_name>/val_predictions.csv
outputs/experiments/<run_name>/metrics_*.csv
```

### `scripts/train_transformer_dyadic.py`

Purpose: train an encoder-only Transformer from any dyadic manifest.

It follows the same windowing, masking, reconstruction, and metric layout as `train_tcn_dyadic.py`, but uses the Transformer encoder model.

## Current Modelling Code

The ACM repo now contains both role-level and dyadic baseline trainers for TCN, Transformer, and XGBoost. The role-level scripts remain useful baselines; the dyadic scripts should be used when modelling both people as a time-aligned interaction sequence.
