# ACM Engagement Modelling Pipeline

Clean preprocessing and modelling code for NOXI / NOXI-J continuous engagement regression.

The repository is intended to contain code, configs, and lightweight documentation only. Keep cached streams, processed tensors, transforms, checkpoints, and prediction files out of git.

## Feature Sets

The pipeline currently supports these full stream sets:

```text
audio_egemaps      -> audio.egemapsv2
audio_w2vbert2     -> audio.w2vbert2_embeddings
visual_swin        -> swin
visual_openface    -> openface2 + openface3
visual_openpose    -> openpose
```

## Preprocessing Flow

1. Align one feature set to the 25 Hz engagement target grid.
2. Fit train-only normalization.
3. Export one transform branch:
   - `raw`: normalized features, no dimensionality reduction
   - `pca`: normalized features followed by PCA
   - `random_projection`: normalized features followed by Gaussian random projection

Example:

```powershell
python scripts\noxi_prepare_feature_tensors_25hz.py --feature-set audio_egemaps
python scripts\noxi_fit_apply_feature_transform.py --input-manifest outputs\manifests\model_processed_manifest_audio_egemaps_25hz.csv --method raw
python scripts\noxi_fit_apply_feature_transform.py --input-manifest outputs\manifests\model_processed_manifest_audio_w2vbert2_25hz.csv --method pca --n-components 128
```

## Baseline Models

Both baseline trainers consume the same transformed manifest contract:

```text
tensor_relative_path -> NPZ with x, y, target_mask
n_features
model_split
```

That means the same scripts can train on `raw`, `pca`, or `random_projection` branches.

```powershell
python scripts\train_tcn.py --manifest outputs\manifests\model_processed_manifest_audio_egemaps_raw.csv --run-name egemaps_raw_tcn
python scripts\train_xgboost.py --manifest outputs\manifests\model_processed_manifest_audio_egemaps_pca8.csv --run-name egemaps_pca8_xgb
```

## Living Documentation

The project documentation is intended to grow with the pipeline:

```text
docs/01_codebase_structure.md
docs/02_preprocessing_progress.md
```

Later UCloud training runs should add separate notes for analysis steps, results, and interpretation.
