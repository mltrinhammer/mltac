# ACM Engagement Modelling Pipeline

Clean preprocessing and modelling code for NOXI / NOXI-J continuous engagement regression.

The repository is intended to contain code, configs, documentation, and the currently generated pipeline artifacts so runs can be inspected and moved to UCloud. Very large raw caches should still live in persistent storage rather than git.

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

Role-specific transform and dyadic fusion examples:

```powershell
python scripts\noxi_fit_apply_feature_transform_by_role.py --input-manifest outputs\manifests\model_processed_manifest_audio_egemaps_25hz.csv --method pca --n-components 8
python scripts\noxi_build_dyadic_tensors.py --input-manifest outputs\manifests\model_processed_manifest_audio_egemaps_pca8_by_role.csv
```

## Baseline Models

The role-level baseline trainers consume the same transformed manifest contract:

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

Dyadic trainer variants consume dyadic manifests where the two people are aligned at each time step and `y` has novice/expert channels:

```powershell
python scripts\train_tcn_dyadic.py --manifest outputs\manifests\model_processed_manifest_audio_egemaps_raw_dyadic.csv --head-type shared --run-name egemaps_raw_dyadic_tcn_shared
python scripts\train_tcn_dyadic.py --manifest outputs\manifests\model_processed_manifest_audio_egemaps_raw_dyadic.csv --head-type role_specific --run-name egemaps_raw_dyadic_tcn_role_heads
python scripts\train_tcn_partner_lag.py --manifest outputs\manifests\model_processed_manifest_audio_egemaps_raw_dyadic.csv --partner-lags -25 0 25 --run-name egemaps_raw_partner_lag_tcn
python scripts\train_tcn_attention.py --manifest outputs\manifests\model_processed_manifest_audio_egemaps_raw_dyadic.csv --attention-context joint --attention-past-frames 1500 --save-attention --run-name egemaps_raw_joint_attention_tcn
python scripts\train_tcn_gated_pool.py --manifest outputs\manifests\model_processed_manifest_audio_egemaps_raw_dyadic.csv --partner-pool-frames 750 --save-gates --run-name egemaps_raw_gated_pool_30s_tcn
python scripts\train_transformer_dyadic.py --manifest outputs\manifests\model_processed_manifest_audio_egemaps_raw_dyadic.csv --run-name egemaps_raw_dyadic_transformer
python scripts\train_xgboost_dyadic.py --manifest outputs\manifests\model_processed_manifest_audio_egemaps_raw_dyadic.csv --run-name egemaps_raw_dyadic_xgb
```

## Living Documentation

The project documentation is intended to grow with the pipeline:

```text
docs/01_codebase_structure.md
docs/02_preprocessing_progress.md
docs/03_tcn_architecture.md
docs/04_transformer_architecture.md
docs/05_dyadic_representation.md
docs/tcn_modelling.md
docs/tcn_evaluation_template.md
```

Later UCloud training runs should add separate notes for analysis steps, results, and interpretation.
