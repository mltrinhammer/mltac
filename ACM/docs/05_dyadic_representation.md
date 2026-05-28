# Dyadic Representation

This document explains the dyadic session representation used for interaction-aware modelling. It also tracks the transform choices used before dyadic fusion.

## Why Dyadic Tensors Are Needed

Role-level tensors represent one person at a time:

```text
novice x: [time, features]
expert x: [time, features]
```

If those are vertically stacked as one sequence:

```text
rows 0...k     = novice over time
rows k+1...m   = expert over time
```

a temporal model sees an artificial transition:

```text
novice final frame -> expert first frame
```

That is not a real temporal event. For interaction modelling, both people should be represented at the same time step.

## Dyadic Format

For each session, pair the roles by frame index after 25 Hz alignment:

```text
X_t = [novice_features_t, expert_features_t]
y_t = [novice_engagement_t, expert_engagement_t]
```

Final tensor contract:

```text
x           [time, 2 * feature_dim]
y           [time, 2]
target_mask [time, 2]
role_order  ["novice", "expert"]
frame_idx   [time]
```

The first target channel is always novice, and the second target channel is always expert.

## Transform Order

Transforms happen **before** dyadic fusion.

```text
role-level aligned tensors
-> normalization / optional PCA or RP
-> dyadic fusion
```

This keeps the dyadic fusion script simple and lets us compare different transform branches with the same fusion logic.

## Shared vs Role-Specific Transforms

### Shared Transform

Existing shared transform script:

```text
scripts/noxi_fit_apply_feature_transform.py
```

Fit set:

```text
all train frames from novice + expert
```

Application:

```text
same normalizer/PCA/RP applied to both roles
```

Dyadic result:

```text
novice_sharedPCA [T, K] + expert_sharedPCA [T, K]
-> dyadic [T, 2K]
```

Benefit:

```text
same latent axes for both roles
symmetric and easy to compare
fewer fitted objects
```

Tradeoff:

```text
role-specific variance patterns may be compressed less well
```

### Role-Specific Transform

Role-specific transform script:

```text
scripts/noxi_fit_apply_feature_transform_by_role.py
```

Fit set:

```text
novice normalizer/PCA/RP fit on novice train frames only
expert normalizer/PCA/RP fit on expert train frames only
```

Application:

```text
novice transform applied to novice rows
expert transform applied to expert rows
```

Dyadic result:

```text
novice_rolePCA [T, K] + expert_rolePCA [T, K]
-> dyadic [T, 2K]
```

Benefit:

```text
captures role-specific feature distributions
may preserve novice/expert behavior patterns better
```

Tradeoff:

```text
novice PCA component 1 and expert PCA component 1 no longer mean the same latent direction
more fitted objects to track
```

## Dyadic Fusion Script

Script:

```text
scripts/noxi_build_dyadic_tensors.py
```

Input:

```text
any role-level transformed manifest
```

Examples:

```text
model_processed_manifest_audio_egemaps_raw.csv
model_processed_manifest_audio_egemaps_pca8.csv
model_processed_manifest_audio_egemaps_pca8_by_role.csv
model_processed_manifest_audio_egemaps_rp8.csv
```

Output:

```text
processed/dyadic/<branch_name>/<dataset>/<session_id>/<session_id>.<branch_name>.npz
outputs/manifests/model_processed_manifest_<branch_name>.csv
```

The same dyadic builder is used for raw, shared PCA/RP, and role-specific PCA/RP.

## Assertions and Checks

The dyadic builder checks:

```text
novice and expert both exist for each session
both roles have the same model split
role-level x/y/target_mask lengths match internally
features are finite
targets and masks are finite
novice/expert feature dimensions match
novice/expert frame counts differ by no more than the tolerance
x.shape[-1] == 2 * feature_dim
y.shape[-1] == 2
target_mask.shape[-1] == 2
```

Alignment is by frame index because the existing tensors are already aligned to the 25 Hz target grid. A tiny tail mismatch is handled by using the shared prefix.

The dyadic output is one NPZ file per dataset/session. Future dyadic data loaders should create windows within each session tensor only, which prevents windows from crossing session boundaries. Person boundaries are already removed from the temporal axis because novice and expert are fused into the same time step.

## Example Commands

Build dyadic tensors from shared raw branch:

```powershell
python scripts\noxi_build_dyadic_tensors.py `
  --input-manifest outputs\manifests\model_processed_manifest_audio_egemaps_raw.csv
```

Build role-specific PCA branch:

```powershell
python scripts\noxi_fit_apply_feature_transform_by_role.py `
  --input-manifest outputs\manifests\model_processed_manifest_audio_egemaps_25hz.csv `
  --method pca `
  --n-components 8 `
  --max-fit-frames 2000
```

Build dyadic tensors from role-specific PCA:

```powershell
python scripts\noxi_build_dyadic_tensors.py `
  --input-manifest outputs\manifests\model_processed_manifest_audio_egemaps_pca8_by_role.csv
```

Train dyadic baselines from a dyadic manifest:

```powershell
python scripts\train_tcn_dyadic.py `
  --manifest outputs\manifests\model_processed_manifest_audio_egemaps_raw_dyadic.csv `
  --head-type shared `
  --run-name egemaps_raw_dyadic_tcn_shared

python scripts\train_tcn_dyadic.py `
  --manifest outputs\manifests\model_processed_manifest_audio_egemaps_raw_dyadic.csv `
  --head-type role_specific `
  --run-name egemaps_raw_dyadic_tcn_role_heads

python scripts\train_tcn_partner_lag.py `
  --manifest outputs\manifests\model_processed_manifest_audio_egemaps_raw_dyadic.csv `
  --partner-lags -25 0 25 `
  --run-name egemaps_raw_partner_lag_tcn

python scripts\train_tcn_attention.py `
  --manifest outputs\manifests\model_processed_manifest_audio_egemaps_raw_dyadic.csv `
  --attention-context joint `
  --attention-past-frames 1500 `
  --save-attention `
  --run-name egemaps_raw_joint_attention_tcn

python scripts\train_tcn_gated_pool.py `
  --manifest outputs\manifests\model_processed_manifest_audio_egemaps_raw_dyadic.csv `
  --partner-pool-frames 750 `
  --save-gates `
  --run-name egemaps_raw_gated_pool_30s_tcn

python scripts\train_transformer_dyadic.py `
  --manifest outputs\manifests\model_processed_manifest_audio_egemaps_raw_dyadic.csv `
  --run-name egemaps_raw_dyadic_transformer

python scripts\train_xgboost_dyadic.py `
  --manifest outputs\manifests\model_processed_manifest_audio_egemaps_raw_dyadic.csv `
  --run-name egemaps_raw_dyadic_xgb
```

## Planned Comparisons

Representation comparisons:

```text
person-level raw
dyadic raw
dyadic shared PCA
dyadic role-specific PCA
dyadic shared random projection
dyadic role-specific random projection
```

Model comparisons:

```text
dyadic TCN with shared 2-channel head
dyadic TCN with one head per role
dyadic partner-lag TCN with separate role encoders and separate role heads
dyadic attention TCN with self/partner/joint context
dyadic gated pooled-context TCN
dyadic Transformer
dyadic XGBoost
partner-aware attention / cross-attention later
```

## Current Status

Implemented:

```text
role-specific raw/PCA/RP transform script
shared-transform dyadic fusion
role-specific-transform dyadic fusion
shape/assertion checks in dyadic builder
dyadic TCN trainer
shared and role-specific dyadic TCN heads
partner-lag dyadic TCN with separate role encoders and heads
attention dyadic TCN with optional attention diagnostics
gated pooled-context dyadic TCN with optional gate diagnostics
dyadic Transformer trainer
dyadic XGBoost trainer
dyadic metrics split by novice/expert channel
```

Smoke-tested:

```text
audio_egemaps_raw_dyadic
audio_egemaps_pca8_by_role
audio_egemaps_pca8_by_role_dyadic
```

Tiny dyadic model smoke tests on `audio_egemaps_raw_dyadic`:

```text
TCN shared head:         val_ccc=0.12774
TCN role-specific heads: val_ccc=0.01804
TCN partner lag:         val_ccc=0.08297
TCN joint attention:     val_ccc=-0.07045
TCN gated pool:          val_ccc=0.09410
Transformer:             val_ccc=-0.02809
XGBoost:                 val_ccc=0.30641
```

These smoke metrics only confirm that data loading, model output shapes, masking, reconstruction, and CSV writing work. They should not be interpreted as final model performance.

Next practical step:

```text
run full UCloud experiments for selected feature sets and transform branches
```
