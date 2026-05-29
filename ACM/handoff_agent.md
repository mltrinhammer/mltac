# Hand-Off: Agent Workflow For ACM / NOXI Training

## Context

Repository:

```text
C:\Users\anec\projects\mltac\ACM
https://github.com/mltrinhammer/mltac.git
branch: main
latest known pushed commit: aa7cfa8 Add ACM gated pooled TCN experiment
```

The ACM codebase contains reusable modelling code for NOXI / NOXI-J engagement regression. The immediate task is to use the actual data available to the collaborator, integrate or adapt the official organizer repository for input loading, then train and evaluate the ACM models one at a time.

Use the organizer repository mainly as a data-layout, split, config, and official-evaluation/submission compatibility layer. Keep ACM as the modelling core.

## Integration Principle

Do not replace the ACM model scripts with the organizer baseline. Build a thin adapter:

```text
organizer-format data
  -> ACM-compatible manifests/tensors
  -> ACM training scripts
  -> ACM metrics/diagnostics
  -> optional organizer-format prediction/submission output
```

The stable ACM dyadic tensor contract is:

```text
x           [time, 2 * feature_dim]
y           [time, 2]
target_mask [time, 2]
role_order  ["novice", "expert"]
```

For raw eGeMAPS:

```text
feature_dim per role = 88
x dim = 176
target channel 0 = novice
target channel 1 = expert
```

Windows are created lazily by ACM loaders and must never cross session boundaries.

## Relevant ACM Files

Preprocessing and tensor preparation:

```text
scripts/noxi_prepare_feature_tensors_25hz.py
scripts/noxi_fit_apply_feature_transform.py
scripts/noxi_fit_apply_feature_transform_by_role.py
scripts/noxi_build_dyadic_tensors.py
```

Dyadic data and training utilities:

```text
src/acm_pipeline/dyadic_data.py
src/acm_pipeline/dyadic_train_utils.py
src/acm_pipeline/metrics.py
src/acm_pipeline/models_tcn.py
```

Model scripts:

```text
scripts/train_tcn_dyadic.py
scripts/train_transformer_dyadic.py
scripts/train_xgboost_dyadic.py
scripts/train_tcn_partner_lag.py
scripts/train_tcn_attention.py
scripts/train_tcn_gated_pool.py
```

Main expected dyadic manifest path:

```text
outputs/manifests/model_processed_manifest_audio_egemaps_raw_dyadic.csv
```

## Loading The Actual Data

1. Inspect the organizer repository data loader, dataset configs, and split definitions.
2. Identify the actual feature and label files for NOXI / NOXI-J.
3. Map organizer session IDs, roles, features, labels, and masks into ACM tensor files.
4. Emit or update an ACM manifest where each row points to one session-level tensor file.
5. Confirm that each session tensor contains `x`, `y`, `target_mask`, and role metadata.
6. Confirm that role order is stable: channel 0 is novice and channel 1 is expert.
7. Confirm that no generated training window crosses a session boundary.
8. Do not commit raw data, large derived tensors, private paths, or credentials.

If the organizer repo produces feature arrays directly, the adapter can bypass parts of ACM preprocessing as long as the final manifest/tensor contract is identical.

## Suggested Training Order

Start with the simplest dyadic model, then add interaction-specific variants.

### 1. Dyadic TCN, Role-Specific Heads

Purpose: baseline dyadic regression with separate novice/expert heads.

```powershell
python scripts\train_tcn_dyadic.py `
  --manifest outputs\manifests\model_processed_manifest_audio_egemaps_raw_dyadic.csv `
  --head-type role_specific `
  --run-name noxi_egemaps_raw_dyadic_role_heads
```

### 2. Partner-Lag TCN

Purpose: test whether earlier partner behaviour helps predict target-role engagement.

```powershell
python scripts\train_tcn_partner_lag.py `
  --manifest outputs\manifests\model_processed_manifest_audio_egemaps_raw_dyadic.csv `
  --partner-lags -75 -750 `
  --run-name noxi_egemaps_raw_partner_lag_3s_30s
```

Lag convention:

```text
-75  = partner 3 seconds back at 25 Hz
-750 = partner 30 seconds back at 25 Hz
```

Avoid same-time partner context for interpretability unless explicitly running an ablation.

### 3. Attention TCN

Purpose: let the model search over a past interaction window.

```powershell
python scripts\train_tcn_attention.py `
  --manifest outputs\manifests\model_processed_manifest_audio_egemaps_raw_dyadic.csv `
  --attention-context joint `
  --attention-past-frames 1500 `
  --exclude-current-frame `
  --save-attention `
  --run-name noxi_egemaps_raw_attention_joint_60s
```

Diagnostics from `--save-attention` include attention by lag, source, session phase, and top-k attended frames.

### 4. Gated Pooled TCN

Purpose: use the partner's recent past as a pooled context and learn how much to blend it into the target role's prediction.

```powershell
python scripts\train_tcn_gated_pool.py `
  --manifest outputs\manifests\model_processed_manifest_audio_egemaps_raw_dyadic.csv `
  --partner-pool-frames 750 `
  --gate-type scalar `
  --save-gates `
  --run-name noxi_egemaps_raw_gated_pool_30s
```

Also consider `--partner-pool-frames 1500` for a 60 second partner-history condition.

### 5. MoE Later

The next planned architecture is an interaction Mixture-of-Experts model. Suggested first experts:

```text
own_only
partner_lag_3s
partner_lag_30s
partner_pool_30s_or_60s
```

The router should learn per-role/per-time weights over these experts, with diagnostics analogous to the gated model:

```text
router_by_role.csv
router_by_expert.csv
router_by_session.csv
router_by_session_phase.csv
router_timeseries_sample.csv
```

Do not implement this until the actual data loading and existing model training path are stable.

## Evaluation Requirements

Use ACM regression metrics, especially CCC, and always respect `target_mask`.

Report:

```text
overall / mean CCC
novice CCC
expert CCC
MAE or MSE if available
run config
diagnostic files produced
```

For each completed run, add a short note using:

```text
docs/tcn_evaluation_template.md
```

Minimum run note fields:

```text
Run:
Model:
Settings:
Metrics:
Diagnostics:
Key insights:
Next action:
```

## Smoke-Test Acceptance Criteria

Before long training:

```text
1. Organizer-format data is readable on the collaborator machine.
2. ACM-compatible dyadic manifest is produced.
3. At least one tensor can be loaded and inspected.
4. x, y, target_mask, and role_order match the ACM contract.
5. A tiny dyadic TCN run completes.
6. Metrics are written.
7. No window crosses a session boundary.
```

After that, run the full dyadic role-specific TCN before the interaction variants.

## Existing Smoke Results In ACM

These were wiring checks only, not meaningful model performance:

```text
smoke_tcn_dyadic_shared_head:        val_ccc=0.12774
smoke_tcn_dyadic_role_heads:         val_ccc=0.01804
smoke_tcn_partner_lag_raw:           val_ccc=0.08297
smoke_tcn_attention_joint_diag_fast: val_ccc=-0.07045
smoke_tcn_gated_pool_raw:            val_ccc=0.09410
```

Do not use these as benchmark results. They only confirm that scripts are wired.

