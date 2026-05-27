# TCN Modelling

This document tracks the overall TCN experiment setup and the runs we try. It is separate from `03_tcn_architecture.md`, which explains what the architecture parameters mean.

## Purpose

Use TCN models as the first sequence-native baselines for continuous engagement regression.

The TCN experiments should answer:

```text
does dyadic time alignment improve over role-level modelling?
does a shared dyadic output head work as well as role-specific heads?
how do raw, PCA, and random-projection inputs compare?
how do shared versus role-specific dimensionality reduction branches compare?
```

## Data Setup

The first stable feature set is:

```text
audio_egemaps
```

The first dyadic manifest used for smoke testing is:

```text
outputs/manifests/model_processed_manifest_audio_egemaps_raw_dyadic.csv
```

The dyadic input format is:

```text
x           [time, 2 * feature_dim]
y           [time, 2]
target_mask [time, 2]
role_order  ["novice", "expert"]
```

The two target channels are:

```text
channel 0 = novice engagement
channel 1 = expert engagement
```

## Common Training Setup

Default temporal setup:

```text
sample rate: 25 Hz
window size: 500 frames = 20 seconds
stride: 125 frames = 5 seconds
prediction: frame-level sequence-to-sequence regression
validation: average overlapping window predictions back to full sessions
```

Primary metric:

```text
CCC
```

Additional metrics:

```text
MAE
RMSE
Pearson
```

Reported groupings:

```text
overall
role channel
dataset
session
```

The grouping is important because an overall CCC can hide whether the model works better for novice or expert targets.

## Current TCN Scripts

Role-level TCN:

```text
scripts/train_tcn.py
```

Dyadic TCN:

```text
scripts/train_tcn_dyadic.py
```

The dyadic script supports two output-head variants:

```text
--head-type shared
--head-type role_specific
```

Shared head:

```text
one TCN encoder
one 2-channel prediction head
predicts novice and expert engagement jointly
```

Role-specific heads:

```text
one TCN encoder
one prediction head for novice
one prediction head for expert
heads are concatenated back to [novice, expert]
```

This keeps the temporal encoder identical and changes only the final prediction mapping.

## First Planned Comparison

The immediate comparison is:

```text
dyadic TCN, raw eGeMAPS, shared head
dyadic TCN, raw eGeMAPS, role-specific heads
```

Reason:

```text
both runs use the same dyadic input
both runs use the same encoder settings
only the output head changes
```

This is a clean first test of whether novice and expert benefit from separate output mappings.

## Example Commands

Shared dyadic head:

```powershell
python scripts\train_tcn_dyadic.py `
  --manifest outputs\manifests\model_processed_manifest_audio_egemaps_raw_dyadic.csv `
  --head-type shared `
  --run-name egemaps_raw_dyadic_tcn_shared
```

Role-specific dyadic heads:

```powershell
python scripts\train_tcn_dyadic.py `
  --manifest outputs\manifests\model_processed_manifest_audio_egemaps_raw_dyadic.csv `
  --head-type role_specific `
  --run-name egemaps_raw_dyadic_tcn_role_heads
```

For full UCloud runs, use the same commands without smoke-test limits. For local wiring checks, add small limits such as:

```powershell
--epochs 1 --max-train-windows 8 --batch-size 4 --hidden-channels 8 --levels 1
```

## Smoke Runs Completed

These runs only check that the code paths work. They should not be interpreted as model performance.

| Date | Run Name | Manifest | Head Type | Window | Stride | Channels | Levels | Epochs | Train Windows | Val CCC | Notes |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 2026-05-27 | `smoke_tcn_dyadic_shared_head` | `model_processed_manifest_audio_egemaps_raw_dyadic.csv` | `shared` | 500 | 125 | 8 | 1 | 1 | 8 | 0.12774 | Wiring check only. |
| 2026-05-27 | `smoke_tcn_dyadic_role_heads` | `model_processed_manifest_audio_egemaps_raw_dyadic.csv` | `role_specific` | 500 | 125 | 8 | 1 | 1 | 8 | 0.01804 | Wiring check only. |

## Experiment Log Template

Add full training runs here.

| Date | Run Name | Manifest | Representation | Transform | Head Type | Window | Stride | Channels | Levels | Kernel | Dropout | CCC Weight | Val CCC | Novice CCC | Expert CCC | Notes |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|

## Next Experiments

Suggested order:

1. Run full dyadic raw eGeMAPS TCN with shared head.
2. Run full dyadic raw eGeMAPS TCN with role-specific heads.
3. Compare overall CCC and role-channel CCC.
4. Repeat the better head setup on shared PCA and role-specific PCA branches.
5. Add random projection only after PCA/raw behavior is understood.

Keep the first full runs conservative:

```text
hidden_channels: 32 or 64
levels: 3 or 4
kernel_size: 5
dropout: 0.2
ccc_weight: 0.5
```
