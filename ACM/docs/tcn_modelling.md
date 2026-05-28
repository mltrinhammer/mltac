# TCN Modelling

This document tracks the overall TCN experiment setup and the runs we try. It is separate from `03_tcn_architecture.md`, which explains what the architecture parameters mean.

## Purpose

Use TCN models as the first sequence-native baselines for continuous engagement regression.

The TCN experiments should answer:

```text
does dyadic time alignment improve over role-level modelling?
does a shared dyadic output head work as well as role-specific heads?
do separate role encoders with lagged partner context improve interaction modelling?
does attention over self, partner, or joint history improve the role-specific TCN?
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

Partner-lag TCN:

```text
scripts/train_tcn_partner_lag.py
```

Attention TCN:

```text
scripts/train_tcn_attention.py
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

The partner-lag script supports a more explicit interaction setup:

```text
novice features -> novice TCN encoder -> novice hidden sequence
expert features -> expert TCN encoder -> expert hidden sequence
```

Then each role head uses target-role hidden states and lagged partner hidden states:

```text
novice head: novice_hidden_t + expert_hidden_lags -> novice_engagement_t
expert head: expert_hidden_t + novice_hidden_lags -> expert_engagement_t
```

Lag convention:

```text
--partner-lags -25 0 25
-25 = partner one second earlier at 25 Hz
0   = partner same frame
25  = partner one second later at 25 Hz
```

Positive lags are offline-only because they use future partner context.

The attention script supports self, partner, and joint attention over a past window:

```text
self:    target role attends to its own hidden history
partner: target role attends to partner hidden history
joint:   target role attends to own + partner hidden history
```

The main attention-window option is:

```text
--attention-past-frames 1500
```

At 25 Hz, 1500 frames is 60 seconds.

If `--save-attention` is used, the best checkpoint writes:

```text
attention_by_lag.csv
attention_by_lag_bin.csv
attention_by_source.csv
attention_by_session_phase.csv
attention_topk.csv
```

The primary timing diagnostic is `relative_lag_frames`. Negative values mean the model attended to a source frame before the prediction time.

## First Planned Comparison

The immediate comparison is:

```text
dyadic TCN, raw eGeMAPS, shared head
dyadic TCN, raw eGeMAPS, role-specific heads
partner-lag TCN, raw eGeMAPS, separate role encoders and separate role heads
attention TCN, raw eGeMAPS, self/partner/joint context variants
```

Reason:

```text
the first two runs use the same dyadic input and encoder settings, changing only the output head
the partner-lag run uses the same dyadic input but tests explicit role-specific encoders and partner timing
the attention runs test whether learned weighting over recent self/partner hidden states helps beyond fixed lag choices
```

Together, these runs separate three questions: whether novice/expert need separate output mappings, whether fixed lagged partner context helps, and whether learned attention over recent history helps.

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

Partner-lag TCN with one-second past/current/future partner context:

```powershell
python scripts\train_tcn_partner_lag.py `
  --manifest outputs\manifests\model_processed_manifest_audio_egemaps_raw_dyadic.csv `
  --partner-lags -25 0 25 `
  --run-name egemaps_raw_partner_lag_tcn
```

Joint attention over the previous minute:

```powershell
python scripts\train_tcn_attention.py `
  --manifest outputs\manifests\model_processed_manifest_audio_egemaps_raw_dyadic.csv `
  --attention-context joint `
  --attention-past-frames 1500 `
  --save-attention `
  --run-name egemaps_raw_joint_attention_tcn
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
| 2026-05-28 | `smoke_tcn_partner_lag_raw` | `model_processed_manifest_audio_egemaps_raw_dyadic.csv` | `separate_role_encoders_heads_lags_-25_0_25` | 500 | 125 | 8 | 1 | 1 | 8 | 0.08297 | Wiring check only. |
| 2026-05-28 | `smoke_tcn_attention_joint_diag_fast` | `model_processed_manifest_audio_egemaps_raw_dyadic.csv` | `joint_attention_past_50` | 125 | 5000 | 8 | 1 | 1 | 4 | -0.07045 | Wiring check only; diagnostics enabled. |

## Experiment Log Template

Add full training runs here.

| Date | Run Name | Manifest | Representation | Transform | Encoder Setup | Head Setup | Partner Lags | Attention Context | Attention Window | Window | Stride | Channels | Levels | Kernel | Dropout | CCC Weight | Val CCC | Novice CCC | Expert CCC | Notes |
|---|---|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|

## Next Experiments

Suggested order:

1. Run full dyadic raw eGeMAPS TCN with shared head.
2. Run full dyadic raw eGeMAPS TCN with role-specific heads.
3. Run full partner-lag raw eGeMAPS TCN with lags `-25 0 25`.
4. Run attention TCN variants on raw eGeMAPS: `self`, `partner`, and `joint`.
5. Compare overall CCC, role-channel CCC, and attention diagnostics.
6. Repeat the strongest setup on shared PCA and role-specific PCA branches.
7. Add random projection only after PCA/raw behavior is understood.

Keep the first full runs conservative:

```text
hidden_channels: 32 or 64
levels: 3 or 4
kernel_size: 5
dropout: 0.2
ccc_weight: 0.5
```
