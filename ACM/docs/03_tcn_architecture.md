# TCN Architecture Notes

This document explains the adjustable parts of the TCN baseline and keeps a running log of configurations tried. Update the second section whenever a new TCN experiment is run.

## Adjustable Architecture Components

### Input Features

Controlled by the manifest passed to:

```powershell
python scripts\train_tcn.py --manifest <manifest.csv>
```

Examples:

```text
raw eGeMAPS
PCA-reduced eGeMAPS
random-projection eGeMAPS
W2V-BERT2 PCA
Swin raw/PCA/RP
OpenFace raw/PCA/RP
OpenPose raw/PCA/RP
```

The TCN reads `n_features` from the manifest and sets:

```text
input_dim = n_features
```

Effect:

```text
larger input_dim gives the model more feature channels
larger input_dim also increases first-layer parameter count and overfitting risk
PCA/RP reduce input_dim before the TCN
```

### Window Size

CLI argument:

```powershell
--window-size 500
```

At 25 Hz:

```text
250 frames = 10 seconds
500 frames = 20 seconds
750 frames = 30 seconds
```

Effect:

```text
larger windows give the model more temporal context
larger windows use more memory and may include less locally relevant context
smaller windows train faster and focus on shorter dynamics
too-small windows may miss slow engagement changes
```

Current default:

```text
500 frames = 20 seconds
```

### Stride

CLI argument:

```powershell
--stride 125
```

At 25 Hz:

```text
125 frames = 5 seconds
250 frames = 10 seconds
```

Effect:

```text
smaller stride creates more overlapping windows
smaller stride gives denser training coverage but more compute
larger stride is faster but gives fewer training examples
validation predictions are averaged where windows overlap
```

Current default:

```text
125 frames = 5 seconds
```

### Hidden Channels / Filters

CLI argument:

```powershell
--hidden-channels 64
```

This is the number of convolutional channels in each TCN block.

Effect:

```text
more channels increase model capacity
more channels can capture richer feature interactions
more channels increase memory, compute, and overfitting risk
fewer channels are faster and safer for small datasets
```

Typical values to test:

```text
16, 32, 64, 128
```

Current default:

```text
64
```

### Number of Levels / Blocks

CLI argument:

```powershell
--levels 4
```

Each level is one residual temporal convolution block.

Effect:

```text
more levels increase temporal receptive field
more levels increase depth and capacity
too many levels can overfit or become harder to optimize
fewer levels are simpler and faster
```

Current default:

```text
4
```

### Kernel Size / Filter Width

CLI argument:

```powershell
--kernel-size 5
```

The kernel size is how many time steps each convolution sees before dilation.

At 25 Hz:

```text
kernel_size 3 covers 0.12 seconds before dilation
kernel_size 5 covers 0.20 seconds before dilation
kernel_size 9 covers 0.36 seconds before dilation
```

Effect:

```text
larger kernel size captures wider local temporal patterns
larger kernel size increases parameters and compute
smaller kernel size focuses on sharper local changes
kernel size interacts with dilation and levels to determine receptive field
```

Typical values to test:

```text
3, 5, 7, 9
```

Current default:

```text
5
```

### Dilation

Current implementation:

```text
dilation = 2 ** level_index
```

So:

```text
levels=4 -> dilations 1, 2, 4, 8
levels=5 -> dilations 1, 2, 4, 8, 16
```

Effect:

```text
dilation expands temporal context without downsampling
higher dilation lets later layers see farther back/forward in the window
too much dilation can skip over short local details
```

Current CLI control:

```text
adjust --levels to indirectly change the dilation sequence
```

Possible future CLI options:

```text
--dilation-base 2
--dilations 1 2 4 8
```

### Dropout

CLI argument:

```powershell
--dropout 0.2
```

Effect:

```text
higher dropout regularizes more strongly
higher dropout may help high-dimensional streams like W2V-BERT2 or Swin
too much dropout can underfit
lower dropout may work for compact inputs like eGeMAPS
```

Typical values:

```text
0.1, 0.2, 0.3, 0.5
```

Current default:

```text
0.2
```

### Loss Weighting

CLI argument:

```powershell
--ccc-weight 0.5
```

Current loss:

```text
masked MSE + ccc_weight * CCC loss
```

Effect:

```text
higher ccc_weight prioritizes concordance/correlation structure
lower ccc_weight prioritizes absolute frame-level error
CCC is the primary validation metric, so this weight is worth tuning
```

Typical values:

```text
0.0, 0.25, 0.5, 1.0
```

Current default:

```text
0.5
```

### Learning Rate and Weight Decay

CLI arguments:

```powershell
--lr 0.001
--weight-decay 0.0001
```

Effect:

```text
higher learning rate trains faster but can become unstable
lower learning rate is safer but slower
higher weight decay regularizes weights more strongly
too much weight decay can underfit
```

Current defaults:

```text
lr = 0.001
weight_decay = 0.0001
```

### Early Stopping

CLI arguments:

```powershell
--epochs 50
--patience 12
```

Effect:

```text
more epochs allow more fitting
patience stops training if validation CCC does not improve
early stopping helps reduce overfitting
```

Current defaults:

```text
epochs = 50
patience = 12
```

### Dyadic Output Head

CLI argument for dyadic TCN only:

```powershell
--head-type shared
--head-type role_specific
```

Shared head:

```text
one temporal encoder
one 2-channel Conv1d head
outputs [novice_engagement, expert_engagement]
```

Role-specific heads:

```text
one temporal encoder
one 1-channel Conv1d head for novice
one 1-channel Conv1d head for expert
outputs are concatenated back to [novice_engagement, expert_engagement]
```

Effect:

```text
shared head tests whether one output mapping is sufficient for both roles
role-specific heads test whether novice and expert need separate output mappings
the encoder stays identical, so the comparison isolates the final prediction layer
```

## Receptive Field Intuition

The receptive field is the approximate temporal span that can influence one output frame.

For the current two-convolution residual block and dilation sequence:

```text
larger kernel_size -> wider local context per block
more levels -> more dilated blocks
higher dilation -> wider temporal reach
larger window_size -> more available context, but the model's effective context is limited by receptive field
```

Example default:

```text
kernel_size = 5
levels = 4
dilations = 1, 2, 4, 8
```

Approximate receptive field:

```text
1 + 2 * (kernel_size - 1) * sum(dilations)
= 1 + 2 * 4 * (1 + 2 + 4 + 8)
= 121 frames
= 4.84 seconds at 25 Hz
```

This is smaller than the 20-second window. The full window still matters for batching and validation coverage, but each frame prediction mostly depends on a local receptive field unless the model is made deeper/wider in time.

## Current Default Command

```powershell
python scripts\train_tcn.py `
  --manifest outputs\manifests\model_processed_manifest_audio_egemaps_raw.csv `
  --window-size 500 `
  --stride 125 `
  --hidden-channels 64 `
  --levels 4 `
  --kernel-size 5 `
  --dropout 0.2 `
  --ccc-weight 0.5
```

## Experiment Log

Add one row per run.

| Date | Run Name | Manifest | Window | Stride | Channels | Levels | Kernel | Dilations | Dropout | Loss | Val CCC | Notes |
|---|---|---|---:|---:|---:|---:|---:|---|---:|---|---:|---|
| 2026-05-27 | `smoke_tcn_raw` | `model_processed_manifest_audio_egemaps_raw.csv` | 500 | 125 | 8 | 1 | 5 | 1 | 0.2 | MSE + 0.5 CCC | 0.01543 | Smoke test only: 1 epoch, 16 train windows. Not a real result. |
| 2026-05-27 | `smoke_tcn_dyadic_shared_head` | `model_processed_manifest_audio_egemaps_raw_dyadic.csv` | 500 | 125 | 8 | 1 | 5 | 1 | 0.2 | MSE + 0.5 CCC | 0.12774 | Smoke test only: shared 2-channel dyadic head, 1 epoch, 8 train windows. |
| 2026-05-27 | `smoke_tcn_dyadic_role_heads` | `model_processed_manifest_audio_egemaps_raw_dyadic.csv` | 500 | 125 | 8 | 1 | 5 | 1 | 0.2 | MSE + 0.5 CCC | 0.01804 | Smoke test only: one dyadic head per role, 1 epoch, 8 train windows. |

## Planned Comparisons

Candidate first real TCN runs:

```text
eGeMAPS raw, small TCN
eGeMAPS PCA, same TCN
eGeMAPS random projection, same TCN
W2V-BERT2 PCA, small/regularized TCN
Swin PCA, small/regularized TCN
OpenFace raw or PCA
OpenPose raw or PCA
```

Dyadic note:

```text
train_tcn.py is role-level and predicts y [time].
train_tcn_dyadic.py predicts y [time, 2] from dyadic tensors.
Use --head-type shared or --head-type role_specific to compare dyadic output heads.
```

Architecture ablations to consider:

```text
kernel_size: 3 vs 5 vs 9
hidden_channels: 32 vs 64 vs 128
levels: 3 vs 4 vs 5
dropout: 0.2 vs 0.3 vs 0.5
ccc_weight: 0.25 vs 0.5 vs 1.0
window_size: 250 vs 500 vs 750
```
