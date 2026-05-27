# Transformer Architecture Notes

This document explains the adjustable parts of the encoder-only Transformer baseline and keeps a running log of configurations tried. Update the experiment log whenever a Transformer run is launched.

## Model Type

The baseline uses an **encoder-only Transformer**.

Reason:

```text
input:  observed feature sequence x [time, features]
output: aligned engagement prediction y_hat [time]
```

No decoder is needed because the task is not token generation, translation, or autoregressive next-step prediction. The model reads the whole feature window and predicts one scalar engagement value per frame.

## Adjustable Architecture Components

### Input Features

Controlled by the manifest passed to:

```powershell
python scripts\train_transformer.py --manifest <manifest.csv>
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

The script reads:

```text
input_dim = n_features
```

Then projects:

```text
input_dim -> d_model
```

Effect:

```text
larger input_dim increases the projection-layer parameter count
PCA/RP can reduce high-dimensional streams before Transformer training
d_model controls the internal Transformer representation size
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
larger windows provide more context
self-attention cost grows roughly with window_size^2
larger windows require more GPU memory
smaller windows are cheaper but may miss slow engagement changes
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

Effect:

```text
smaller stride creates more overlapping windows
more overlap gives denser training coverage and smoother validation reconstruction
larger stride is faster but gives fewer training examples
```

Current default:

```text
125 frames = 5 seconds
```

### Model Dimension

CLI argument:

```powershell
--d-model 128
```

This is the internal feature size used by the Transformer.

Effect:

```text
larger d_model increases capacity
larger d_model increases memory and compute
d_model must be divisible by n_heads
smaller d_model is safer for small datasets
```

Typical values:

```text
64, 128, 256
```

Current default:

```text
128
```

### Attention Heads

CLI argument:

```powershell
--n-heads 4
```

Multi-head attention splits `d_model` into multiple attention heads.

Effect:

```text
more heads allow different attention patterns in parallel
too many heads make each head smaller if d_model is fixed
d_model must divide evenly by n_heads
```

Examples:

```text
d_model=64,  n_heads=4 -> 16 dims per head
d_model=128, n_heads=4 -> 32 dims per head
d_model=128, n_heads=8 -> 16 dims per head
```

Current default:

```text
4
```

### Number of Encoder Layers

CLI argument:

```powershell
--n-layers 2
```

Effect:

```text
more layers increase temporal modelling capacity
more layers increase memory, compute, and overfitting risk
fewer layers are safer as a first baseline
```

Typical values:

```text
1, 2, 4
```

Current default:

```text
2
```

### Feedforward Dimension

CLI argument:

```powershell
--dim-feedforward 256
```

Each Transformer encoder layer has:

```text
self-attention block
feedforward block
```

The feedforward dimension controls the hidden size inside that per-frame MLP.

Effect:

```text
larger feedforward dimension increases capacity
larger feedforward dimension increases compute and overfitting risk
common setting is 2x to 4x d_model
```

Current default:

```text
256
```

for:

```text
d_model = 128
```

### Positional Encoding

Current implementation:

```text
fixed sinusoidal positional encoding
```

Reason:

```text
self-attention alone does not know frame order
positional encoding tells the model where each frame sits within the window
fixed sinusoidal encoding avoids adding trainable position parameters
```

Possible future option:

```text
learned positional embeddings
relative positional bias
```

### Padding / Attention Mask

The data loader pads short or tail windows to fixed length.

The Transformer receives:

```text
frame_mask [batch, time]
```

It uses this mask to prevent attention to padded frames.

Effect:

```text
real frames can attend to other real frames
padded frames are ignored by attention and loss
```

### Dropout

CLI argument:

```powershell
--dropout 0.2
```

Effect:

```text
higher dropout regularizes attention and feedforward blocks
higher dropout may help high-dimensional streams
too much dropout can underfit
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
higher ccc_weight prioritizes concordance/correlation
lower ccc_weight prioritizes absolute frame-level error
CCC is the primary validation metric
```

Current default:

```text
0.5
```

### Learning Rate and Weight Decay

CLI arguments:

```powershell
--lr 0.0005
--weight-decay 0.0001
```

Effect:

```text
Transformers can be more sensitive to learning rate than TCNs
lower learning rate is safer for first runs
weight decay helps regularize the projection, attention, and MLP weights
```

Current defaults:

```text
lr = 0.0005
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
patience stops training when validation CCC no longer improves
early stopping is important because Transformers can overfit quickly
```

## Current Default Command

```powershell
python scripts\train_transformer.py `
  --manifest outputs\manifests\model_processed_manifest_audio_egemaps_raw.csv `
  --window-size 500 `
  --stride 125 `
  --d-model 128 `
  --n-heads 4 `
  --n-layers 2 `
  --dim-feedforward 256 `
  --dropout 0.2 `
  --ccc-weight 0.5
```

## Small First-Run Suggestions

For UCloud GPU smoke tests:

```text
window_size = 250 or 500
d_model = 64
n_heads = 4
n_layers = 1 or 2
dim_feedforward = 128 or 256
dropout = 0.2 or 0.3
```

For larger runs:

```text
window_size = 500
d_model = 128
n_heads = 4
n_layers = 2
dim_feedforward = 256
dropout = 0.2-0.4
```

## Experiment Log

Add one row per run.

| Date | Run Name | Manifest | Window | Stride | d_model | Heads | Layers | FF Dim | Dropout | Loss | Val CCC | Notes |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---|
| 2026-05-27 | `smoke_transformer_raw` | `model_processed_manifest_audio_egemaps_raw.csv` | 125 | 125 | 32 | 4 | 1 | 64 | 0.1 | MSE + 0.5 CCC | -0.03236 | Local syntax/data smoke only: 1 epoch, 8 train windows. Not a real result. |

## Planned Comparisons

Candidate first Transformer runs:

```text
eGeMAPS raw, small Transformer
eGeMAPS PCA, same Transformer
eGeMAPS random projection, same Transformer
W2V-BERT2 PCA, small regularized Transformer
Swin PCA, small regularized Transformer
OpenFace raw/PCA
OpenPose raw/PCA
```

Dyadic note:

```text
The current Transformer trainer is role-level and predicts y [time].
Dyadic tensors use y [time, 2], so they need a dyadic Transformer trainer/head before training.
```

Architecture ablations to consider:

```text
window_size: 250 vs 500
d_model: 64 vs 128
n_layers: 1 vs 2 vs 4
n_heads: 4 vs 8
dropout: 0.2 vs 0.3 vs 0.5
ccc_weight: 0.25 vs 0.5 vs 1.0
```
