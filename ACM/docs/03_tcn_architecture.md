# Turn-Level TCN Architecture

## Shared input contract

The active trainer is `scripts/train_tcn_turns.py`. It consumes paired turn manifests produced by preprocessing and builds batches from `src/acm_pipeline/turn_data.py`.

For one turn batch:

- `x` has shape `[batch, 2F, T]`
- the first `F` channels are novice features
- the next `F` channels are expert features
- `y` has shape `[batch, T, 2]` in role order `[novice, expert]`
- `target_mask` and `loss_mask` have the same target layout

Turns are variable-length. `turn_collate_fn` pads each batch to the local maximum turn length and uses masks so padding never affects losses or metrics.

## Model ladder

### 1. `simple`

Implemented by `IndependentDyadicTCNRegressor`.

- Splits the input into novice and expert halves.
- Applies the same person-level TCN weights to each role independently.
- Predicts both roles for every turn interval.
- Provides the fairest no-interaction baseline because the non-speaker is retained as a separate person stream, not dropped.

This model never mixes novice and expert hidden states.

### 2. `dyadic_shared`

Implemented by `DyadicTCNRegressor`.

- Receives the concatenated novice and expert channels as one dyadic input tensor.
- Uses one temporal encoder over the paired signal.
- Uses one shared two-channel prediction head to output novice and expert engagement jointly.

This is the lightest interaction model in the active comparison ladder.

### 3. `attention`

Implemented by `RoleAttentionTCNRegressor`.

- Uses separate novice and expert temporal encoders.
- Applies role-specific multi-head attention over self, partner, or joint history.
- Concatenates each role's local hidden state with its attended context before prediction.

The current launcher uses:

- `attention_context=joint`
- `attention_past_frames=1500`
- `exclude_current_frame=true`

At 25 Hz, `1500` frames correspond to roughly 60 seconds of past context.

## Training objective

Each model is trained with the same masked objective:

`loss = masked_mse_loss + ccc_weight * ccc_loss`

Other shared training behavior:

- gradient clipping at `max_norm=5.0`
- reproducible seeding for Python, NumPy, and PyTorch
- validation reconstruction back to full session timelines
- early stopping on validation overall CCC with `min_epochs`, `min_delta`, and `patience`
- reload of the best checkpoint before final metrics and prediction export

The validation reports use raw CCC, where higher is better.

## Validation outputs

Each run directory contains:

- `metrics_overall.csv`
- `metrics_by_role.csv`
- `metrics_by_dataset.csv`
- `metrics_by_session.csv`
- `val_predictions.csv`

`reconstruct_validation()` averages predictions from non-overlapping turn segments back onto session frames so the exported metrics stay aligned with the original role-level timelines.

## Psychological and technical validity of role combination

- Novice and expert are always aligned on the same observed turn interval.
- Both targets are predicted on every turn, including the non-speaker's engagement.
- The `simple` baseline keeps both people fully separate and therefore does not assume direct interaction inside the model.
- The interaction models only combine role information after both people have been aligned to the same interval and represented in a consistent role order.

This makes the three-model ladder an ordered comparison from no cross-person interaction, to shared dyadic encoding, to explicit contextual interaction.