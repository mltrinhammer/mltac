# Turn-Level TCN Architecture

## Shared input contract

The active unimodal trainer is `scripts/train_tcn_turns.py`. It consumes paired interval manifests produced by preprocessing and builds batches from `src/acm_pipeline/turn_data.py`. The same interval contract now serves both speech turns and legacy fixed windows. Winner-only multimodal runs use `scripts/train_tcn_multimodal.py`, which consumes joined multimodal turn manifests from the same module.

For one turn batch:

- `x` has shape `[batch, 2F, T]`
- the first `F` channels are novice features
- the next `F` channels are expert features
- `y` has shape `[batch, T, 2]` in role order `[novice, expert]`
- `target_mask` and `loss_mask` have the same target layout

Turns are variable-length. `turn_collate_fn` pads each batch to the local maximum turn length and uses masks so padding never affects losses or metrics.

For multimodal batches, `multimodal_turn_collate_fn` keeps each modality separate as its own `[batch, 2F_m, T]` tensor until the fusion module combines modalities within each role.

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

## Winner-only multimodal fusion

Implemented by `RoleWiseMultimodalFusion` and `MultimodalTurnTCNRegressor` in `src/acm_pipeline/models_tcn.py` and trained through `scripts/train_tcn_multimodal.py`.

- The winner backbone is resolved from completed unimodal turn runs by majority win count across registered feature sets.
- One representative audio, text, and visual feature set is selected from that winning backbone.
- Each modality is projected to a shared channel size independently for novice and expert.
- Fusion happens within role first, not across roles.
- The fused novice stream and fused expert stream are then passed into the winning backbone.

Two fusion modes are supported:

- `gated`: frame-level softmax gating across modalities after projection. This is the main multimodal method.
- `concat`: concatenation of projected modality channels. This is the lightweight internal baseline.

For gated runs, the trainer writes `val_gate_weights.csv` so the mean validation contribution of each modality remains inspectable per role.

## Legacy fixed-window comparison

The legacy unit of analysis is restored through `compute_window_segments()` in `src/acm_pipeline/turns.py` and `scripts/noxi_build_window_manifest.py`.

- Default window size: `500` frames.
- Default stride: `125` frames.
- Session tails are covered by an end-anchored final window when needed.

The same `scripts/train_tcn_turns.py` trainer handles the unimodal window manifests without model-side changes, and `scripts/train_tcn_multimodal.py` handles the winner-only multimodal window manifests. This keeps the comparison controlled: the backbone, loss, early stopping, and validation logic stay fixed while only the interval definition changes.

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
- `val_submission_format/`

Multimodal gated runs also add:

- `val_gate_weights.csv`

`reconstruct_validation()` averages interval-level predictions back onto session frames. For turns this averages non-overlapping segments; for legacy windows it averages overlapping windows by frame coverage count so the exported metrics stay aligned with the original role-level timelines.

In addition to the internal long-form `val_predictions.csv`, the trainers now export an organizer-style session tree under `val_submission_format/`. For NOXI test splits this maps directly onto the challenge submission layout, using `noxi-base`, `noxi-additional`, and `noxi-j` directory names where the split metadata is available.

## Psychological and technical validity of role combination

- Novice and expert are always aligned on the same observed turn interval.
- Both targets are predicted on every turn, including the non-speaker's engagement.
- The `simple` baseline keeps both people fully separate and therefore does not assume direct interaction inside the model.
- The interaction models only combine role information after both people have been aligned to the same interval and represented in a consistent role order.

This makes the three-model ladder an ordered comparison from no cross-person interaction, to shared dyadic encoding, to explicit contextual interaction.

The multimodal extension preserves the same psychological ordering: modalities are first combined within each person, and only then is cross-person interaction introduced by the winning dyadic backbone. The legacy-window comparison preserves the same modelling assumptions while swapping only the unit of analysis for both the unimodal winner backbone and the winner-only multimodal combinations.