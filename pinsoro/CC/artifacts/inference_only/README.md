# Inference-Only Track

This track mirrors the CR reproducibility layout while keeping the original CC run directories available through symlinks. It assumes the organizer regenerates or provides the same processed PinSoRo tensors from the embedding streams.

Each submitted/partner run keeps `config.json`, `model_best.pt`, compact metrics/logs, and `test_submission_format/` where available. The push-ready bundle intentionally omits `model_last.pt`, prepared `.npz` tensors, large `*_prediction_scores.csv.gz` files, and redundant per-frame prediction CSVs.

## Submitted Models

- `submitted_models/cc_task_shared_linear_shared_tcn`: submitted CC-task source checkpoint.
- `submitted_models/cc_social_head_adapters_late_linear`: submitted CC-social source checkpoint.

CC task submission post-processing uses the documented HMM settings. CC social is documented as no-HMM in the June 30 note.

## Feature Ablations

`feature_ablations/` contains submitted-checkpoint modality-mask outputs for CC task and CC social. These are inference-time masks over the submitted checkpoints, not retrained feature-family models. The summary table is still available at:

```text
artifacts/ablation_outputs/pinsoro_cc_submitted_checkpoint_modality_masks_3006/submitted_checkpoint_modality_mask_summary.csv
```

To recompute these ablations after placing the processed tensors:

```bash
DEVICE=cuda bash scripts/reproduce_modality_masks_from_checkpoints.sh
```

## Partner / Encoder Ablations

`partner_encoder_ablations/` contains the matched CC partner comparisons. For CC these are partner-only comparisons: no partner, late-linear partner, and late-gated partner for task and social.

## Submission Inference

After regenerating the prepared tensors, use `--eval-only` with the same arguments as the final training wrappers to load an included `model_best.pt` and regenerate validation/test score exports without retraining. For example, for CC task:

```bash
OUT=artifacts/runs DEVICE=cuda bash scripts/train_cc_task_final.sh --eval-only
```

If using the wrapper directly is inconvenient, run `scripts/upstream/train_person_interaction_fusion_temporal.py --eval-only` with the same arguments shown in `scripts/train_cc_task_final.sh` or `scripts/train_cc_social_final.sh`, setting `--output-root artifacts/runs` and the packaged run name.

Then rerun HMM/Viterbi for CC task after score regeneration:

```bash
bash scripts/reproduce_cc_task_hmm_from_logits.sh
```

## Sensitivity

`sensitivity/` is currently a placeholder. The June 30 note says the CC task sensitivity analysis used the correct submitted task base model and that CC social was removed from HMM sensitivity because the submitted social interpretation is no-HMM. The actual sensitivity outputs still need to be added.
