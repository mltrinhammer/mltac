# Inference-Only Track

This track contains the selected artifacts needed to reproduce CR inference without retraining, assuming the organizer regenerates the same prepared tensors from the PinSoRo embedding streams.

Each run directory keeps `config.json`, `model_best.pt`, compact metrics/logs, selected HMM outputs, and `test_submission_format/` where available. It intentionally omits `model_last.pt`, prepared `.npz` tensors, and large `*_prediction_scores.csv.gz` files.


## Inference Command Pattern

After regenerating the prepared tensors, use `--eval-only` to load an included
`model_best.pt` and regenerate validation/test score exports without training.
For example, for a packaged run directory:

```bash
python MoE/pinsoro_noxi_settings/train_person_interaction_fusion_temporal.py \
  --eval-only \
  --manifest MoE/moe_data/outputs/windows_w2400_s1200/audio_w2vbert2_w2400_s1200_dyadic.csv \
             MoE/moe_data/outputs/windows_w2400_s1200/text_xlm_roberta_w2400_s1200_dyadic.csv \
             MoE/moe_data/outputs/windows_w2400_s1200/visual_videomae_w2400_s1200_dyadic.csv \
  --domain-scope CR \
  --output-root artifacts/inference_only/submitted_models \
  --run-name cr_task_shared_tcn_no_partner \
  --fusion-mode concat \
  --fusion-channels 64 \
  --person-hidden-channels 64 \
  --person-levels 5 \
  --person-kernel-size 11 \
  --dropout 0.2 \
  --modality-dropout 0.1 \
  --causal-tcn \
  --encoder-sharing shared \
  --head-architecture shared_tcn \
  --interaction-mode none \
  --metadata-mode age_gender_role \
  --metadata-embedding-dim 16 \
  --metadata-dropout 0.2 \
  --temporal-delta-weight 0.1 \
  --soft-label-mode none \
  --active-heads task \
  --batch-size 32 \
  --seed 13 \
  --device cuda
```

Then rerun HMM/Viterbi decoding with the same manifest files and active head.
The selected settings already used for the packaged outputs are stored in each
run's `best_hmm_setting.json`.

## Submitted Models

- `submitted_models/cr_task_shared_tcn_no_partner`: clean submitted CR-task candidate.
- `submitted_models/cr_social_dyadic_shared_tcn_pre_encoder`: submitted CR-social source.

These use the same selected architecture family as the paper-final CR models but should be kept as their own submitted-model artifacts.

## Feature Ablations

`feature_ablations/` contains task and social runs for all feature sets: `atv`, `at`, `av`, `tv`, `a`, `t`, and `v`.

## Partner / Encoder Ablations

`partner_encoder_ablations/` contains the clean TCN partner/encoder comparisons only. Head-adapter runs are deliberately excluded.

Included CR-task variants: shared no-partner, shared linear-partner, separate no-partner, separate linear-partner, and dyadic pre-encoder.

Included CR-social variants: shared no-partner, shared linear-partner, separate no-partner, separate linear-partner, and dyadic pre-encoder.

## Sensitivity

`sensitivity/` contains the CR soft-kappa sensitivity results guided by the CC-agent method note. The selected summary reports canonical, numbered-blank, and augmented target subsets, including confidence-weighted soft kappa.
