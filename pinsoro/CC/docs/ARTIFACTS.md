# Artifact Index

## Final submitted checkpoints

- CC task: `artifacts/runs/pinsoro_cc_task_shared_linear_shared_tcn_delta010_metadata_seed13/model_best.pt`
- CC social: `artifacts/runs/pinsoro_cc_headarch_head_adapters_delta010_metadata_seed13/model_best.pt`

## Named configs

- `configs/cc_task_submitted_late_linear.json`
- `configs/cc_social_submitted_late_linear.json`
- `configs/cc_task_no_partner.json`
- `configs/cc_task_late_gated.json`
- `configs/cc_social_no_partner.json`
- `configs/cc_social_late_gated.json`

## Aligned inference-only view

- `artifacts/inference_only/submitted_models/`
- `artifacts/inference_only/feature_ablations/`
- `artifacts/inference_only/partner_encoder_ablations/`
- `artifacts/inference_only/sensitivity/` currently contains a placeholder for the pending CC sensitivity package.

## Final CC predictions

- `submissions/pinsoro-cc/`

## Original note from RunPod

- `docs/cc_ablation_reproducibility_2026-06-30.md`

This note was found on the RunPod network volume at:

```text
/workspace/ACM/ACM-clean/cc_ablation_reproducibility_2026-06-30.md
```

## Source code

- `src/acm_pipeline/`: shared pipeline package needed by the training/evaluation scripts.
- `scripts/upstream/train_person_interaction_fusion_temporal.py`: final/ablation trainer with `--eval-only` checkpoint inference support.
- `scripts/upstream/evaluate_submitted_modality_masks.py`: submitted-checkpoint modality-mask evaluator.
- `scripts/upstream/apply_person_interaction_hmm_active_heads.py`: HMM/Viterbi post-processing.
- `scripts/upstream/train_gated_fusion.py` and `scripts/upstream/train_pinsoro_tcn.py`: sibling dependencies imported by the trainer/evaluator.
