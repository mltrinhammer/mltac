# ACM CC Track Reproducibility Bundle

This repository is a CC-track-only reproducibility package for the final submitted PinSoRo CC models and the CC ablations reported in the paper.

It contains:

- training/evaluation code copied from the RunPod project used for the final CC experiments;
- final submitted CC task and CC social checkpoints;
- matched partner-ablation run directories;
- submitted-checkpoint feature/modality ablation outputs;
- the final copied `pinsoro-cc` prediction folder;
- a CR-aligned two-track artifact layout under `artifacts/`;
- wrapper scripts for cached verification, checkpoint inference, HMM post-processing, and final-model training.

## Quick Verification Without Retraining

Run this first. It only reads bundled CSVs and should work without the processed feature tensors:

```bash
cd /work/acm_cc_repro
bash scripts/verify_cached_outputs.sh
```

Expected partner ablation table:

| Head | No partner | Late linear partner | Late gated partner |
|---|---:|---:|---:|
| CC task | 0.330583 | 0.376920 | 0.373723 |
| CC social | 0.354842 | 0.346729 | 0.347857 |

Expected submitted-checkpoint modality mask table:

| Modalities | CC task | CC social |
|---|---:|---:|
| A+T+V | 0.376920 | 0.346729 |
| A+T | 0.140607 | 0.195056 |
| A+V | 0.373855 | 0.351916 |
| T+V | 0.382419 | 0.314679 |
| A | 0.089435 | 0.164105 |
| T | 0.160351 | 0.100059 |
| V | 0.380484 | 0.290584 |

`A = audio_w2vbert2`, `T = text_xlm_roberta`, `V = visual_videomae`.

## Two-Track Reproducibility Layout

The artifact tree mirrors the CR bundle:

- `artifacts/training_pipeline/`: end-to-end retraining notes and wrappers.
- `artifacts/inference_only/`: symlinked view of submitted models, feature ablations, partner ablations, and a sensitivity placeholder.

See `artifacts/inference_only/ARTIFACT_MANIFEST.csv` for the run-level mapping.

The push-ready bundle keeps compact metrics, configs, `model_best.pt`, logs, and submission-format outputs. It intentionally omits `model_last.pt`, processed tensors, large score dumps, and redundant per-frame prediction CSVs.

## Data Layout For Full Runs

The bundled manifests expect processed PinSoRo tensors at paths like:

```text
MoE/moe_data/processed/domain_norm/<feature>/<split>/<session>/<role>.<feature>.raw.npz
```

relative to the repository root. The manifests themselves are bundled under:

```text
artifacts/manifests/windows_w2400_s1200/
```

To run checkpoint inference or retraining, place or symlink the processed `MoE/moe_data/processed` tree into this repo. The raw challenge data and feature extraction steps are intentionally not vendored here.

## Recompute Submission Inference / HMM From Checkpoints

Requires the processed tensor tree described above. Use `--eval-only` through the final wrappers to regenerate validation/test score exports from bundled `model_best.pt` checkpoints without retraining:

```bash
cd /work/acm_cc_repro
OUT=artifacts/runs DEVICE=cuda bash scripts/train_cc_task_final.sh --eval-only
OUT=artifacts/runs DEVICE=cuda bash scripts/train_cc_social_final.sh --eval-only
```

Then rerun CC-task HMM/Viterbi after score regeneration:

```bash
bash scripts/reproduce_cc_task_hmm_from_logits.sh
```

CC social is documented as no-HMM in the June 30 note.

## Recompute Feature Ablations From Checkpoints

Requires the processed tensor tree described above.

```bash
cd /work/acm_cc_repro
DEVICE=cuda bash scripts/reproduce_modality_masks_from_checkpoints.sh
```

This uses the exact submitted CC task and CC social checkpoints and masks excluded modalities to zero at inference time. It does not retrain models, does not apply logit bias, and does not apply HMM smoothing.

## Re-run Final Model Training

Requires processed tensors and a GPU for practical runtime.

```bash
cd /work/acm_cc_repro
DEVICE=cuda bash scripts/train_cc_task_final.sh
DEVICE=cuda bash scripts/train_cc_social_final.sh
```

## Important Interpretation

The feature/modality ablations are inference-time masks over the submitted late-linear checkpoints. The partner ablations are separate matched-architecture training runs. These two tables answer different questions and should not be mixed as if they used the same ablation mechanism.

More details are in `docs/REPRODUCIBILITY.md`, `docs/ABLATIONS.md`, and the original RunPod note `docs/cc_ablation_reproducibility_2026-06-30.md`.
