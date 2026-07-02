# Reproducibility Protocol

## Scope

This bundle covers only the PinSoRo CC track. It is intended to be combined later with separate CR and NoXi reproducibility bundles.

## Final Submitted CC Models

### CC task

Run directory:

```text
artifacts/runs/pinsoro_cc_task_shared_linear_shared_tcn_delta010_metadata_seed13
```

Configuration summary:

- domain: CC
- active head: task
- features: audio_w2vbert2, text_xlm_roberta, visual_videomae
- windows: w2400_s1200 dyadic manifests
- fusion: concat
- encoder sharing: shared
- head architecture: shared_tcn
- partner interaction: late linear residual, scale 0.1
- metadata: age, gender, role
- temporal delta loss weight: 0.1
- CC task weighting: targeted, class1 weight 2.0
- seed: 13

Raw validation kappa: `0.37692018200554195`.

Submitted post-processing recorded in the June 30 note:

```text
logit bias: [0.5, -0.25, -0.5, -0.25]
HMM: mix=1.0, strength=12.0, alpha=1.0
```

### CC social

Run directory:

```text
artifacts/runs/pinsoro_cc_headarch_head_adapters_delta010_metadata_seed13
```

Configuration summary:

- domain: CC
- active heads: task, social
- features: audio_w2vbert2, text_xlm_roberta, visual_videomae
- windows: w2400_s1200 dyadic manifests
- fusion: concat
- encoder sharing: shared
- head architecture: head_adapters, one adapter level
- partner interaction: late linear residual, scale 0.1
- metadata: age, gender, role
- temporal delta loss weight: 0.1
- seed: 13

Raw validation kappa: `0.3467286999282731`.

Submitted social logit bias recorded in the June 30 note:

```text
[0.0, 0.0, 1.0, -1.5, 0.75]
```

The June 30 note states that the submitted CC social interpretation used no HMM.

## Bundled Artifacts

- `artifacts/runs/`: final and partner-ablation run directories with configs, best checkpoints, logs, compact metrics, and submission-format outputs. Large cached score/prediction dumps are not bundled.
- `artifacts/manifests/windows_w2400_s1200/`: dyadic and individual manifests used by the CC runs.
- `artifacts/ablation_outputs/pinsoro_cc_submitted_checkpoint_modality_masks_3006/`: cached submitted-checkpoint modality-mask metrics and summary.
- `artifacts/inference_only/`: CR-aligned symlinked view of submitted models, feature ablations, partner ablations, and a sensitivity placeholder.
- `submissions/pinsoro-cc/`: copied final CC prediction folder.
- `configs/`: named copies of the relevant run configs.

## Reproduction Levels

1. Cached-output verification: `bash scripts/verify_cached_outputs.sh`.
2. Checkpoint inference for feature ablations: `bash scripts/reproduce_modality_masks_from_checkpoints.sh` after placing processed tensors.
3. Checkpoint inference: run the final wrappers with `OUT=artifacts/runs ... --eval-only` after placing processed tensors. This regenerates the score exports omitted from the push-ready bundle.
4. HMM from regenerated CC task logits: `bash scripts/reproduce_cc_task_hmm_from_logits.sh` after eval-only inference, because transition matrices are estimated from train labels in the manifests.
5. Full retraining: `bash scripts/train_cc_task_final.sh` and `bash scripts/train_cc_social_final.sh` after placing processed tensors.

## Environment

Minimal Python dependencies are listed in `requirements.txt`:

```text
numpy
torch
```

The original runs used `/usr/bin/python3` on RunPod with CUDA. CPU can verify cached outputs and may run small inference checks, but full training should be done on GPU.
