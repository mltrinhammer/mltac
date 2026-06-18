# ACM Clean Handoff

Date: 2026-06-14

This directory is a non-destructive handoff plan for turning the current
`/work/ACM` workspace into a clean collaborator/GitHub/RunPod package.

The current workspace should remain the source of truth until the clean package
has been verified. Do not delete or move original experiment folders as part of
the first cleanup pass.

## Intended Outputs

1. GitHub/shareable repository
   - Code, configs, documentation, small result tables, and run scripts.
   - No large checkpoints, tensors, raw videos/audio, cache folders, virtualenvs,
     or generated bulk outputs.

2. External artifact package
   - Checkpoints, preprocessed tensors, manifests, predictions, logs, gate
     weights, and submission exports needed to reproduce the chosen lines.
   - This package can be copied to RunPod Network Storage and shared separately
     with a collaborator.

3. RunPod working copy
   - The GitHub/shareable repository plus the external artifact package mounted
     or synced at paths compatible with the existing manifests.
   - Prefer preserving `/work/ACM` on RunPod or creating symlinks for absolute
     paths.

## Core Decisions

- Final/deployable PinSoRo line: MoE1 soft-confidence metadata-head experts,
  two-head combiner, and HMM/Viterbi smoothing.
- Standalone NOXI model: early gated multimodal dyadic-shared TCN described in
  `/work/ACM/mltac-github-clean/ACM/noxi settings.txt` and stored under
  `/work/ACM/mltac-main/ACM/MoE/noxi_joint_settings`.
- NOXI MoE-style metadata-head experiments are separate ablation/transfer
  history and should not be confused with the standalone NOXI gated-fusion
  model.
- Newest architecture direction: early-fusion multimodal PinSoRo model with
  expert-per-person and partner interaction modeled on logits. Preserve scripts,
  manifests, checkpoints, and resume commands.
- MPII final/generalizable-epoch work: preserve LOSO epoch-selection outputs,
  final multimodal model, transformed tensors, manifests, and scripts.

## Documents

- `docs/KEEP_MAP.md`: exact keep/externalize/exclude map.
- `docs/CLEAN_REPO_PLAN.md`: proposed cleaned repository structure.
- `docs/RUNPOD_ARTIFACT_PLAN.md`: artifact package for RunPod execution.
- `docs/OPEN_GAPS.md`: points that still need verification before deleting or
  publishing anything.
