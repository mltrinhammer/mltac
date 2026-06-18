# Open Gaps Before Publishing Or Deleting Anything

Date: 2026-06-14

## Must Verify

1. Dirty git state
   - `/work/ACM/mltac-main` has many tracked modifications/deletions and
     important untracked content, including `ACM/MoE/`.
   - `/work/ACM/mltac-github-clean` has modified/untracked MPII/NOXI files.
   - Build the clean repo from the working tree, not from a fresh clone.

2. Dependency state
   - Existing `requirements.txt` files may not fully match the active virtualenv.
   - Preserve the frozen package list from `EXPERIMENTS_RUNBOOK.md`.
   - Before RunPod, test install in a fresh environment.

3. True NOXI artifact classification
   - The standalone NOXI gated-fusion model appears to be
     `ACM/MoE/noxi_joint_settings/.../noxi_noxij_audio_text_visual_w500_s125_gated_dyadic_shared_seed13`.
   - The separate `ACM/MoE/experiments/noxi_moe1_*` folders are MoE-style
     metadata-head transfer/ablation history.
   - Keep both categories if space allows, but label them clearly.

4. Person-interaction smoke test
   - The runbook currently verifies paths/checkpoints and gives full resume
     commands.
   - It still needs a true 1-5 minute training smoke command, or a script flag
     such as `--epochs 1` on a tiny subset, verified before spending GPU time.

5. Absolute paths
   - Many manifests likely contain `/work/ACM/...`.
   - RunPod should preserve `/work/ACM` or provide symlinks.

6. Data sufficiency
   - The 100 GB package supports immediate continuation/evaluation.
   - It may not support full regeneration of all features from raw data.
   - Raw data and full historical processed trees should remain on university
     storage unless explicitly needed.
   - The full `moe_data_soft_labels` tree is 192 GB. Only its small `outputs`
     folder belongs in the first 100 GB package unless retraining soft-label
     experts is required.

## Useful Additions

- A root `README.md` for the clean repo explaining the four model stories.
- A `.gitignore` that blocks heavy files and caches.
- A machine-generated artifact manifest with size, file count, and checksum for
  each external artifact folder.
- A `scripts/smoke_check_artifacts.py` that validates required paths.
- A `scripts/sync_to_runpod.sh` template generated from the artifact manifest.

## Cleanup Rule

No original folder in `/work/ACM` should be deleted until:

```text
clean repo imports work
artifact manifest exists
required checkpoints/data pass existence checks
RunPod smoke check passes
collaborator-facing README is readable
```
