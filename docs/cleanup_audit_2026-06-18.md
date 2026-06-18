# Cleanup Audit

Date: 2026-06-18

This audit is for cleaning `/work/ACM/ACM-clean` and the GitHub branch
`acm-clean-main` while preserving all results so far.

## Current Git State

- Local branch: `main`
- Upstream: `origin/acm-clean-main`
- Clean import commit: `55e85b9 Initial clean ACM import`
- Remote `origin/main` is unrelated older history and should not be used as the
  cleanup base.
- Local uncommitted work that must be preserved:
  - `src/acm_pipeline/group_models.py`
  - `run_logs/run_mpii_groupmeanpool_submission_prediction_cu128.sh`
  - `run_logs/run_noxi_submission_prediction_queue_cu128.sh`

## Main Finding

The right cleanup boundary is not "delete every file not directly imported by a
training script." Some important result files are not imported by code, but they
are the evidence for model choices and submissions. The safer boundary is:

1. GitHub keeps source code, docs, run recipes, manifests, and small result
   summaries.
2. External artifact storage keeps checkpoints, generated tensors, prediction
   dumps, submissions, and heavy output trees.
3. Old exploratory code stays only if it supports one of the preserved result
   stories or is needed to regenerate a small summary.

## Code To Keep In GitHub

### Core package

Keep all of:

```text
src/acm_pipeline/
requirements.txt
README.md
.gitignore
```

The active scripts import `src.acm_pipeline` modules throughout, so this package
is the dependency root.

### Active general/MPII scripts

Keep:

```text
scripts/build_group_windows_from_multimodal_turn_manifest.py
scripts/build_mpii_group_window_manifest.py
scripts/collect_results.py
scripts/infer_tcn_multimodal.py
scripts/mpii_singlemodality_loso.py
scripts/run_mpiii_test_multimodal_eval.sh
scripts/train_mpii_group_meanpool_multimodal.py
scripts/train_mpii_group_meanpool_multimodal_calibration.py
scripts/train_mpii_group_meanpool_multimodal_temporal.py
scripts/train_tcn_multimodal.py
scripts/train_tcn_turns.py
```

Also keep preprocessing/build scripts under `scripts/` until their manifests are
fully replaced by committed or external artifact manifests.

### Active NOXI / NOXI-J scripts

Keep:

```text
MoE/noxi_joint_settings/evaluate_group_ema_smoothing.py
MoE/noxi_joint_settings/run_group_ema_smoothing_queue.py
MoE/noxi_joint_settings/run_noxi_group_meanpool_regression_queue.py
```

The latest run scripts and submission scripts reference these paths directly.

### Active PinSoRo follow-up scripts

Keep:

```text
MoE/pinsoro_noxi_settings/apply_person_interaction_hmm.py
MoE/pinsoro_noxi_settings/apply_person_interaction_hmm_active_heads.py
MoE/pinsoro_noxi_settings/combine_horizon_experts.py
MoE/pinsoro_noxi_settings/export_gated_fusion_checkpoint.py
MoE/pinsoro_noxi_settings/run_followup_experiment_queue.py
MoE/pinsoro_noxi_settings/run_gated_fusion_4gpu.py
MoE/pinsoro_noxi_settings/run_person_interaction_4gpu.py
MoE/pinsoro_noxi_settings/train_gated_fusion.py
MoE/pinsoro_noxi_settings/train_person_interaction_fusion.py
MoE/pinsoro_noxi_settings/train_person_interaction_fusion_temporal.py
```

These are directly referenced in the current RunPod handoff and queue scripts.

### Run recipes

Move these out of `run_logs/` and keep them under a source-like path, for
example `scripts/runpod/` or `scripts/submission/`:

```text
run_logs/run_metadata_head_queue_cu128.sh
run_logs/run_metadata_head_resume_cu128.sh
run_logs/run_noxij_calibration_sweep_cu128.sh
run_logs/run_noxij_delta_sweep_cu128.sh
run_logs/run_pinsoro_head_specialists_temporal_cu128.sh
run_logs/run_pinsoro_temporal_delta010_cu128.sh
run_logs/run_temporal_loss_queue_cu128.sh
run_logs/run_mpii_groupmeanpool_submission_prediction_cu128.sh
run_logs/run_noxi_submission_prediction_queue_cu128.sh
```

`run_logs/*.out`, PID files, and generated logs should not be tracked.

## Results To Preserve

### Keep small summaries in GitHub

Small CSV/JSON/Markdown result summaries can stay in GitHub, but should be
organized under a deliberate path, for example:

```text
experiments/summaries/
docs/results/
```

Current tracked summaries under `experiments/summaries/MoE/` preserve many
historic comparisons. They should not be deleted until each has either:

- a short result table in `docs/results/`, or
- an explicit entry in an archive manifest.

### Preserve local checkpoints and heavy outputs externally

Current local checkpoint files include:

```text
MoE/noxi_joint_settings/experiments_group_meanpool/*/model_best.pt
MoE/noxi_joint_settings/experiments_group_meanpool_calibration_sweep/*/model_best.pt
outputs/experiments/mpii_group_meanpool_loso/*/model_best.pt
```

These are ignored by Git and must be copied to an external artifact root before
any local pruning.

Recommended artifact root:

```text
/work/ACM/artifacts/acm-clean-2026-06-18/
```

Recommended first artifact groups:

```text
noxi_joint_settings/
mpii_group_meanpool_loso/
submissions/
manifests/
result_summaries/
```

## Candidate Cleanup Actions

### Safe immediately

- Keep GitHub branch `acm-clean-main` as the working cleanup branch.
- Commit current dirty source/run-script changes before restructuring.
- Add stronger `.gitignore` entries for generated logs and submissions.
- Move runnable shell recipes from `run_logs/` into `scripts/runpod/` or
  `scripts/submission/`.
- Replace `run_logs/*.out` in Git with a short summary note or archive pointer.

### Safe after artifact copy

- Remove local generated `outputs/`, `processed/`, `cache/`, and `submissions/`
  directories from the Git worktree if they have been copied to the artifact
  root or are fully regenerable.
- Remove ignored checkpoints from the local source tree only after verifying
  artifact copies by checksum.

### Needs review before deleting

- Older top-level `MoE/*.py` ablation and diagnosis scripts.
- Large historical `experiments/summaries/MoE/` subtrees.
- Old Markdown handoff notes that duplicate newer docs.

These are not necessarily needed for future training, but they explain why the
current result lines were chosen.

## Proposed GitHub End State

```text
README.md
requirements.txt
docs/
  cleanup_audit_2026-06-18.md
  results/
  handoff/
src/acm_pipeline/
scripts/
  preprocessing/
  training/
  evaluation/
  submission/
  runpod/
MoE/
  noxi_joint_settings/
  pinsoro_noxi_settings/
experiments/
  summaries/
artifacts/
  README.md
```

## Recommended Next Pass

1. Commit the current dirty training-code/run-script changes.
2. Create an artifact manifest of all local checkpoints, submissions, manifests,
   and generated result trees.
3. Copy those artifacts to `/work/ACM/artifacts/acm-clean-2026-06-18/`.
4. Verify artifact copies with checksums.
5. Move run scripts out of `run_logs/` and update `.gitignore`.
6. Make a cleanup commit on `acm-clean-main`.
7. Only after that, consider pruning older MoE scripts and historical summary
   folders.

## Non-Goals For This Pass

- Do not force-push over `origin/main`.
- Do not delete raw data.
- Do not delete ignored checkpoints before artifact verification.
- Do not remove old result summaries until the current best-result story is
  documented in a smaller, explicit form.
