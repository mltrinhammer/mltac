# MPII Collaborator Handoff - 2026-06-18

## Context

Main repo: `/work/ACM/ACM-clean`

Runpod SSH, if needed:

```bash
ssh root@69.19.136.83 -p 13561 -i /work/.ssh/runpod_acm_ed25519
```

Private key location:

```text
/work/.ssh/runpod_acm_ed25519
```

Current plan from the primary thread: leave PinSoRo for tomorrow, and let the completed NOXI/NOXI-J results inform the next modeling decisions. You are expected to work mainly on MPII.

## Current MPII Baseline

Best MPII validation reference from today:

```text
Group-meanpool LOSO validation
Frame-weighted CCC: 0.6097
Unweighted mean CCC: 0.6212
Frame-weighted MAE: 0.0923
Frame-weighted RMSE: 0.1165
Frame-weighted Pearson: 0.6231
```

Per holdout CCC:

```text
008: 0.4975
009: 0.6038
010: 0.5136
026: 0.7597
027: 0.7009
028: 0.6516
```

Submission predictions already used the group-meanpool holdout028 alpha1 setup. The relevant submission zip from earlier today was:

```text
/work/ACM/submission_1806_3_rebuilt_rootfixed.zip
```

## MPII Code Touched Today

Relevant modified files:

```text
scripts/train_mpii_group_meanpool_multimodal.py
scripts/train_mpii_group_meanpool_multimodal_calibration.py
scripts/train_mpii_group_meanpool_multimodal_temporal.py
src/acm_pipeline/group_models.py
```

Important additions/changes:

- `prediction_head_sharing` support in group models.
- Role-specific loss override plumbing in calibration trainer.
- Temporal delta axis fix in temporal/calibration variants.
- Metadata/head-sharing support was kept in the shared group model path.

Before extending MPII, inspect these files and run syntax checks. Do not assume everything has been committed yet unless `git status` is clean.

## NOXI/NOXI-J Result To Learn From

Two joint NOXI+NOXI-J dyadic controls were run tonight:

1. `noxi_noxij_dyadic_roleheads_ccconly_seed13`
   - shared dyadic encoder
   - role-specific novice/expert heads
   - best overall val CCC: `0.9028220169477441`
2. `noxi_noxij_dyadic_sharedhead_ccconly_seed13`
   - shared dyadic encoder
   - one shared 2-channel output head
   - best overall val CCC: `0.9095221766682586`

Takeaway: the simpler shared 2-channel output head slightly beat role-specific heads in this joint dyadic setting. That suggests partner/dyadic context and joint training may matter more than separating the final heads.

Old preserved NOXI/NOXI-J best was also a joint NOXI+NOXI-J `dyadic_shared` model, not the attention variant. It had strong NOXI-J validation (`0.7174`) and produced the preserved best NOXI-J submission files.

## MPII Modeling Ideas Inspired By NOXI/NOXI-J

These are suggested next steps, not completed work:

1. Test a more explicitly dyadic/group-aware MPII design rather than only group-mean pooling.
   - Current MPII winner is group meanpool, but NOXI/NOXI-J suggests explicit partner/joint representation can be valuable.
   - For MPII, consider a shared encoder over participants followed by a shared output head, not necessarily participant-specific heads.

2. Compare shared-head vs role/person-specific heads carefully.
   - NOXI/NOXI-J result argues against assuming separated heads are automatically better.
   - For MPII, if you add role/person-specific heads, include a shared-head control.

3. Revisit temporal smoothing/loss only after architecture control.
   - MPII already benefited from EMA/smoothing on predictions.
   - The temporal-loss experiments in NOXI were not clearly successful.
   - Keep CCC-only or simple CCC+small MSE as the first architecture control.

4. Preserve LOSO discipline.
   - Any MPII architecture change should be evaluated with the same holdout sessions or at least a clearly comparable subset before preparing predictions.

5. Consider a session/grouped sampler if training becomes IO-heavy.
   - PinSoRo already has `SessionBatchSampler`.
   - NOXI was slowed by random full-session cache loading.
   - MPII may benefit from grouped batches if it uses full-session tensors.

## Concrete Next Commands To Inspect MPII State

```bash
cd /work/ACM/ACM-clean

git status --short

find outputs/experiments/mpii_group_meanpool_loso -maxdepth 3 -type f | sort | head -200

for f in outputs/experiments/mpii_group_meanpool_loso/*/metrics_overall.csv; do
  echo FILE:$f
  sed -n '1,8p' "$f"
done
```

Potential syntax checks:

```bash
python3 -m py_compile \
  scripts/train_mpii_group_meanpool_multimodal.py \
  scripts/train_mpii_group_meanpool_multimodal_temporal.py \
  scripts/train_mpii_group_meanpool_multimodal_calibration.py \
  src/acm_pipeline/group_models.py
```

## Submission/Artifact Caution

Do not overwrite:

```text
/work/ACM/best noxi-j
```

Do not commit large model checkpoints unless explicitly requested. The NOXI/NOXI-J completed experiment folders have local `model_best.pt` files for reconstruction, but they are experiment artifacts, not normal source files.

## Suggested MPII Experiment Queue Shape

A good first MPII queue tomorrow would be small and comparable:

1. Current group-meanpool baseline rerun or verified checkpoint.
2. Shared dyadic/group encoder with shared 2-channel/person output, CCC-only.
3. Same architecture with EMA/alpha sweep on validation predictions.
4. Only then try role/person-specific heads or temporal losses.

Keep the first queue validation-only until it beats the current baseline.
