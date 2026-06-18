# ACM Clean Handoff - 2026-06-18 Evening

## Runpod connection

```bash
ssh root@69.19.136.83 -p 13561 -i /work/.ssh/runpod_acm_ed25519
```

Private key persists locally at:

```text
/work/.ssh/runpod_acm_ed25519
```

Primary repo:

```text
/work/ACM/ACM-clean
/workspace/ACM/ACM-clean  # Runpod
```

## Current high-level instruction

Do not interrupt the active NOXI/NOXI-J queue unless it crashes or clearly hangs. Let both experiments finish, then pull Runpod artifacts back to local, commit/push to GitHub, and update this handoff with final metrics.

User specifically said to skip immediate detailed analysis after the first NOXI/NOXI-J run. Let experiments run first.

## Active Runpod queue

Script:

```text
/workspace/ACM/ACM-clean/run_logs/run_noxi_noxij_dyadic_roleheads_joint_cu128.sh
```

Output root:

```text
/workspace/ACM/ACM-clean/MoE/noxi_joint_settings/experiments_noxi_noxij_dyadic_roleheads_joint
```

Two experiments in order:

1. `noxi_noxij_dyadic_roleheads_ccconly_seed13`
   - joint NOXI + NOXI-J training
   - dyadic input
   - shared dyadic encoder
   - role-specific novice/expert heads
   - CCC-only
   - `--test-splits` intentionally empty: no test predictions during these runs
2. `noxi_noxij_dyadic_sharedhead_ccconly_seed13`
   - same data/training setup
   - old-style shared 2-channel head (`dyadic_shared`)
   - CCC-only
   - control for old preserved best architecture family

### Important queue modifications

The active queue was changed from batch size 32 to batch size 8 because first-batch cache warmup was extremely heavy with full-session tensors. It now also uses `--progress-every 1`.

Temporary diagnostics were added:

- `scripts/train_tcn_multimodal.py`: startup and batch progress prints
- `src/acm_pipeline/data.py`: tensor load timing prints

These diagnostics are useful for monitoring but should probably be removed or gated before committing final polished code, unless the user wants them kept behind a flag.

## Status at last check

First experiment completed:

```text
noxi_noxij_dyadic_roleheads_ccconly_seed13
```

Best epoch: `6`
Best overall val CCC: `0.9028220169477441`
Early stopped at epoch 12.

Training log:

```text
/workspace/ACM/ACM-clean/MoE/noxi_joint_settings/experiments_noxi_noxij_dyadic_roleheads_joint/noxi_noxij_dyadic_roleheads_ccconly_seed13/training_log.csv
```

Epoch summary:

```text
1  0.89156
2  0.86654
3  0.88881
4  0.89744
5  0.88280
6  0.90282  best
7  0.86248
8  0.88918
9  0.88668
10 0.89125
11 0.89358
12 0.89500 early stop
```

Latest metrics on disk after first run:

```text
metrics_by_dataset.csv:
noxi  CCC 0.7743103621647743
noxij CCC 0.6886446347848072

metrics_by_role.csv:
novice CCC 0.9169937585052
expert CCC 0.8159086232139576
```

Note: dataset x role cross-tab was not available during live epoch metrics. It can be computed from `val_predictions.csv` after the best-checkpoint export exists. The first run should now have final export files because it completed.

Second experiment is running:

```text
noxi_noxij_dyadic_sharedhead_ccconly_seed13
```

At last check it had started epoch 1.

Check status with:

```bash
ssh root@69.19.136.83 -p 13561 -i /work/.ssh/runpod_acm_ed25519 \
'ps -eo pid,ppid,stat,etime,%cpu,%mem,rss,cmd | grep -E "run_noxi_noxij_dyadic|train_tcn_multimodal" | grep -v grep || true; \
for f in /workspace/ACM/ACM-clean/MoE/noxi_joint_settings/experiments_noxi_noxij_dyadic_roleheads_joint/logs/*.log; do echo FILE:$f; grep -E "epoch=|epoch_start|validation_start|early_stop|Run directory" "$f" | tail -80; done'
```

## Prepared PinSoRo queues

Do not start until NOXI/NOXI-J queue finishes unless user explicitly changes priority.

### 1. Joint CC+CR, task/social still separated

Script:

```text
/workspace/ACM/ACM-clean/run_logs/run_pinsoro_joint_domains_head_specialists_temporal_cu128.sh
/work/ACM/ACM-clean/run_logs/run_pinsoro_joint_domains_head_specialists_temporal_cu128.sh
```

Runs two models:

- `domain-scope both`, `active-heads task`
- `domain-scope both`, `active-heads social`

Purpose: test whether task specialist and social specialist benefit from joint CC+CR training while preserving task/social separation.

HMM evaluator patch:

```text
MoE/pinsoro_noxi_settings/apply_person_interaction_hmm_active_heads.py
```

It now filters validation/test score rows by `--domain`; this matters for `domain-scope both` runs evaluated separately as CC and CR.

### 2. CR-social imbalance handling

Script:

```text
/workspace/ACM/ACM-clean/run_logs/run_pinsoro_cr_social_weight_oversample_cu128.sh
/work/ACM/ACM-clean/run_logs/run_pinsoro_cr_social_weight_oversample_cu128.sh
```

Four planned experiments:

1. `pinsoro_cr_social_delta010_metadata_cap5_seed13`
2. `pinsoro_cr_social_delta010_metadata_sqrt_inverse_seed13`
3. `pinsoro_cr_social_delta010_metadata_class3oversample3_unweighted_seed13`
4. `pinsoro_cr_social_delta010_metadata_class3oversample3_cap5_seed13`

Run this only after deciding based on the joint CC+CR specialist results.

## PinSoRo performance/loading note

PinSoRo already uses `SessionBatchSampler`, so it groups shuffled training batches by session and should avoid the worst NOXI cache-loading behavior. It also uses bounded tensor cache. No code change is needed before launching PinSoRo.

There is support for `--mmap-cache-root`, but no usable mmap cache was found on the Runpod at:

```text
/workspace/ACM/mltac-main/ACM/MoE/moe_data/processed/domain_norm_mmap
/workspace/ACM/ACM-clean/MoE/moe_data/processed/domain_norm_mmap
/workspace/ACM/ACM-clean/processed/pinsoro_mmap
```

Do not build mmap cache while NOXI/NOXI-J queue is active because it would consume CPU/disk IO.

## Best preserved NOXI-J submission asset

Preserved folder:

```text
/work/ACM/best noxi-j
```

Important README there documents provenance. Key point: preserved filled predictions came from old `submission_2_evalai_rootfixed_filled.zip`, fingerprint:

```text
e7b534020aecdfa42d97d956c588b12dac769860e986dc680705c3692f26aef4
```

Source model/export inferred:

```text
/work/ACM/mltac-main/ACM/MoE/noxi_joint_settings/experiments/noxi_noxij_audio_text_visual_w500_s125_gated_dyadic_shared_seed13/test_submission_format/noxi-j
```

Old preserved source model validation reference:

```text
overall CCC 0.913456654016731
NOXI val CCC 0.8454521049920518
NOXI-J val CCC 0.7174113997416623
novice CCC 0.9259290197913771
expert CCC 0.8709342587398391
```

## Submission state

Current rebuilt upload zip from earlier today:

```text
/work/ACM/submission_1806_3_rebuilt_rootfixed.zip
```

It includes:

- NOXI shared EMA base/additional
- MPII groupmeanpool holdout028 alpha1
- PinSoRo specialist-HMM
- root-fixed layout

Do not overwrite preserved best NOXI-J files.

## Tomorrow workflow

1. Check Runpod queue status.
2. If both NOXI/NOXI-J experiments finished, rsync/pull the experiment root and run logs back to local.
3. Compute/record final comparison:
   - role-head joint model vs shared-head control
   - overall CCC
   - NOXI CCC
   - NOXI-J CCC
   - novice/expert CCC
   - dataset x role CCC from final `val_predictions.csv`
4. Remove/gate temporary diagnostic logging if committing code.
5. Commit and push code/scripts/docs to GitHub.
6. Start PinSoRo joint-domain specialist queue if user agrees.
7. After PinSoRo joint-domain results, decide on CR-social imbalance queue.

## Useful pull command template

After NOXI queue finishes:

```bash
rsync -avz -e "ssh -p 13561 -i /work/.ssh/runpod_acm_ed25519" \
  root@69.19.136.83:/workspace/ACM/ACM-clean/MoE/noxi_joint_settings/experiments_noxi_noxij_dyadic_roleheads_joint/ \
  /work/ACM/ACM-clean/MoE/noxi_joint_settings/experiments_noxi_noxij_dyadic_roleheads_joint/

rsync -avz -e "ssh -p 13561 -i /work/.ssh/runpod_acm_ed25519" \
  root@69.19.136.83:/workspace/ACM/ACM-clean/run_logs/ \
  /work/ACM/ACM-clean/run_logs/
```

Be careful with run logs: local `run_logs/` contains several new queue scripts and may also contain unrelated local files. Use `git status` before committing.

## Local git status at handoff creation

There are modified code files and several untracked run scripts. Check with:

```bash
git -C /work/ACM/ACM-clean status --short
```

Known modified files include:

```text
MoE/pinsoro_noxi_settings/apply_person_interaction_hmm_active_heads.py
MoE/pinsoro_noxi_settings/train_person_interaction_fusion_temporal.py
scripts/train_mpii_group_meanpool_multimodal.py
scripts/train_mpii_group_meanpool_multimodal_calibration.py
scripts/train_mpii_group_meanpool_multimodal_temporal.py
scripts/train_tcn_multimodal.py
src/acm_pipeline/data.py
src/acm_pipeline/group_models.py
src/acm_pipeline/models_tcn.py
```

Do not blindly revert: many changes are intentional from today. But consider cleaning/gating temporary NOXI diagnostics before the final commit.

## Final NOXI/NOXI-J Queue Results - Updated Night 2026-06-18

Both experiments completed and artifacts were pulled locally with tar-over-SSH because `rsync` was unavailable on the Runpod image.

Local artifact root:

```text
/work/ACM/ACM-clean/MoE/noxi_joint_settings/experiments_noxi_noxij_dyadic_roleheads_joint
```

Best checkpoints for reconstruction are present locally:

```text
/work/ACM/ACM-clean/MoE/noxi_joint_settings/experiments_noxi_noxij_dyadic_roleheads_joint/noxi_noxij_dyadic_roleheads_ccconly_seed13/model_best.pt
/work/ACM/ACM-clean/MoE/noxi_joint_settings/experiments_noxi_noxij_dyadic_roleheads_joint/noxi_noxij_dyadic_sharedhead_ccconly_seed13/model_best.pt
```

### Role-specific heads

Run:

```text
noxi_noxij_dyadic_roleheads_ccconly_seed13
```

Best epoch: `6`
Best overall val CCC: `0.9028220169477441`

Best-checkpoint metrics:

```text
overall CCC: 0.9028220169477441
NOXI CCC:    0.8262437713892979
NOXI-J CCC:  0.6627059918595385
novice CCC:  0.9157277826551604
expert CCC:  0.8587114860964329
```

### Shared 2-channel head

Run:

```text
noxi_noxij_dyadic_sharedhead_ccconly_seed13
```

Best epoch: `4`
Best overall val CCC: `0.9095221766682586`

Best-checkpoint metrics:

```text
overall CCC: 0.9095221766682586
NOXI CCC:    0.839191473471661
NOXI-J CCC:  0.683118987440375
novice CCC:  0.9156446643063633
expert CCC:  0.8715939864250254
```

Interpretation: the shared 2-channel dyadic head beat the role-specific-head variant overall and on both NOXI and NOXI-J. This supports the idea that the old preserved NOXI-J model's strength came from joint NOXI+NOXI-J dyadic shared modeling more than from separate role heads.

PinSoRo should continue tomorrow: first run the joint CC+CR head-specialist temporal queue, then decide whether to run the CR-social imbalance queue.
