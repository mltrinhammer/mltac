# Agent Handoff: RunPod Encoder-Sharing Experiments

Date: 2026-06-16

## Current Objective

Run the first controlled shared-vs-separate encoder experiments for NOXI/Noxi-J and PinSoRo.

Keep metadata out for now. The isolated experimental variable is encoder sharing:

```text
shared encoder vs separate role/person encoders
```

Partner modelling is fixed for this batch:

```text
linear post-prediction/post-logit partner interaction
```

Do not switch to hidden-state partner modelling in this batch.

## RunPod Access

Current pod TCP SSH:

```bash
ssh root@38.128.232.9 -p 37312 -i /home/ucloud/.ssh/runpod_acm_ed25519
```

RunPod UI SSH label shown by user:

```bash
ssh 9w2i5s3i3vy53j-64410c7b@ssh.runpod.io -i /.ssh/id_ed25519
```

Use the TCP route above from this machine. The local private key is:

```text
/home/ucloud/.ssh/runpod_acm_ed25519
```

## RunPod Environment

Verified on pod:

```text
Workspace: /workspace
Project: /workspace/ACM/ACM-clean
GPU: NVIDIA A100 80GB PCIe
CPU count: 252
Python: 3.10.12
PyTorch: 2.1.0+cu118
CUDA available: True
```

`tmux` is not installed. Persistent jobs are launched with `nohup` wrapper scripts under:

```text
/workspace/ACM/ACM-clean/run_logs/
```

## Data/Artifact Transfer State

Transferred and verified on RunPod:

```text
/workspace/ACM/ACM-clean                                                     ~589M
/workspace/ACM/mltac-main/ACM/MoE/noxi_joint_settings                       ~278M
/workspace/ACM/mltac-main/ACM/MoE/noxi_data/processed/normalized             36G, 456 files
/workspace/ACM/mltac-main/ACM/MoE/noxi_j_data/processed/normalized           18G, 306 files
/workspace/ACM/mltac-main/ACM/MoE/moe_data/outputs/windows_w2400_s1200       ~4.9M
/workspace/ACM/mltac-main/ACM/MoE/moe_data/processed/domain_norm             35G, 336 files
```

Symlinks created so clean-repo manifests resolve project-relative paths:

```text
/workspace/ACM/ACM-clean/MoE/noxi_data  -> /workspace/ACM/mltac-main/ACM/MoE/noxi_data
/workspace/ACM/ACM-clean/MoE/noxi_j_data -> /workspace/ACM/mltac-main/ACM/MoE/noxi_j_data
/workspace/ACM/ACM-clean/MoE/moe_data -> /workspace/ACM/mltac-main/ACM/MoE/moe_data
```

Manifest path checks passed for both NOXI and PinSoRo.

## Local Code Changes Made

Changed regression training defaults to CCC-only:

```text
scripts/train_mpii_group_meanpool_multimodal.py
scripts/train_tcn_multimodal.py
scripts/train_tcn_turns.py
```

Defaults are now:

```text
--ccc-weight 1.0
--mse-weight 0.0
```

Added shared/separate encoder support and post-prediction linear partner interaction for group regression:

```text
src/acm_pipeline/group_models.py
scripts/train_mpii_group_meanpool_multimodal.py
MoE/noxi_joint_settings/run_noxi_group_meanpool_regression_queue.py
```

Important: group regression now does post-prediction interaction, not hidden-state partner context:

```text
multimodal input -> gated fusion -> shared/separate encoder -> scalar role prediction
-> linear residual over [self_prediction, partner_prediction] -> CCC-only loss
```

Added shared/separate role encoder support for PinSoRo while preserving post-logit interaction:

```text
MoE/pinsoro_noxi_settings/train_person_interaction_fusion.py
MoE/pinsoro_noxi_settings/run_followup_experiment_queue.py
```

PinSoRo flow remains:

```text
purple/yellow multimodal input -> concat fusion -> shared/separate encoder
-> task/social logits -> linear post-logit partner interaction -> HMM after training
```

No metadata is included in this batch.

## Smoke Tests Already Passed

On RunPod, forward-pass smoke tests passed:

```text
group shared (2, 12, 2)
group separate (2, 12, 2)
pinsoro shared {'task': (2, 2, 12, 4), 'social': (2, 2, 12, 5)}
pinsoro separate {'task': (2, 2, 12, 4), 'social': (2, 2, 12, 5)}
```

Dry-runs passed for NOXI/Noxi-J queue and PinSoRo queue.

## Current Running Job

Current active queue:

```text
/workspace/ACM/ACM-clean/run_logs/run_noxi_encoder_queue.sh
PID file: /workspace/ACM/ACM-clean/run_logs/run_noxi_encoder_queue.pid
Outer log: /workspace/ACM/ACM-clean/run_logs/run_noxi_encoder_queue.outer.log
```

Launched command inside wrapper:

```bash
cd /workspace/ACM/ACM-clean
export PYTHONPATH=/workspace/ACM/ACM-clean
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
python3 MoE/noxi_joint_settings/run_noxi_group_meanpool_regression_queue.py --python python3 --num-workers 2 --gpu 0
```

The queue runs these sequentially:

```text
noxi_audio_text_visual_w500_s125_postpred_linear_shared_encoder_seed13
noxi_audio_text_visual_w500_s125_postpred_linear_separate_encoder_seed13
noxij_audio_text_visual_w500_s125_postpred_linear_shared_encoder_seed13
noxij_audio_text_visual_w500_s125_postpred_linear_separate_encoder_seed13
```

First attempt with `--num-workers 16` failed because a DataLoader worker was killed. Relaunched with `--num-workers 2`, which is running.

Current observed status at `2026-06-16T15:27:05+00:00`:

```text
Current run: noxi_audio_text_visual_w500_s125_postpred_linear_shared_encoder_seed13
epoch 1 completed
train_loss=0.334705757440897
val_ccc=0.8457678386180164
val_mae=0.0644986940536379
val_rmse=0.0817275030264946
val_pearson=0.86068046099821
best_epoch=1
```

The process was still alive after epoch 1.

Check status with:

```bash
ssh root@38.128.232.9 -p 37312 -i /home/ucloud/.ssh/runpod_acm_ed25519   'run=/workspace/ACM/ACM-clean/MoE/noxi_joint_settings/experiments_group_meanpool/noxi_audio_text_visual_w500_s125_postpred_linear_shared_encoder_seed13;    tail -20 "$run/training_log.csv" 2>/dev/null || true;    tail -30 /workspace/ACM/ACM-clean/run_logs/run_noxi_encoder_queue.outer.log;    pgrep -af "run_noxi|train_mpii_group_meanpool" || true;    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader'
```

## PinSoRo Queue: Not Yet Launched

Do not launch PinSoRo until NOXI/Noxi-J queue completes or the user asks to parallelize.

When ready, launch:

```bash
ssh root@38.128.232.9 -p 37312 -i /home/ucloud/.ssh/runpod_acm_ed25519
cd /workspace/ACM/ACM-clean
mkdir -p run_logs
cat > run_logs/run_pinsoro_encoder_queue.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail
cd /workspace/ACM/ACM-clean
export PYTHONPATH=/workspace/ACM/ACM-clean
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
python3 MoE/pinsoro_noxi_settings/run_followup_experiment_queue.py --python python3 --num-workers 2 --gpu 0 --apply-hmm
SH
chmod +x run_logs/run_pinsoro_encoder_queue.sh
nohup bash run_logs/run_pinsoro_encoder_queue.sh > run_logs/run_pinsoro_encoder_queue.outer.log 2>&1 & echo $! > run_logs/run_pinsoro_encoder_queue.pid
```

PinSoRo queue will run, sequentially:

```text
CC shared encoder + linear post-logit interaction + HMM
CC separate encoder + linear post-logit interaction + HMM
CR shared encoder + linear post-logit interaction + HMM
CR separate encoder + linear post-logit interaction + HMM
```

CR manifest supervision has already been checked:

```text
CR train/val: purple_supervised=yes, yellow_supervised=no
```

So CR trains/evaluates purple only while yellow remains available as partner/context input.

## Scientific Framing

For this batch:

```text
Question: Does role/person specialization improve over a shared behavioral encoder?
Fixed: multimodal input, prediction heads, linear post-prediction/logit partner interaction, CCC-only for regression.
Varied: shared vs separate encoders.
Deferred: metadata heads, TCN partner interaction, HMM transition-matrix variants, soft labels.
```

Terminology:

- For NOXI/Noxi-J, separate encoders are meaningful as novice/expert role-specialized encoders.
- For PinSoRo, separate encoders are purple/yellow role-specialized encoders.
- Avoid calling this a full Mixture-of-Experts unless later adding a learned router/gate. Safer wording: `role-specialized encoders` or `role-specialized experts`.

## Important Cautions

- Do not reintroduce hidden-state partner modelling for this batch.
- Do not add metadata yet.
- Keep `--num-workers 2` unless there is a strong reason to tune; `16` workers killed a DataLoader worker.
- The pod lacks `tmux`; use `nohup` wrapper scripts.
- `rsync` is not installed on the pod. Transfers used `tar | ssh` with remote `tar --no-same-owner`.
- RunPod paths use `/workspace/ACM/...`; local university paths use `/work/ACM/...`.
