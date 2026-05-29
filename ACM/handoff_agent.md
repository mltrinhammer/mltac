# Hand-Off: Agent Workflow For ACM / NOXI Training

## First Response To The Collaborator

When this hand-off is given to another agent, its first message to the collaborator should do three things:

1. Say that it has read and understood this hand-off.
2. Say that it will start with data integration first, because the models only make sense once the actual organizer-repo data layout is understood.
3. Say that the next discussion should cover what has already been done in the organizer repository, and only after that move into the modelling plan.

Suggested first response:

```text
I have read the hand-off. I will start by inspecting the organizer repository and the actual data setup, then I will connect that data to the ACM input format. After we know exactly what the organizer repo already does with features, labels, splits, and evaluation, I will move on to the modelling plan and train the ACM models one at a time.
```

## Big Picture

There are two jobs. Do them in this order.

First, understand the official organizer repository and the real data:

```text
Where are the features?
Where are the labels?
What splits are used?
What format does the organizer repo expect?
What has already been processed?
What is still missing?
```

Second, train the ACM models:

```text
simple TCN baseline
basic dyadic TCN baselines
role-aware dyadic TCN
interaction models
MoE later
```

Do not begin by rewriting models. The first technical task is data integration.

## Repositories

ACM modelling repository:

```text
C:\Users\anec\projects\mltac\ACM
https://github.com/mltrinhammer/mltac.git
branch: main
```

External organizer repository, if available locally:

```text
C:\Users\anec\projects\mltac\_external\MultiMediate26
```

Use the organizer repository as the official data and evaluation wrapper. Use ACM as the modelling codebase.

## What Integration Means

In plain terms, the organizer repo knows how the competition data is laid out. ACM knows how to train the models we have built.

The integration should look like this:

```text
organizer-format data
  -> small adapter/conversion step
  -> ACM manifest and tensors
  -> ACM training scripts
  -> ACM metrics and diagnostics
  -> optional organizer-style prediction/submission files
```

Do not replace ACM with the organizer baseline. Build the bridge between them.

## First Data Tasks

Before modelling, inspect the organizer repo and report:

```text
1. Which datasets are present: NOXI, NOXI-J, PinSoRo, or only some of them.
2. Which feature files exist and at what frame rate.
3. Which labels exist and how they are aligned to features.
4. Which train/dev/test or internal splits are defined.
5. Whether missing values, masks, or invalid labels are already handled.
6. Whether the organizer repo already builds tensors, dataloaders, or only raw file lists.
7. Whether the organizer repo has an official evaluation script or submission format.
8. What needs to be converted before ACM can train.
```

The first useful output is a data-readiness note, not a model result.

## ACM Input Formats

ACM currently has two relevant input styles.

### Role-Level Input

This is the simpler input used by the plain TCN:

```text
one role/person at a time
x = features over time
y = engagement over time
target_mask = which labels are valid
```

Example manifest:

```text
outputs/manifests/model_processed_manifest_audio_egemaps_raw.csv
```

This is useful for testing basic TCN settings such as:

```text
window size
stride
hidden channels
number of TCN levels
kernel/filter size
dropout
learning rate
CCC loss weight
```

### Dyadic Input

This is the paired novice/expert input used by the interaction models:

```text
x           [time, 2 * feature_dim]
y           [time, 2]
target_mask [time, 2]
role_order  ["novice", "expert"]
```

For raw eGeMAPS:

```text
feature_dim per role = 88
x dim = 176
target channel 0 = novice
target channel 1 = expert
```

Example manifest:

```text
outputs/manifests/model_processed_manifest_audio_egemaps_raw_dyadic.csv
```

Important rule:

```text
Training windows must never cross session boundaries.
```

## Relevant ACM Files

Preprocessing and tensor preparation:

```text
scripts/noxi_prepare_feature_tensors_25hz.py
scripts/noxi_fit_apply_feature_transform.py
scripts/noxi_fit_apply_feature_transform_by_role.py
scripts/noxi_build_dyadic_tensors.py
```

Model scripts:

```text
scripts/train_tcn.py
scripts/train_xgboost.py
scripts/train_transformer.py
scripts/train_tcn_dyadic.py
scripts/train_transformer_dyadic.py
scripts/train_xgboost_dyadic.py
scripts/train_tcn_partner_lag.py
scripts/train_tcn_attention.py
scripts/train_tcn_gated_pool.py
```

Shared modules:

```text
src/acm_pipeline/dyadic_data.py
src/acm_pipeline/dyadic_train_utils.py
src/acm_pipeline/metrics.py
src/acm_pipeline/models_tcn.py
```

## Modelling Ladder

Train from simplest to most complex. Each step answers a different question.

### Step 1: Simple Role-Level TCN

Question:

```text
Can a basic TCN predict engagement from one person's features?
```

This is not an interaction model. It is useful because it tests ordinary TCN settings before adding dyadic complexity.

Example:

```powershell
python scripts\train_tcn.py `
  --manifest outputs\manifests\model_processed_manifest_audio_egemaps_raw.csv `
  --run-name noxi_egemaps_raw_simple_tcn
```

Settings to tune here first:

```text
--window-size
--stride
--hidden-channels
--levels
--kernel-size
--dropout
--lr
--ccc-weight
```

The TCN uses increasing dilation internally. More levels means the model sees a longer temporal history.

### Step 2: Basic Dyadic TCN, Shared Head

Question:

```text
If we put novice and expert features together, can one shared model predict both engagement curves?
```

This is a basic comparison model. It does not have separate novice/expert prediction heads.

```powershell
python scripts\train_tcn_dyadic.py `
  --manifest outputs\manifests\model_processed_manifest_audio_egemaps_raw_dyadic.csv `
  --head-type shared `
  --run-name noxi_egemaps_raw_dyadic_shared_head
```

Use this as a simple dyadic baseline.

### Step 3: Dyadic TCN, Role-Specific Heads

Question:

```text
Does prediction improve when novice and expert get separate final prediction heads?
```

The temporal encoder is still shared, but the final mapping to engagement is different for novice and expert.

```powershell
python scripts\train_tcn_dyadic.py `
  --manifest outputs\manifests\model_processed_manifest_audio_egemaps_raw_dyadic.csv `
  --head-type role_specific `
  --run-name noxi_egemaps_raw_dyadic_role_heads
```

This is preferred for NOXI because novice and expert roles are asymmetric.

### Step 4: Partner-Lag TCN

Question:

```text
Does the partner's earlier behaviour help predict the target person's current engagement?
```

This model has separate novice and expert TCN encoders. It is more role-separated than the basic dyadic TCN.

```powershell
python scripts\train_tcn_partner_lag.py `
  --manifest outputs\manifests\model_processed_manifest_audio_egemaps_raw_dyadic.csv `
  --partner-lags -75 -750 `
  --run-name noxi_egemaps_raw_partner_lag_3s_30s
```

Lag meaning at 25 Hz:

```text
-75  = partner 3 seconds earlier
-750 = partner 30 seconds earlier
```

Avoid same-time partner input unless running a deliberate ablation.

### Step 5: Gated Pooled TCN

Question:

```text
How much should the model use the partner's recent past?
```

This model pools the partner's previous frames, then learns a gate that controls how strongly that partner history enters the prediction.

```powershell
python scripts\train_tcn_gated_pool.py `
  --manifest outputs\manifests\model_processed_manifest_audio_egemaps_raw_dyadic.csv `
  --partner-pool-frames 750 `
  --gate-type scalar `
  --save-gates `
  --run-name noxi_egemaps_raw_gated_pool_30s
```

Also test:

```text
--partner-pool-frames 1500
```

That corresponds to 60 seconds at 25 Hz.

### Step 6: Attention TCN

Question:

```text
Which earlier frames does the model look at?
```

Attention is flexible, but less controlled than fixed lags or pooled history.

```powershell
python scripts\train_tcn_attention.py `
  --manifest outputs\manifests\model_processed_manifest_audio_egemaps_raw_dyadic.csv `
  --attention-context joint `
  --attention-past-frames 1500 `
  --exclude-current-frame `
  --save-attention `
  --run-name noxi_egemaps_raw_attention_joint_60s
```

### Step 7: MoE Later

Do not start here. MoE should come after the data path and simpler comparisons are stable.

Planned first MoE experts:

```text
own-only expert
short partner-lag expert
long partner-lag expert
pooled partner-past expert
```

The router learns when to trust each expert.

## Basic Versus Role-Separated Models

Be explicit when reporting results:

```text
Basic role-level TCN:
  one person at a time; no dyadic partner input.

Basic dyadic shared-head TCN:
  novice and expert features together; one shared output head.

Dyadic role-specific-head TCN:
  novice and expert features together; separate final heads.

Partner-lag / attention / gated models:
  separate role encoders; explicit interaction mechanisms.
```

The basic models are important comparison points. They show whether the added interaction machinery actually helps.

## Evaluation

Use ACM regression metrics and always respect `target_mask`.

Report:

```text
run name
model type
feature set
split
important settings
overall CCC
novice CCC
expert CCC
MAE or MSE if available
diagnostic files produced
short interpretation
next action
```

For each completed run, fill in:

```text
docs/tcn_evaluation_template.md
```

## Minimum Acceptance Criteria

Before long training:

```text
1. Organizer repo and actual data layout have been inspected.
2. Current feature, label, split, and evaluation status is reported.
3. ACM-compatible role-level or dyadic manifest is available.
4. At least one tensor can be loaded and inspected.
5. A tiny simple TCN run completes.
6. A tiny dyadic TCN run completes.
7. Metrics are written.
8. No training window crosses a session boundary.
```

Only after this should the agent run full training jobs.

