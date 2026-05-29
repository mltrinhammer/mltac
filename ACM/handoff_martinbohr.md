# Hand-Off: ACM / NOXI Training Overview For Martin

## Model Clarification

There are three different "simple vs separated" levels:

```text
1. Simple role-level TCN: scripts/train_tcn.py
   One role/person sample at a time.
   It does not see novice + expert together.
   It is not separated into two TCNs internally.
   It is just one TCN model trained across role-level examples.

2. Basic dyadic TCN: scripts/train_tcn_dyadic.py
   Sees novice and expert together as one combined input.
   Uses one shared TCN encoder over the concatenated dyadic feature vector.
   This is the one that "mixes" the roles from the beginning.

3. Interaction models:
   train_tcn_partner_lag.py
   train_tcn_attention.py
   train_tcn_gated_pool.py

   These split the input back into novice features and expert features.
   They use separate novice/expert TCN encoders, then combine information through lag, attention, or gated partner context.
```

So the clean phrasing is:

```text
The simple TCN is role-level and non-interactional.
The dyadic TCNs are the basic comparison models where novice and expert are combined from the start in one shared TCN encoder.
The later interaction models use separate role encoders.
```

## First Step

Before training models, the agent should inspect the organizer repository and the actual data setup.

The first output from the agent should be:

```text
1. I have read the hand-off.
2. I will start with data integration first, then move to modelling.
3. I will first discuss what the organizer repo already does with the data, then introduce the modelling plan.
```

The reason is simple: the models are already mostly built in ACM, but they are only useful if the actual features, labels, masks, and splits are loaded correctly.

## Big Picture

There are two parts to the work.

First:

```text
Use the organizer repo to understand and load the real data.
```

Second:

```text
Use ACM to train and compare the models.
```

The organizer repo should mainly help with:

```text
official data layout
feature paths
labels
splits
evaluation or submission format
```

ACM should remain the place for:

```text
model architectures
training scripts
diagnostics
model comparisons
```

## Where To Find More Information

Start here:

```text
README.md
docs/01_codebase_structure.md
docs/02_preprocessing_progress.md
docs/05_dyadic_representation.md
docs/03_tcn_architecture.md
docs/tcn_modelling.md
docs/tcn_evaluation_template.md
```

What the files are for:

```text
docs/01_codebase_structure.md
  Overview of the repository, scripts, source code, manifests, and outputs.

docs/02_preprocessing_progress.md
  What has been done in preprocessing and which feature branches exist.

docs/05_dyadic_representation.md
  How novice and expert are paired together in the dyadic tensors.

docs/03_tcn_architecture.md
  The TCN model family and smoke-test history.

docs/tcn_modelling.md
  Practical notes on the dyadic and interaction TCN experiments.

docs/tcn_evaluation_template.md
  Template for summarising completed runs.
```

## The Data Formats In Plain Terms

ACM has two main ways of feeding data into models.

### 1. Role-Level Data

This means:

```text
train on one person/role at a time
```

For example, one model input could be the novice's audio features and the target is the novice's engagement score.

This is used by the simple TCN:

```text
scripts/train_tcn.py
```

This model is useful for testing ordinary TCN settings before adding partner interaction.

### 2. Dyadic Data

This means:

```text
train on novice and expert together
```

The dyadic tensor format is:

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

Most of the interaction models use this dyadic format.

Important rule:

```text
training windows must stay inside one session
```

No window should mix the end of one session with the start of another.

## Why Start With A Simple TCN

Before testing interaction ideas, it is useful to train the simple TCN.

It answers:

```text
Can a normal temporal model predict engagement from one person's features?
```

It is also the best place to tune basic settings:

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

The TCN uses dilation internally. In layman terms, this means deeper TCN layers can look farther back in time without making the model huge.

If the simple TCN is weak, the interaction models may not be the problem. The issue could be features, labels, splits, or basic settings.

## Model Ladder

The suggested training order is from simplest to most complex.

### 1. Simple TCN

Script:

```text
scripts/train_tcn.py
```

This is one person/role at a time. It does not model interaction.

Purpose:

```text
test basic TCN settings and get a simple baseline
```

### 2. Basic Dyadic TCN With Shared Head

Script:

```text
scripts/train_tcn_dyadic.py --head-type shared
```

This puts novice and expert features together, but uses one shared prediction setup for both roles.

Purpose:

```text
basic dyadic comparison model
```

This is important because not all comparison models should already be role-specialised.

### 3. Dyadic TCN With Role-Specific Heads

Script:

```text
scripts/train_tcn_dyadic.py --head-type role_specific
```

This still uses both people together, but novice and expert get separate final prediction heads.

Purpose:

```text
test whether novice and expert need different mappings from features to engagement
```

This is likely important for NOXI because novice and expert are asymmetric roles.

### 4. Partner-Lag TCN

Script:

```text
scripts/train_tcn_partner_lag.py
```

This has separate TCN encoders for novice and expert. It asks whether the partner's earlier behaviour helps predict the target person's current engagement.

Suggested lags:

```text
3 seconds back
30 seconds back
```

At 25 Hz these are:

```text
-75 frames
-750 frames
```

### 5. Gated Pooled TCN

Script:

```text
scripts/train_tcn_gated_pool.py
```

This summarizes the partner's recent past and learns how much to use it.

Purpose:

```text
test whether partner history helps in a smooth and interpretable way
```

Recommended first setting:

```text
30 second partner pool
scalar gate
save gate diagnostics
```

### 6. Attention TCN

Script:

```text
scripts/train_tcn_attention.py
```

This lets the model search over earlier frames.

Purpose:

```text
test where in the past the model looks for useful interaction information
```

Recommended setting:

```text
joint past context
60 second window
exclude current frame
save attention diagnostics
```

### 7. Mixture-of-Experts Later

MoE should come after the data pipeline and simpler models are stable.

The first planned MoE should use controlled, interpretable experts:

```text
own-only expert
short partner-lag expert
long partner-lag expert
pooled partner-past expert
```

The router then learns when each type of explanation is useful.

## Basic Versus Role-Separated Models

This distinction should be clear in reporting.

Basic comparison models:

```text
simple TCN:
  one person at a time, no partner input

dyadic shared-head TCN:
  novice and expert together, but one shared prediction head
```

Role-aware or interaction models:

```text
dyadic role-specific-head TCN:
  novice and expert have separate final heads

partner-lag, attention, gated pooled TCN:
  use separate role encoders and explicit partner-history mechanisms
```

The basic models matter because they show whether the more complex interaction models are actually improving anything.

## Suggested Training Steps

1. Inspect the organizer repo and document what it already does with the data.
2. Confirm available features, labels, masks, splits, and evaluation format.
3. Convert or connect the organizer data to ACM's role-level and dyadic manifests.
4. Run a tiny simple TCN smoke test.
5. Run a tiny dyadic TCN smoke test.
6. Train the simple TCN and use it to tune basic TCN settings.
7. Train the dyadic shared-head TCN.
8. Train the dyadic role-specific-head TCN.
9. Train Partner-Lag TCN.
10. Train Gated Pooled TCN.
11. Train Attention TCN.
12. Summarize each run using `docs/tcn_evaluation_template.md`.
13. Move to MoE only after these runs are stable.

## What To Report Per Run

Please report:

```text
run name
model type
feature set
train/dev/test split
important settings
overall CCC
novice CCC
expert CCC
MAE or MSE if available
diagnostics produced
short interpretation
recommended next run
```

For interaction models, diagnostics are important because the goal is not only to improve the score. The goal is also to understand whether the model uses own behaviour, partner history, or both.

## Practical Starting Point

First practical target:

```text
make the organizer data load into ACM format
```

First modelling target:

```text
simple TCN on role-level eGeMAPS
```

First dyadic comparison:

```text
dyadic TCN with shared head
dyadic TCN with role-specific heads
```

First interaction model:

```text
gated pooled TCN with 30 second partner pool
```
