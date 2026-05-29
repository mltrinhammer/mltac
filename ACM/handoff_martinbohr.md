# Hand-Off: ACM / NOXI Model Training Overview

## Short Summary

The ACM repository contains the modelling experiments. The organizer repository should mainly be used to load the official data layout, splits, and potentially export predictions in the expected format.

The recommended setup is:

```text
organizer repo reads official data
  -> adapter converts data to ACM manifest/tensor format
  -> ACM trains the models
  -> ACM evaluates with CCC and diagnostics
  -> optional organizer-compatible prediction/submission export
```

This keeps the competition wrapper and the modelling code separate.

## Where To Find More Detail

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

Useful orientation:

```text
docs/01_codebase_structure.md
  Overview of scripts, source modules, manifests, and outputs.

docs/02_preprocessing_progress.md
  Current preprocessing status and supported feature branches.

docs/05_dyadic_representation.md
  The dyadic tensor format and novice/expert channel convention.

docs/03_tcn_architecture.md
  TCN model families and architecture details.

docs/tcn_modelling.md
  Practical modelling notes and run descriptions.

docs/tcn_evaluation_template.md
  Template for summarising training runs.
```

## Current Stable Data Contract

The stable ACM dyadic input is session-level tensors listed in a manifest such as:

```text
outputs/manifests/model_processed_manifest_audio_egemaps_raw_dyadic.csv
```

Each dyadic tensor is expected to contain:

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

The important rule is that training windows must never cross session boundaries.

## Current Model Setups

### 1. Dyadic TCN

This is the first model to train seriously.

It uses both roles' features and predicts both roles' engagement. The preferred NOXI setup is role-specific heads, because novice and expert roles are asymmetric.

Use it as the main baseline before testing interaction-specific models.

### 2. Partner-Lag TCN

This model asks whether one person's earlier behaviour helps predict the other person's current engagement.

Suggested lags:

```text
-75 frames  = partner 3 seconds back
-750 frames = partner 30 seconds back
```

This is interpretable because it avoids relying on the partner's exact same-time frame.

### 3. Attention TCN

This model lets the network search over a past interaction window.

Recommended setting:

```text
joint past context
60 second window
exclude current frame
save attention diagnostics
```

It is flexible, but less controlled than fixed-lag models, so the diagnostics are important.

### 4. Gated Pooled TCN

This model summarizes the partner's recent past, then learns how much to blend that partner context into the target role's prediction.

This is currently the most robust and interpretable interaction-focused setup.

Recommended first setting:

```text
30 second partner pool
scalar gate
save gate diagnostics
```

Also test a 60 second pool if training time allows.

### 5. Mixture-of-Experts Next

The next planned model is an interaction MoE.

The first version should not be a large unconstrained expert system. It should use controlled experts with clear interpretations:

```text
own-only expert
short partner-lag expert
long partner-lag expert
pooled partner-past expert
```

The router then learns when each explanation is useful for novice and expert engagement prediction.

This is relevant to later generalisation across NOXI, NOXI-J, and PinSoRo, but it should first be validated on NOXI dyadic regression.

## Suggested Training Steps

1. Confirm that the organizer data loader can access the actual NOXI / NOXI-J data.
2. Convert or adapt the loaded data into ACM's dyadic manifest/tensor format.
3. Run a tiny smoke test with the dyadic role-specific TCN.
4. Train the full dyadic role-specific TCN baseline.
5. Train Partner-Lag TCN with 3s and 30s partner lags.
6. Train Gated Pooled TCN with 30s partner pool.
7. Train Attention TCN with joint 60s past context and current-frame exclusion.
8. Summarize each run with `docs/tcn_evaluation_template.md`.
9. Only after this path is stable, move to the MoE architecture.

## What To Report Per Run

Please report at minimum:

```text
Run name
Model type
Feature set and transform
Train/dev/test split used
Important hyperparameters
Overall CCC
Novice CCC
Expert CCC
Other regression metrics if available
Diagnostics produced
Short interpretation
Recommended next run
```

For interaction models, diagnostics are as important as the score, because the goal is to understand which interaction information the model uses.

## Practical Starting Point

Recommended first serious run:

```text
audio eGeMAPS raw dyadic
Dyadic TCN
role-specific heads
```

Recommended first interaction run:

```text
audio eGeMAPS raw dyadic
Gated Pooled TCN
30 second partner pool
scalar gate
save gates
```

This gives a clean baseline and a clean interaction model before moving to more complex attention or MoE experiments.

