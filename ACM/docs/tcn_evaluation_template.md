# TCN Evaluation Template

Use this file to summarize completed TCN experiments. Keep entries brief and decision-focused. The goal is to help the next agent understand what was tried, what changed between runs, and what the results suggest for the next architecture choice.

## How To Fill This Out

For each completed run, add one compact entry under the relevant model family.

Include:

```text
model: short structural description
settings: only the settings that define the ablation/design choice
metrics: CCC first, then role-specific CCC if available
key insights: what the result suggests about future model design
next action: one concrete follow-up experiment or design change
```

Do not paste full logs. Link or name the run directory instead.

## Evaluation Metrics

Primary:

```text
CCC overall
```

Also report when available:

```text
CCC novice
CCC expert
MAE
RMSE
Pearson
```

For interaction models, also summarize diagnostics:

```text
attention: source, relative lag, near/far past, session phase
gates: mean gate by role/session/phase
```

## Compact Entry Format

```text
Run:
Model:
Settings:
Metrics:
Diagnostics:
Key insights:
Next action:
```

## Completed Experiments

### Dyadic TCN: Shared Encoder, Shared Head

```text
Run:
Model: mixed dyadic TCN encoder + shared 2-channel head
Settings:
Metrics:
Diagnostics:
Key insights:
Next action:
```

### Dyadic TCN: Shared Encoder, Role-Specific Heads

```text
Run:
Model: mixed dyadic TCN encoder + separate novice/expert heads
Settings:
Metrics:
Diagnostics:
Key insights:
Next action:
```

### Partner-Lag TCN

```text
Run:
Model: separate role TCNs + separate role heads + fixed lagged partner hidden states
Settings: e.g. partner lags = -75, -750 frames
Metrics:
Diagnostics:
Key insights:
Next action:
```

### Attention TCN

```text
Run:
Model: separate role TCNs + role-specific attention heads
Settings: e.g. attention_context = joint, attention_past_frames = 1500, exclude_current_frame = true
Metrics:
Diagnostics: e.g. partner information attended from far past; own information attended from near past
Key insights:
Next action:
```

### Gated Pooled-Context TCN

```text
Run:
Model: separate role TCNs + pooled past partner context + learned role-specific gates
Settings: e.g. partner_pool_frames = 750, gate_type = scalar, current frame excluded
Metrics:
Diagnostics: e.g. novice mean gate > expert mean gate; gates increase late in session
Key insights:
Next action:
```

## Ablation Planning Notes

Use the filled entries to decide which design axis to change next:

```text
representation: role-level vs dyadic
encoder sharing: shared vs separate role encoders
head sharing: shared vs separate role heads
partner timing: fixed lags vs pooled past vs attention
context source: self vs partner vs joint
context window: 30s vs 60s vs longer
gate type: scalar vs channel
feature transform: raw vs shared PCA/RP vs role-specific PCA/RP
```

Prefer changing one axis at a time.
