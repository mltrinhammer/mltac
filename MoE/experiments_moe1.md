# MoE 1 Experiment Setups

Date: 2026-06-09

Scope: first MoE pass for PinSoRo using three modality experts per domain. CC
runs first; CR uses the same architecture and combiner ablations with separate
normalization, experts, and router weights.

## Fixed Data

Use prepared MoE data only:

```text
ACM/MoE/moe_data/
```

Domains:

```text
CC and CR, always run and fit separately
```

Expert manifests:

```text
ACM/MoE/moe_data/outputs/windows_w2400_s1200_by_domain/visual_videomae/visual_videomae_w2400_s1200_dyadic_cc.csv
ACM/MoE/moe_data/outputs/windows_w2400_s1200_by_domain/audio_w2vbert2/audio_w2vbert2_w2400_s1200_dyadic_cc.csv
ACM/MoE/moe_data/outputs/windows_w2400_s1200_by_domain/text_xlm_roberta/text_xlm_roberta_w2400_s1200_dyadic_cc.csv
```

Fast-loading cache:

```text
ACM/MoE/moe_data/processed/domain_norm_mmap
```

Training split:

```text
train_internal
```

Validation split:

```text
val_internal
```

## Fixed Expert Architecture

All modality experts use the selected VideoMAE temporal setting, without
attention:

```text
model:           dyadic_shared
window / stride: 2400 / 1200
levels:          5
kernel size:     11
hidden channels: 64
dropout:         0.2
causal TCN:      yes
attention:       no
heads:           task_engagement and social_engagement
```

Rationale: `dyadic_shared` consumes both roles at each time step, so each
expert hidden state is based on synchronized purple and yellow features.

## Experiment E1: Train CC Modality Experts

Train three independent CC experts:

```text
E1a: CC visual_videomae dyadic_shared TCN
E1b: CC audio_w2vbert2 dyadic_shared TCN
E1c: CC text_xlm_roberta dyadic_shared TCN
```

Outputs required from each expert:

```text
model_best.pt
model_last.pt
training_log.csv
metrics_by_domain.csv
metrics_by_domain_role.csv
val_predictions.csv
val_prediction_scores.csv.gz
```

The `val_prediction_scores.csv.gz` files are required for MoE combiner fitting
and ablations.

## Lightweight Combiner Experiments

After the frozen experts finish, run these cheap ablations from saved expert
score files. They do not retrain the TCN experts.

```text
E2: best_single
    Evaluate each expert alone on validation. This anchors the MoE against the
    best individual modality.

E3: uniform
    Equal-weight logit average: (video + audio + text) / 3. No learned
    parameters.

E4: prob_uniform
    Equal-weight probability average. This tests whether probability averaging
    is more stable than raw logit averaging.

E5: shared
    One non-negative video/audio/text weight vector fit on train logits and
    evaluated on validation. This is the clean shared-router result.

E6: two_head
    Separate train-fitted weight vectors for task and social heads. This is the
    expected main MoE 1 candidate.

E7: role_head
    Separate train-fitted weight vectors for role x head: purple-task,
    yellow-task, purple-social, yellow-social. This checks whether the dyadic
    roles need different modality mixtures.

E8: val_shared_upper
E9: val_two_head_upper
E10: val_role_head_upper
    Fit the same router families directly on validation logits and evaluate on
    validation. These are optimistic upper-bound diagnostics, not clean main
    results.
```

Main claim candidates should come from train-fitted routers only:

```text
shared
two_head
role_head
```

The validation-fitted modes are useful for estimating whether more flexible
routing has headroom, but they should not be reported as the proper validation
result.



## Metadata Experiments

These are PinSoRo-specific ablations. Keep them out of the core metadata-free
MoE result unless they clearly help and competition rules confirm age/gender are
allowed and available for inference.

```text
E11: metadata_router
    Included in the default combiner phase. Keep the trained visual/audio/text
    experts frozen. Train a small linear softmax router that receives role
    metadata and outputs modality weights.

    Inputs per frame/role/head:
    - expert logits from video, audio, and text
    - role metadata only: normalized age and gender one-hot

    Explicitly excluded:
    - participant ID
    - session ID
    - any combined CC+CR fit

    Output:
    - modality weights over video/audio/text

    Main clean version:
    - train router on train-domain logits and metadata
    - evaluate on val-domain
    - run separately for CC and CR

    Diagnostic version:
    - val_metadata_router_upper fits on validation and is optimistic only

E12: metadata_prediction_head
    Retrain modality experts with metadata injected after the TCN encoder and
    before the task/social prediction heads.

    Inputs per window:
    - normal dyadic modality tensor
    - per-role metadata vector repeated/broadcast at the head stage

    Architecture:
    - TCN encoder unchanged
    - concatenate hidden state with metadata before 1x1 task/social heads
    - compare no metadata, age-only, gender-only, age+gender
    - use metadata dropout so the model cannot rely entirely on static fields

    This is a deep-training ablation. It is now wired as a separate overnight
    branch with isolated output roots, followed by the same frozen-logit
    combiner ablations used for the metadata-free experts.
```

Recommended order:

```text
1. Finish metadata-free MoE 1 for CC and CR.
2. The overnight runner builds a reusable MoE metadata table from raw age/gender annotations.
3. E11 metadata_router runs with the other lightweight combiners after expert training.
4. E12 metadata_prediction_head also runs in the default overnight sequence,
   but under separate output roots so it can be ignored if it overfits.
```

Generalization note:

```text
NoXi compatibility should be preserved by keeping the base MoE metadata-free.
Metadata-conditioned variants must be optional adapters that can be disabled
when metadata is unavailable or not comparable across datasets.
```

## Overnight Run Order

Run only E1 overnight first if scripts are not yet proven:

```text
1. train visual_videomae expert
2. train audio_w2vbert2 expert
3. train text_xlm_roberta expert
4. export train and validation logits for all three experts
```

Then run E2-E11 as lightweight combiner experiments. Combiner runs should be
fast compared with expert training. E12 trains a second set of metadata-head
experts and then applies the same combiner suite to those expert outputs.

## Advancement Criteria

Prefer the simplest combiner that improves over the best single CC expert on
`val-cc`.

Track separately:

```text
CC task kappa
CC social kappa
mean(CC task, CC social)
class recall and prediction prevalence for both heads
session-level collapse or dominance
```

Do not advance a combiner that improves one head by destroying the other.



## Automation Scripts

Run the full prepared sequence with:

```bash
cd /work/ACM/mltac-main
/work/ACM/mltac-main/ACM/.venv-gpu/bin/python ACM/MoE/run_moe1_overnight.py --gpus 0,1,2
```

Default order:

```text
1. extract reusable age/gender metadata table
2. train/export CC visual, audio, text metadata-free experts in parallel
3. fit/evaluate CC E2-E11 combiner ablations
4. train/export CC visual, audio, text metadata-head experts in parallel
5. fit/evaluate CC combiners on metadata-head experts
6. train/export CR visual, audio, text metadata-free experts in parallel
7. fit/evaluate CR E2-E11 combiner ablations
8. train/export CR visual, audio, text metadata-head experts in parallel
9. fit/evaluate CR combiners on metadata-head experts
```

The output roots are domain-specific:

```text
ACM/MoE/experiments/moe1_cc_experts/
ACM/MoE/experiments/moe1_cc_combiners/
ACM/MoE/experiments/moe1_cr_experts/
ACM/MoE/experiments/moe1_cr_combiners/
ACM/MoE/experiments/moe1_cc_metadata_head_experts/
ACM/MoE/experiments/moe1_cc_metadata_head_combiners/
ACM/MoE/experiments/moe1_cr_metadata_head_experts/
ACM/MoE/experiments/moe1_cr_metadata_head_combiners/
```

Rerunning the expert launcher uses `--resume` and skips expert folders with a
`.complete` marker. Rerunning the combiner refreshes the same domain/mode
summary files; it does not write into the other domain's folders.

For separate runs:

```bash
# CC only
/work/ACM/mltac-main/ACM/.venv-gpu/bin/python ACM/MoE/run_moe1_overnight.py --domains CC --gpus 0,1,2

# CR only
/work/ACM/mltac-main/ACM/.venv-gpu/bin/python ACM/MoE/run_moe1_overnight.py --domains CR --gpus 0,1,2

# Fit combiners only, after experts are complete
/work/ACM/mltac-main/ACM/.venv-gpu/bin/python ACM/MoE/run_moe1_overnight.py --domains CC CR --skip-experts

# Skip metadata-head deep training if you only want the metadata-free MoE
/work/ACM/mltac-main/ACM/.venv-gpu/bin/python ACM/MoE/run_moe1_overnight.py --domains CC CR --skip-metadata-head
```

## Later CR Repeat

After CC MoE 1 is debugged, repeat the same setup using CR-only manifests:

```text
ACM/MoE/moe_data/outputs/windows_w2400_s1200_by_domain/*/*_cr.csv
```

Do not reuse CC normalizers, class weights, routers, or calibration parameters
for CR.
