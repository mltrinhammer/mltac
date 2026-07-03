# CC Annotator Sensitivity Reproducibility Package

This compact package contains the material needed to reproduce and explain the CC annotator-sensitivity analysis reported for the paper.

It uses the same target construction, subset definitions, HMM selection, and soft-kappa calculations mirrored by the CR sensitivity package.

## What Is Reported In The Paper

The paper reports three values for each CC head:

1. **Original canonical kappa**
   - Evaluated only on frames with a valid original canonical annotation.
   - Targets are hard one-hot labels.
   - The soft-kappa implementation is equivalent to ordinary Cohen's kappa on this subset.

2. **Extended soft kappa**
   - Evaluated on the extended/augmented set: canonical frames plus frames where canonical was blank but numbered annotations existed.
   - Canonical frames remain hard one-hot targets.
   - Added numbered-annotation frames use a normalized vote distribution as the target.

3. **Extended confidence-weighted soft kappa**
   - Same extended frame set and same soft targets.
   - Added frames are weighted by annotation confidence, defined as the maximum vote share.
   - Canonical frames have confidence 1.0.

## Reported Values

The exact values are also in `reported_values.csv`.

| Head | Original canonical kappa | Extended soft kappa | Extended confidence-weighted soft kappa |
|---|---:|---:|---:|
| CC task | 0.428847 | 0.227192 | 0.271776 |
| CC social submitted | 0.553552 | 0.229522 | 0.290806 |

Additional added-only diagnostics:

| Head | Added-only soft kappa | Added-only confidence-weighted soft kappa |
|---|---:|---:|
| CC task | 0.108324 | 0.100233 |
| CC social submitted | 0.104371 | 0.102818 |

## Files

```text
docs/cc_hmm_sensitivity_method_for_cr.txt
```

The original explanatory note, including target construction, coverage reporting, metric definitions, HMM selection, exact formulas, and pseudocode.

```text
scripts/run_cc_annotator_sensitivity_2906.py
scripts/run_cc_annotator_sensitivity_submitted_social_2906.py
```

Scripts that produced the CC-task and submitted-CC-social sensitivity summaries on RunPod.

```text
outputs/cc_task/coverage.csv
outputs/cc_task/selected_by_confidence_weighted_soft_kappa.csv
outputs/cc_task/sensitivity_metrics.csv
```

CC-task sensitivity outputs. Note: this directory's selected CSV also contains an older hidden-attention social row. For the paper's submitted social value, use `outputs/cc_social_submitted/` instead.

```text
outputs/cc_social_submitted/coverage.csv
outputs/cc_social_submitted/selected_by_confidence_weighted_soft_kappa.csv
outputs/cc_social_submitted/sensitivity_metrics.csv
```

Submitted CC-social sensitivity outputs.

## Source Paths On RunPod

These files were copied from the network-volume project at:

```text
/workspace/ACM
/workspace/ACM/ACM-clean
```

Original output directories:

```text
/workspace/ACM/ACM-clean/MoE/experiments/cc_annotator_sensitivity_2906
/workspace/ACM/ACM-clean/MoE/experiments/cc_annotator_sensitivity_submitted_social_2906
```

Original scripts:

```text
/workspace/ACM/run_cc_annotator_sensitivity_2906.py
/workspace/ACM/run_cc_annotator_sensitivity_submitted_social_2906.py
```

## Reproduction Notes

The sensitivity analysis did not retrain models with additional numbered annotations. It used already-trained validation logits, applied the relevant submitted-model bias where applicable, ran HMM/Viterbi decoding, then evaluated the same predictions against different target subsets:

- `canonical`: original canonical labels only.
- `numbered_blank`: frames where canonical was blank and numbered annotations supplied votes.
- `augmented`: canonical plus numbered_blank.

For CR, the most important points to mirror are:

- canonical labels take precedence;
- numbered annotations fill only canonical blanks;
- added numbered annotations become soft vote-distribution targets;
- confidence is max vote share;
- HMM settings are selected by confidence-weighted soft kappa;
- report both extended soft kappa and extended confidence-weighted soft kappa.
