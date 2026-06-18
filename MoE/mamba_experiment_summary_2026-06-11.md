# Mamba Experiment Summary

Date: 2026-06-11

This summarizes the PinSoRo dyadic Mamba experiment before removing the bulky
checkpoints, prediction dumps, and submission-format outputs. The experiment
trained Mamba modality experts for CC and CR, then evaluated the same small
combiner families used elsewhere in MoE.

Feature order for listed combiner weights is:

1. `visual_videomae`
2. `audio_w2vbert2`
3. `text_xlm_roberta`

## Retained Paths Removed After Summary

The following mamba-specific artifacts were deleted after this note was saved:

| Path | Approx. size | Reason |
| --- | ---: | --- |
| `ACM/MoE/experiments/mamba_cc_experts/` | 1.7G | Mamba CC checkpoints, predictions, logits, diagnostics. |
| `ACM/MoE/experiments/mamba_cr_experts/` | 553M | Mamba CR checkpoints, predictions, logits, diagnostics. |
| `ACM/MoE/experiments/mamba_smoke/` | 147M | One-off smoke run output. |
| `ACM/MoE/experiments/mamba_cc_combiners/` | 116K | Reproducible from expert score exports; summarized below. |
| `ACM/MoE/experiments/mamba_cr_combiners/` | 116K | Reproducible from expert score exports; summarized below. |
| `ACM/MoE/mamba/` | 128K | Mamba-only scripts and bytecode; no shared data/manifests. |

Shared `moe_data`, `noxi_data`, manifests, processed tensors, and non-mamba
experiment outputs were left untouched.

## Overall Result

| Setup | CC mean kappa | CR mean kappa | Four-head mean | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Best fair mamba setting per domain | 0.4255 | 0.2313 | 0.3284 | Below the current clean metadata-head + two_head + HMM no-prior line at 0.3853. |
| Best diagnostic/oracle mamba setting per domain | 0.4538 | 0.2587 | 0.3562 | Still below the HMM no-prior baseline; uses validation-fitted upper-bound settings. |
| Current clean deployable line | - | - | 0.3853 | From `experiment_score_summary_2026-06-10.md`; metadata-head + two_head + HMM no-prior. |

The mamba branch did not justify keeping its artifacts. CC results were
reasonable but not competitive after comparing against the cleaned temporal
decoding line. CR results were the main blocker: task was moderate, but social
remained very weak across expert and combiner variants.

## Expert Results

| Domain | Expert | Task kappa | Social kappa | Mean kappa | Interpretation |
| --- | --- | ---: | ---: | ---: | --- |
| CC | visual_videomae | 0.3426 | 0.4156 | 0.3791 | Strongest CC single expert. |
| CC | audio_w2vbert2 | 0.3021 | 0.3639 | 0.3330 | Useful but behind visual. |
| CC | text_xlm_roberta | 0.1837 | 0.2478 | 0.2157 | Weakest CC expert. |
| CR | visual_videomae | 0.3347 | 0.0550 | 0.1948 | Best CR single expert, but social is poor. |
| CR | audio_w2vbert2 | 0.1887 | 0.0615 | 0.1251 | Slightly better CR social than visual, weak overall. |
| CR | text_xlm_roberta | 0.2010 | -0.0157 | 0.0927 | CR social is worse than chance agreement. |

## Combiner Results

| Domain | Mode | Task kappa | Social kappa | Mean kappa | Weights / note |
| --- | --- | ---: | ---: | ---: | --- |
| CC | uniform | 0.3941 | 0.4569 | 0.4255 | Fair best; equal logit blend. |
| CC | shared | 0.3839 | 0.4470 | 0.4155 | Train-fit weights `[0.65, 0.30, 0.05]`; overweights visual/audio, nearly ignores text. |
| CC | two_head | 0.3770 | 0.4418 | 0.4094 | Task `[0.70, 0.25, 0.05]`; social `[0.55, 0.35, 0.10]`. |
| CC | metadata_router | 0.3463 | 0.4193 | 0.3828 | Metadata routing underperformed simple blends. |
| CC | val_two_head_upper | 0.4187 | 0.4547 | 0.4367 | Validation-fitted diagnostic upper bound. |
| CC | val_role_head_upper | 0.4499 | 0.4577 | 0.4538 | Best CC diagnostic result; not deployable as fair validation use. |
| CR | uniform | 0.4066 | -0.0082 | 0.1992 | Task improves, social collapses. |
| CR | shared | 0.3698 | 0.0737 | 0.2218 | Train-fit weights `[0.55, 0.45, 0.00]`; text dropped. |
| CR | two_head | 0.3532 | 0.0737 | 0.2135 | Task `[0.50, 0.50, 0.00]`; social `[0.55, 0.45, 0.00]`. |
| CR | metadata_router | 0.3764 | 0.0862 | 0.2313 | Fair best CR mamba, but still weak because social remains low. |
| CR | val_two_head_upper | 0.4044 | 0.0550 | 0.2297 | Validation-fitted diagnostic does not recover social. |
| CR | val_metadata_router_upper | 0.4523 | 0.0651 | 0.2587 | Best CR diagnostic result; task improves, social remains poor. |

## Interpretation

The mamba architecture did not improve the current PinSoRo MoE direction.
Compared with the current fair metadata-head + two_head + HMM no-prior result
of 0.3853 mean combined kappa, the best fair mamba combination reached about
0.3284 across CC/CR task/social.

The failure is asymmetric. CC mamba experts are usable and benefit from simple
logit blending, but this does not exceed the better temporal decoding line. CR
task can reach moderate kappa, especially under uniform or validation-fitted
blends, but CR social remains near zero or negative in most settings. The CR
social result is not a combiner issue alone: all three mamba experts are weak
for CR social before combination.

Text is not carrying enough signal in this branch. Train-fitted combiners often
assign text little or zero weight, especially for CR. Visual and audio dominate
the useful mamba signal.

The practical conclusion is to drop the mamba branch and keep work focused on
the metadata-head/HMM line and the documented CR social class-specific error
analysis. The large mamba artifacts are not needed for ongoing experiments once
this summary is retained.
