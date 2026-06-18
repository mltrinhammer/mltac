# NOXI / NOXI-J Encoder Sharing Results

Date: 2026-06-16

These runs compare shared vs separate role encoders while keeping the rest of the setup fixed:

- gated multimodal fusion
- post-prediction linear partner interaction
- CCC-only regression loss (`ccc_weight=1.0`, `mse_weight=0.0`)
- seed 13
- NOXI and NOXI-J trained as separate datasets

Outputs copied back from RunPod to:

`/work/ACM/ACM-clean/MoE/noxi_joint_settings/experiments_group_meanpool`

The queue uses `--skip-existing` and will skip these runs because each run directory contains `model_best.pt`.

| Dataset | Encoder | Best val CCC | MAE | RMSE | Pearson | Run directory |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| NOXI | shared | 0.8457678386 | 0.0644986941 | 0.0817275030 | 0.8606804610 | `noxi_audio_text_visual_w500_s125_postpred_linear_shared_encoder_seed13` |
| NOXI | separate | 0.8477020582 | 0.0593171898 | 0.0779969484 | 0.8496818520 | `noxi_audio_text_visual_w500_s125_postpred_linear_separate_encoder_seed13` |
| NOXI-J | shared | 0.5601852281 | 0.0698449157 | 0.0984227233 | 0.5613557729 | `noxij_audio_text_visual_w500_s125_postpred_linear_shared_encoder_seed13` |
| NOXI-J | separate | 0.5245324301 | 0.0800541740 | 0.1084572527 | 0.5664689358 | `noxij_audio_text_visual_w500_s125_postpred_linear_separate_encoder_seed13` |

Summary:

- NOXI: separate encoder narrowly wins over shared (`+0.0019342196` CCC).
- NOXI-J: shared encoder wins over separate (`+0.0356527980` CCC).
- The isolated encoder-sharing effect is mixed; shared is the safer default across NOXI-J, while separate may be worth retaining for NOXI-specific follow-up.
