# PinSoRo CR Reproducibility Package

This folder contains the CR-specific code and manifests needed to reproduce the
PinSoRo CR models and paper ablations.

## Contents

- `src/acm_pipeline/`: shared PinSoRo data loading, metrics, TCN blocks, and
  prediction-writing utilities.
- `scripts/train_pinsoro_tcn.py`: shared helper functions imported by the CR
  training scripts.
- `MoE/pinsoro_noxi_settings/train_person_interaction_fusion_temporal.py`:
  main CR model training script.
- `MoE/pinsoro_noxi_settings/apply_person_interaction_hmm_active_heads.py`:
  HMM/Viterbi validation sweep and test prediction export.
- `MoE/pinsoro_noxi_settings/run_cr_task_clean_arch_queue.py`: clean CR-task
  architecture ablations.
- `MoE/pinsoro_noxi_settings/run_cr_social_clean_arch_queue.py`: clean
  CR-social architecture ablations.
- `MoE/pinsoro_noxi_settings/run_cr_final_arch_modality_ablation_queue.py`:
  final-architecture feature-family ablations.
- `scripts/run_cr_preprocessing_from_embeddings.sh`: CR preprocessing entry
  point from organizer-provided PinSoRo embedding streams to the final 25 Hz
  tensors and 2400/1200 window manifests.
- `scripts/pinsoro_build_raw_manifests.py`,
  `scripts/pinsoro_prepare_feature_tensors_25hz.py`,
  `scripts/pinsoro_fit_apply_domain_feature_transform.py`, and
  `scripts/pinsoro_build_window_manifests.py`: optional preprocessing and
  manifest-building scripts for reconstructing the prepared tensor/manifests
  from extracted feature streams.
- `MoE/moe_data/outputs/windows_w2400_s1200/`: canonical dyadic window
  manifests for audio, text, and visual features, filtered to CR rows.
- `MoE/moe_data/outputs/participant_metadata.csv`: participant metadata used
  by `metadata_mode=age_gender_role`.

The same manifests and metadata are also duplicated under `data/` for clarity.


## Two-Track Reproducibility Layout

This package now includes two complementary reproduction tracks under
`artifacts/`:

- `artifacts/training_pipeline/`: end-to-end scripts and commands for retraining
  from organizer-provided PinSoRo embedding streams after regenerating the
  prepared tensors.
- `artifacts/inference_only/`: selected checkpoints, configs, HMM settings,
  compact metrics, and submission-format predictions for reproducing inference
  without retraining. Prepared `.npz` tensors and large per-frame score dumps are
  intentionally not included.

The inference-only track contains:

- submitted clean CR-task and CR-social model artifacts;
- paper feature-family ablations;
- paper partner/encoder ablations, excluding head-adapter variants;
- the CR soft-kappa sensitivity analysis following the CC-agent method note.

See `artifacts/inference_only/ARTIFACT_MANIFEST.csv` for the run-level mapping.

## Data Assumptions

The reproducibility boundary is the organizer-provided PinSoRo archives with
precomputed SSI embedding streams. The code expects `.stream` and `.stream~`
files for the selected streams:

- `audio.w2vbert2_embeddings`
- `audio.xlm_roberta_embeddings`
- `videomae`

The package does not re-extract W2V-BERT2, XLM-RoBERTa, or VideoMAE embeddings
from raw audio/video. If the organizer data contains those embedding streams,
the preprocessing script below regenerates the 25 Hz `.npz` tensors and window
manifests used by the training queues.

The included CSV manifests describe the prepared 25 Hz PinSoRo CR feature
tensors and expect the corresponding `.npz` tensors to be available at the paths
listed inside the manifest columns. If the tensors are not present, regenerate
them from the organizer embedding streams.

The manifests are CR-filtered copies of the original all-domain manifests and
contain 565 dyadic windows per feature family.

Windowing for these runs is:

- window size: 2400 frames = 96 s at 25 Hz
- stride: 1200 frames = 48 s
- overlap: 50%

Feature families:

- `audio_w2vbert2`, 1024 dimensions
- `text_xlm_roberta`, 768 dimensions
- `visual_videomae`, 1408 dimensions

## Environment

Install the minimal runtime dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The experiments were run with PyTorch and CUDA. Use `--device cpu` only for
small smoke tests; full reproduction is intended for GPU.

## Rebuild Tensors And Manifests

Place the PinSoRo organizer archives in a data directory containing
`train-cr.zip`, `train-cc.zip`, `val.zip`, and `test.zip`, then run:

```bash
DATA_ROOT=/path/to/PinSoRo \
PYTHON_BIN="$(command -v python)" \
bash scripts/run_cr_preprocessing_from_embeddings.sh
```

This writes:

- raw manifests under `outputs/pinsoro/`
- aligned 25 Hz tensors under `processed/pinsoro/`
- domain-normalized tensors under `MoE/moe_data/processed/domain_norm/`
- final 2400/1200 window manifests under
  `MoE/moe_data/outputs/windows_w2400_s1200/`

The training queues below read the `MoE/moe_data` paths by default.

## Reproduce Final CR Models

Run the clean architecture queues. These train the selected final model families
and the architecture ablations used for the paper table, then apply HMM/Viterbi
decoding.

```bash
python MoE/pinsoro_noxi_settings/run_cr_task_clean_arch_queue.py \
  --python "$(command -v python)" \
  --gpu 0

python MoE/pinsoro_noxi_settings/run_cr_social_clean_arch_queue.py \
  --python "$(command -v python)" \
  --gpu 0
```

Important final selected configurations:

- CR-task: `pinsoro_cr_task_shared_tcn_no_partner_delta010_metadata_seed13`
- CR-social: `pinsoro_cr_social_dyadic_shared_tcn_pre_encoder_delta010_metadata_seed13`

Outputs are written to:

- `MoE/experiments/pinsoro_cr_task_clean_arch_delta010_metadata/`
- `MoE/experiments/pinsoro_cr_social_clean_arch_delta010_metadata/`

Each run directory contains raw validation metrics, validation prediction
scores, best checkpoints, and HMM outputs under `hmm_smoothing_cr_task/` or
`hmm_smoothing_cr_social/`.

## Reproduce Feature-Family Ablations

These runs keep the final selected architecture fixed and vary only the input
feature families:

- audio + text + visual
- audio + text
- audio + visual
- text + visual
- audio
- text
- visual

```bash
python MoE/pinsoro_noxi_settings/run_cr_final_arch_modality_ablation_queue.py \
  --python "$(command -v python)" \
  --gpu 0
```

Outputs are written to:

`MoE/experiments/pinsoro_cr_final_arch_modality_ablation_2026_06_29/`

## HMM/Viterbi Post-Processing

The queues automatically call:

```bash
python MoE/pinsoro_noxi_settings/apply_person_interaction_hmm_active_heads.py
```

The HMM sweep estimates transition probabilities from training labels and
selects validation settings over transition strengths and transition mixes. The
selected HMM setting is stored as `best_hmm_setting.json` in each run's HMM
output directory.

## Paper Scope

For the CR paper results, use:

1. Clean partner/encoder ablations from `run_cr_task_clean_arch_queue.py` and
   `run_cr_social_clean_arch_queue.py`. In the packaged artifact set,
   head-adapter social variants are deliberately excluded.
2. Final-architecture feature-family ablations from
   `run_cr_final_arch_modality_ablation_queue.py`.
3. Sensitivity results from
   `artifacts/inference_only/sensitivity/`, which mirrors the CC-agent approach:
   canonical annotations, numbered annotations where canonical labels are blank,
   augmented targets, soft kappa, and confidence-weighted soft kappa.
4. HMM/Viterbi-decoded validation metrics for the final reported comparisons.

Temporal alternatives such as transition-fusion layers, hidden partner-temporal
fusion, forward-filter decoding, and CRF-style sequence losses were explored
during model development but are not part of the main clean CR ablation table.
