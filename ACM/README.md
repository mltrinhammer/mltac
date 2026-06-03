# ACM

ACM now contains one active modelling pipeline: turn-level engagement regression on NOXI and NOXI-J using paired novice/expert turn manifests and TCN models. The cleaned codebase no longer includes fixed-window TCN trainers, transformer or XGBoost baselines, PCA or random-projection branches, or prebuilt dyadic tensor generation.

## Active pipeline

1. Build organizer-derived manifests with `scripts/build_manifests_from_organizer.py`.
2. Align each registered feature set to 25 Hz with `scripts/noxi_prepare_feature_tensors_25hz.py`.
3. Fit a train-only z-score normalizer and export the `raw` transformed branch with `scripts/noxi_fit_apply_feature_transform.py`.
4. Build paired turn manifests from transcripts with `scripts/noxi_build_turn_manifest.py`.
5. Train three turn-level TCN variants with `scripts/train_tcn_turns.py`.
6. Summarize completed runs with `scripts/collect_results.py`.

Each training run is unimodal: one feature set at a time. Novice and expert streams are paired on the same turn interval, but their per-person tensors remain separate until the model forward pass.

## Registered feature sets

- `audio_egemaps`
- `audio_w2vbert2`
- `text_xlm_roberta`
- `visual_swin`
- `visual_openface`
- `visual_openpose`
- `visual_clip`
- `visual_dino`
- `visual_videomae`

## Quick start

Build raw manifests from organizer-format data:

```bash
python ACM/scripts/build_manifests_from_organizer.py --data-root <project-root>
```

Run preprocessing for all registered feature sets:

```bash
bash ACM/scripts/run_preprocessing.sh
```

Train the three active turn-level models:

```bash
bash ACM/scripts/run_training.sh 1 2 3
```

Submit preprocessing and selected training steps through SLURM with dependencies:

```bash
bash ACM/scripts/submit_training_steps.sh 1 2 3
```

Collect result tables from completed experiments:

```bash
python ACM/scripts/collect_results.py
```

## Key artifacts

- `outputs/model_raw_manifest_train_with_split.csv`: session-role target manifest from organizer data.
- `outputs/model_raw_manifest_streams_train.csv`: stream inventory used for feature-set alignment.
- `outputs/manifests/model_processed_manifest_<feature_set>_25hz.csv`: aligned role-level tensors.
- `outputs/manifests/model_processed_manifest_<feature_set>_raw.csv`: normalized role-level tensors.
- `outputs/manifests/model_processed_manifest_<feature_set>_raw_turns.csv`: paired turn manifest used for training.
- `outputs/experiments/<run_name>/`: saved config, best checkpoint, validation metrics, and prediction CSVs.

## Documentation

- `docs/01_codebase_structure.md`
- `docs/02_preprocessing_progress.md`
- `docs/03_tcn_architecture.md`