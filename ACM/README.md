# ACM

ACM now contains one active modelling family: turn-level engagement regression on NOXI and NOXI-J using paired novice/expert interval manifests and TCN models. The current pipeline keeps three unimodal turn-level backbones, adds winner-only multimodal fusion on the empirically strongest backbone, and retains one controlled legacy-window comparison on that same backbone. Transformer or XGBoost baselines, PCA or random-projection branches, and old prebuilt dyadic tensor generation remain out of scope.

## Active pipeline

1. Build organizer-derived manifests with `scripts/build_manifests_from_organizer.py`.
2. Align each registered feature set to 25 Hz with `scripts/noxi_prepare_feature_tensors_25hz.py`.
3. Fit a train-only z-score normalizer and export the `raw` transformed branch with `scripts/noxi_fit_apply_feature_transform.py`.
4. Build paired turn manifests from transcripts with `scripts/noxi_build_turn_manifest.py`.
5. Optionally build paired legacy-window manifests with `scripts/noxi_build_window_manifest.py`.
6. Train three unimodal turn-level TCN variants with `scripts/train_tcn_turns.py`.
7. Resolve the majority-winning unimodal backbone and representative audio, text, and visual streams from completed turn runs.
8. Join the selected unimodal turn manifests into multimodal manifests with `scripts/noxi_build_multimodal_turn_manifest.py` and train winner-only multimodal runs with `scripts/train_tcn_multimodal.py`.
9. Run the same winning backbone on the legacy window manifests through `scripts/train_tcn_turns.py`, and run the winner-only multimodal combinations on legacy-window manifests through `scripts/train_tcn_multimodal.py`.
10. Summarize completed runs with `scripts/collect_results.py`.

The unimodal sweep remains one feature set at a time. Novice and expert streams are paired on the same interval, but their per-person tensors remain separate until the model forward pass. Multimodal runs fuse modalities within each role first and then pass the fused role streams into the winning backbone.

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

Run the winner-only multimodal and legacy-window ablations after unimodal results already exist:

```bash
bash ACM/scripts/run_training.sh 4 5
```

Submit the full dependency-aware HPC flow through SLURM:

```bash
bash ACM/scripts/submit_training_steps.sh 1 2 3 4 5
```

The submit wrapper now adds an extra dependency edge so steps `4` and `5` wait for steps `1` to `3` as well as preprocessing. Multimodal manifest joins are rebuilt lazily in step `4`, and multimodal legacy-window joins are rebuilt lazily in step `5`, if the required winner-specific manifests are not already present.

Collect result tables from completed experiments:

```bash
python ACM/scripts/collect_results.py
```

Resolve the current winning unimodal backbone and representative feature sets without printing the full tables:

```bash
python ACM/scripts/collect_results.py --resolve-turn-backbone --selection-format env
```

## Key artifacts

- `outputs/model_raw_manifest_train_with_split.csv`: session-role target manifest from organizer data.
- `outputs/model_raw_manifest_streams_train.csv`: stream inventory used for feature-set alignment.
- `outputs/manifests/model_processed_manifest_<feature_set>_25hz.csv`: aligned role-level tensors.
- `outputs/manifests/model_processed_manifest_<feature_set>_raw.csv`: normalized role-level tensors.
- `outputs/manifests/model_processed_manifest_<feature_set>_raw_turns.csv`: paired turn manifest used for training.
- `outputs/manifests/model_processed_manifest_<feature_set>_raw_windows.csv`: paired legacy-window manifest used for the controlled unit-of-analysis comparison.
- `outputs/manifests/model_processed_manifest_<combo_name>_multimodal_turns.csv`: joined multimodal turn manifest for winner-only fusion runs.
- `outputs/experiments/<run_name>/`: saved config, best checkpoint, validation metrics, and prediction CSVs.
- `outputs/experiments/<run_name>/val_gate_weights.csv`: mean validation gate weights for multimodal gated-fusion runs.
- `outputs/experiments/<run_name>/val_submission_format/`: organizer-style session tree with `*.engagement.prediction.csv` files, written in addition to the existing long-form `val_predictions.csv` export.

## Documentation

- `docs/01_codebase_structure.md`
- `docs/02_preprocessing_progress.md`
- `docs/03_tcn_architecture.md`