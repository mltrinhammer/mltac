# ACM Codebase Structure

## Top-level layout

- `scripts/`: entry points for manifest building, preprocessing, training, SLURM submission, and result collation.
- `src/acm_pipeline/`: shared library code for manifests, alignment, turn construction, transforms, metrics, and TCN models.
- `processed/`: generated aligned and normalized NPZ tensors.
- `outputs/manifests/`: raw, aligned, normalized, and paired-turn CSV manifests.
- `outputs/transforms/`: fitted normalizer artifacts for each feature set.
- `outputs/experiments/`: one directory per training run, containing configs, checkpoints, metrics, and validation predictions.
- `docs/`: minimal documentation for the active turn-level pipeline.

## Active scripts

### `scripts/build_manifests_from_organizer.py`

Scans organizer-format `noxi/` and `noxij/` directories, creates `ACM/cache/` symlinks, and writes the raw manifests consumed by preprocessing.

### `scripts/noxi_prepare_feature_tensors_25hz.py`

Builds one 25 Hz role-level tensor branch for a single registered feature set. It aligns required streams to the engagement target timeline and writes compressed NPZ tensors plus a processed manifest.

### `scripts/noxi_fit_apply_feature_transform.py`

Fits a train-only z-score normalizer on aligned tensors and exports the normalized `raw` branch. This is the only remaining transform branch.

### `scripts/noxi_build_turn_manifest.py`

Pairs novice and expert role tensors for each session, reads transcripts, computes turn boundaries from speaker changes, and writes the paired turn manifest used by training.

### `scripts/train_tcn_turns.py`

Trains the active turn-level TCN models from paired turn manifests. It supports three model types: `simple`, `dyadic_shared`, and `attention`.

### `scripts/collect_results.py`

Scans completed experiment directories and prints markdown tables over validation metrics.

### `scripts/run_preprocessing.sh`

SLURM-friendly wrapper that runs the full preprocessing chain for all registered feature sets.

### `scripts/run_training.sh`

SLURM-friendly wrapper that trains the three active model ladder steps over every available turn manifest.

### `scripts/submit_training_steps.sh`

Submits preprocessing first and then submits selected training steps with an `afterok` dependency on the preprocessing job.

## Active library modules

### `src/acm_pipeline/alignment.py`

Rate conversion utilities for mapping source feature streams onto the 25 Hz engagement timeline.

### `src/acm_pipeline/data.py`

Typed manifest loading for role-level tensors and the shared NPZ session loader used by later preprocessing and turn datasets.

### `src/acm_pipeline/dyadic_train_utils.py`

Shared metric and prediction writers for the two-target turn-level models.

### `src/acm_pipeline/feature_registry.py`

Registry of all supported unimodal feature sets and their required streams.

### `src/acm_pipeline/io.py`

CSV readers and writers plus low-level helpers for reading stream matrices and targets from local cache paths.

### `src/acm_pipeline/metrics.py`

Masked regression losses and reporting metrics, including CCC.

### `src/acm_pipeline/models_tcn.py`

The active TCN model implementations: shared person-level baseline, shared dyadic baseline, and role-attention model.

### `src/acm_pipeline/transforms.py`

Train-only feature normalization utilities. Only `FeatureNormalizer` remains active.

### `src/acm_pipeline/turns.py`

Transcript parsing and turn boundary computation. Consecutive utterances from the same speaker are merged into one turn run.

### `src/acm_pipeline/turn_data.py`

Paired turn manifest reader, turn dataset, and batch collation logic for variable-length turn segments.

## Output contract

- Role-level tensors are stored per dataset, session, and role as NPZ files with `x`, `y`, and `target_mask` arrays.
- Turn manifests do not duplicate tensors. They reference the source novice and expert tensors and store `start_frame` and `end_frame` for each turn.
- Training outputs include `model_best.pt`, `config.json`, `metrics_overall.csv`, `metrics_by_role.csv`, `metrics_by_dataset.csv`, `metrics_by_session.csv`, and `val_predictions.csv`.