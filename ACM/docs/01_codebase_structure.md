# ACM Codebase Structure

## Top-level layout

- `scripts/`: entry points for manifest building, preprocessing, training, SLURM submission, and result collation.
- `src/acm_pipeline/`: shared library code for manifests, alignment, turn construction, transforms, metrics, and TCN models.
- `processed/`: generated aligned and normalized NPZ tensors.
- `outputs/manifests/`: raw, aligned, normalized, paired-turn, paired-window, and multimodal-turn CSV manifests.
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

### `scripts/noxi_build_window_manifest.py`

Builds paired fixed-window manifests with the legacy 500-frame / 125-frame sliding-window configuration while keeping the same interval schema used by the turn trainer.

### `scripts/noxi_build_multimodal_turn_manifest.py`

Joins multiple unimodal paired turn manifests on the full interval key and emits one multimodal turn manifest with per-modality role tensor references.

### `scripts/train_tcn_turns.py`

Trains interval-level TCN models from paired manifests. It is still the canonical trainer for unimodal speech turns and now also handles the legacy fixed-window comparison because both interval families share the same manifest schema.

### `scripts/train_tcn_multimodal.py`

Trains winner-only multimodal turn models from joined multimodal manifests. It reuses the same optimization and artifact contract as `train_tcn_turns.py` and additionally writes `val_gate_weights.csv` for gated-fusion runs.

### `scripts/collect_results.py`

Scans completed experiment directories, prints unimodal turn tables, resolves the current winner backbone, and prints winner-only multimodal plus turn-vs-window comparison tables when those runs exist.

### `scripts/run_preprocessing.sh`

SLURM-friendly wrapper that runs the full preprocessing chain for all registered feature sets and can additionally build legacy-window manifests plus any currently resolvable winner-only multimodal manifests.

### `scripts/run_training.sh`

SLURM-friendly wrapper that trains the three unimodal turn steps, the winner-only multimodal step, and the winner-only legacy-window comparison step. The legacy-window step now covers both the unimodal winner backbone and the winner-only multimodal combinations. When needed, it lazily builds missing multimodal or window manifests before launching the corresponding runs.

### `scripts/submit_training_steps.sh`

Submits preprocessing first and then submits selected training steps with explicit dependency chaining. Steps `4` and `5` depend on preprocessing and any selected unimodal steps so winner selection is resolved on completed runs rather than future ones.

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

The active TCN model implementations: shared person-level baseline, shared dyadic baseline, role-attention model, and the role-wise multimodal fusion wrapper used for winner-only fusion runs.

### `src/acm_pipeline/transforms.py`

Train-only feature normalization utilities. Only `FeatureNormalizer` remains active.

### `src/acm_pipeline/turns.py`

Transcript parsing, speech-turn boundary computation, and legacy fixed-window interval generation. Consecutive utterances from the same speaker are merged into one turn run.

### `src/acm_pipeline/turn_data.py`

Paired interval manifest readers, unimodal and multimodal turn datasets, and batch collation logic for variable-length interval segments.

## Output contract

- Role-level tensors are stored per dataset, session, and role as NPZ files with `x`, `y`, and `target_mask` arrays.
- Turn and window manifests do not duplicate tensors. They reference the source novice and expert tensors and store `start_frame` and `end_frame` for each interval.
- Multimodal manifests keep the same interval rows but add per-modality role tensor references in a JSON column.
- Training outputs include `model_best.pt`, `config.json`, `metrics_overall.csv`, `metrics_by_role.csv`, `metrics_by_dataset.csv`, `metrics_by_session.csv`, and `val_predictions.csv`.
- Every trainer now also writes `val_submission_format/`, an organizer-style session tree containing `expert.engagement.prediction.csv` and `novice.engagement.prediction.csv` for dyadic NOXI/NOXI-J runs.
- Multimodal gated runs additionally write `val_gate_weights.csv`.