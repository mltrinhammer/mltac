# ACM Preprocessing Pipeline

The active preprocessing branch is intentionally narrow. It produces one aligned role-level tensor branch per feature set, one normalized `raw` branch, and one paired turn manifest branch. Role-specific transform branches, PCA, random projection, and dyadic tensor materialization have been removed.

## Inputs

- Organizer-format `noxi/` and `noxij/` directories.
- Engagement targets named `<role>.engagement.annotation.csv`.
- Transcript files named `<role>.audio.transcript.annotation.csv`.
- Registered stream files listed in `src/acm_pipeline/feature_registry.py`.

## Step 0: raw manifests and cache links

`scripts/build_manifests_from_organizer.py` creates:

- `outputs/model_raw_manifest_train_with_split.csv`
- `outputs/model_raw_manifest_streams_train.csv`
- `cache/noxi_a/<split>` and `cache/noxi_b/<split>` symlinks to organizer data

The raw manifest records dataset, session, role, split, and target path. The stream manifest records which feature streams exist for each dataset, session, and role.

## Step 1: 25 Hz alignment

`scripts/noxi_prepare_feature_tensors_25hz.py` processes one feature set at a time.

For every valid session-role example it:

1. Loads the target sequence.
2. Loads each required feature stream.
3. Aligns each stream to the target grid at 25 Hz.
4. Concatenates aligned streams when a feature set uses multiple streams.
5. Writes one NPZ tensor with `x`, `y`, and `target_mask`.

Outputs:

- `processed/<feature_set>_25hz/<dataset>/<session_id>/<role>.<feature_set>.25hz.npz`
- `outputs/manifests/model_processed_manifest_<feature_set>_25hz.csv`
- `outputs/manifests/feature_status_<feature_set>_25hz.csv`

## Step 2: train-only normalization

`scripts/noxi_fit_apply_feature_transform.py --method raw` fits one shared z-score normalizer on `train_internal` rows only and applies it to every split.

Outputs:

- `processed/transformed/<feature_set>_raw/<dataset>/<session_id>/<role>.<feature_set>.raw.npz`
- `outputs/manifests/model_processed_manifest_<feature_set>_raw.csv`
- `outputs/transforms/<feature_set>_raw/normalizer.npz`
- `outputs/transforms/<feature_set>_raw/transform_config.json`

`raw` means normalized but not dimension-reduced.

## Step 3: paired turn manifest generation

`scripts/noxi_build_turn_manifest.py` converts normalized role-level manifests into paired turn manifests.

For each session it:

1. Locates the novice and expert role tensors.
2. Locates both transcript files.
3. Collects all speech onsets.
4. Merges consecutive utterances from the same speaker into one turn run.
5. Emits one turn row per speaker-change segment.

Each turn row stores:

- dataset and session identifiers
- split and feature metadata
- speaker who initiated the turn
- `start_frame` and `end_frame`
- relative paths to the novice and expert source tensors
- aligned lengths for both roles

Output:

- `outputs/manifests/model_processed_manifest_<feature_set>_raw_turns.csv`

## Representation assumptions

- Every turn covers the same frame interval for novice and expert.
- Preprocessing does not mix novice and expert feature channels together.
- The paired turn manifest is only an index. The source NPZ tensors remain the single source of truth.
- Turn filtering by minimum length happens inside `TurnDataset`, not during tensor export.

## Shell wrappers

- `scripts/run_preprocessing.sh` runs Steps 0 to 3 for all registered feature sets and skips artifacts that already exist.
- `scripts/submit_training_steps.sh` can submit preprocessing first and chain later training jobs behind it.