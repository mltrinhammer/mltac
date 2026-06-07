# PinSoRo Stage 2 Handoff - 2026-06-06

## Scope

This handoff describes the repository state, completed PinSoRo Stage 2
implementation, remaining Stage 2 experiments, and recommended GPU training
setup as of 2026-06-06.

Do not extend the work beyond Stage 2 without a separate decision. Existing
NoXi/NoXi-J modelling and training scripts must remain unchanged.

## Repository Orientation

Active repository:

```text
/work/ACM/mltac-main/ACM
```

Original PinSoRo archives:

```text
/work/ACM/PinSoRo/
```

Important documentation:

```text
docs/04_pinsoro_stage1.md
docs/05_pinsoro_handoff.md
docs/06_pinsoro_stage2_handoff_2026-06-06.md
```

PinSoRo Stage 1 outputs:

```text
cache/pinsoro/                         extracted organizer data
processed/pinsoro/                     aligned and normalized tensors
outputs/pinsoro/manifests/             tensor manifests
outputs/pinsoro/windows/               training-ready window manifests
outputs/pinsoro/validation/            Stage 1 validation reports
```

Stage 1 is complete. The shared Stage 1 data contains:

- 56 sessions shared across all 9 modalities
- 1,008 normalized session-role tensors
- 27,026 shared canonical windows
- 0 reported tensor-integrity errors

The data pipeline uses fixed 10-second windows:

- 250 frames at 25 Hz
- stride 62 frames, approximately 2.5 seconds
- identical canonical window boundaries across modalities and architectures

The storage-heavy directories are expected:

- `cache/pinsoro/`: approximately 123 GB
- `processed/pinsoro/`: approximately 133 GB
- original PinSoRo archives: approximately 124 GB

## Modelling Contract

PinSoRo uses one combined CC/CR training setup.

- CC: purple and yellow are supervised participants.
- CR: purple is supervised; yellow is retained only as dyadic context.
- Task engagement has 4 classes.
- Social engagement has 5 classes.
- Blank, unknown, and unavailable labels are ignored through per-head masks.
- Domain metadata is retained for masking and reporting only.
- Domain and other non-temporal metadata are not model inputs.
- Primary metric: Cohen's kappa.
- Additional metrics: macro-F1 and accuracy.
- Checkpoint selection uses the mean of overall task and social kappa.

The three architectures are:

1. `simple`
   - Individual-role baseline.
   - Receives one role only.
   - Purple and yellow examples share model parameters.
   - No partner information enters the model.
   - CR yellow does not appear as an individual training example.

2. `dyadic_shared`
   - Receives synchronized purple and yellow inputs.
   - Uses one encoder over concatenated role features.
   - Uses joint role-output classification heads.
   - CR yellow is available as context but has zero supervision mask.

3. `attention`
   - Receives synchronized purple and yellow inputs.
   - Uses a shared person-level encoder.
   - Uses shared attention over synchronized self/partner hidden states.
   - Uses shared task and social classification heads across roles.
   - CR yellow is available as context but has zero supervision mask.

## Completed Stage 2 Implementation

All Stage 2 modelling code was added as PinSoRo-specific files. Existing NoXi
model and trainer files were not modified.

Data loading and batching:

```text
src/acm_pipeline/pinsoro_data.py
```

This module provides:

- individual and dyadic window-manifest readers
- fixed-window tensor loading
- supervision masking
- bounded NPZ tensor caching
- session-grouped training batches to reduce repeated NPZ decompression

Models:

```text
src/acm_pipeline/pinsoro_models_tcn.py
```

This module implements:

- `PinSoRoIndividualTCN`
- `PinSoRoDyadicSharedTCN`
- `PinSoRoAttentionTCN`
- shared model factory `build_pinsoro_tcn`

Training utilities:

```text
src/acm_pipeline/pinsoro_train_utils.py
```

This module implements:

- masked multitask cross-entropy
- Cohen's kappa
- macro-F1
- accuracy
- overall, domain, role, and domain-role metric outputs
- validation and test prediction exports

Single-run trainer:

```text
scripts/train_pinsoro_tcn.py
```

The trainer:

- trains one feature set and one architecture
- reconstructs overlapping window logits onto session timelines
- selects the best checkpoint using mean task/social validation kappa
- exports validation metrics and predictions
- exports test predictions
- excludes CR yellow from test prediction output

Full feature-ablation launcher:

```text
scripts/run_pinsoro_training.sh
```

The launcher covers:

- 9 unimodal feature sets
- 3 architectures
- 27 runs per seed

Result collector:

```text
scripts/collect_pinsoro_results.py
```

## Verification Already Completed

The new code passed:

- Python syntax checks
- shell syntax checks
- complete 27-command launcher dry run
- manifest and tensor-schema checks across all 9 modalities
- real-data forward and backward passes for all 3 architectures
- masked multitask loss and nonzero-gradient checks
- optimizer-step smoke test
- overlapping-window reconstruction smoke test
- metric-output smoke test
- CR yellow supervision-mask check
- end-to-end one-epoch `train_pinsoro_tcn.py` CLI smoke test

The CLI smoke test produced:

```text
config.json
model_best.pt
training_log.csv
metrics_overall.csv
metrics_by_domain.csv
metrics_by_role.csv
metrics_by_domain_role.csv
val_predictions.csv
test_predictions.csv
```

A CPU-only smoke-test environment exists at:

```text
/work/ACM/mltac-main/ACM/.venv-smoke
```

It contains PyTorch 2.12.0+cpu and is approximately 938 MB. It is suitable for
development checks, not full training.

## Remaining Stage 2 Work

Stage 2 implementation is complete and training-ready. Stage 2 experiments are
not complete.

The next agent should:

1. Confirm the cluster GPU Python environment has CUDA-enabled PyTorch.
2. Dry-run the full launcher and inspect the generated commands.
3. Run the full 27-experiment grid for seed 13.
4. Monitor failed jobs, GPU utilization, RAM, and shared-storage throughput.
5. Rerun incomplete or failed experiments.
6. Collect results with `scripts/collect_pinsoro_results.py`.
7. Inspect overall, CC/CR, role, and head-specific metrics.
8. Repeat the grid with additional seeds if resources permit.
9. Document the completed Stage 2 experimental results.

Do not consider Stage 2 complete until the training grid has finished and the
results have been collected and inspected.

## Recommended GPU Setup

Each experiment should use one GPU. Do not use multiple GPUs for a single
experiment; the 27 runs are independent and should be parallelized across
experiments.

Recommended starting allocation:

- 4 concurrent GPUs
- 1 experiment per GPU
- 8 to 16 CPU cores per experiment
- 32 to 64 GB RAM per experiment
- `batch_size=32` initially

The likely bottleneck is NPZ decompression and shared-storage throughput rather
than model compute. Start with 4 concurrent jobs, monitor GPU utilization and
I/O performance, and only increase concurrency when storage throughput remains
healthy.

For one seed:

- 27 total experiments
- approximately 6 to 7 sequential experiments per GPU with 4 GPUs

For three seeds:

- 81 total experiments
- 8 to 9 GPUs are preferable if storage throughput supports that concurrency

High-dimensional visual modalities may require a smaller batch size if GPU
memory is insufficient.

## Training Commands

Dry-run the complete grid:

```bash
cd /work/ACM/mltac-main/ACM
DRY_RUN=1 bash scripts/run_pinsoro_training.sh
```

Run the complete grid using the active GPU-enabled Python environment:

```bash
cd /work/ACM/mltac-main/ACM
bash scripts/run_pinsoro_training.sh
```

Specify a Python executable explicitly when needed:

```bash
PYTHON_BIN=/path/to/gpu/python bash scripts/run_pinsoro_training.sh
```

Run selected models or features:

```bash
MODELS="simple dyadic_shared" \
FEATURE_SETS="audio_egemaps visual_openface" \
bash scripts/run_pinsoro_training.sh
```

Run multiple seeds:

```bash
SEEDS="13 17 23" bash scripts/run_pinsoro_training.sh
```

Collect completed results:

```bash
python scripts/collect_pinsoro_results.py
```

Default experiment output:

```text
outputs/pinsoro/experiments/
```

Default collected summary:

```text
outputs/pinsoro/results_summary.csv
```

## Operational Notes

- Full training should use a CUDA-enabled project or cluster environment, not
  `.venv-smoke`.
- The launcher currently runs experiments sequentially inside one shell
  process. Parallel GPU execution should be handled by separate jobs or a
  scheduler, with each job assigned a subset of models/features.
- Keep run names unique when splitting work across jobs.
- Do not delete `cache/pinsoro/` or `processed/pinsoro/` before training.
- Do not modify the NoXi/NoXi-J scripts while completing PinSoRo Stage 2.
- Use the existing canonical manifests so architecture comparisons remain
  controlled.
