# PinSoRo Performance and Resume Handoff - 2026-06-08

## Scope

This note covers PinSoRo only. Existing NoXi and EDA worktree changes are
unrelated and must remain untouched.

The current machine is CPU-only. Full training should run on the separate
four-GPU machine.

## Reconstructed Grid State

The seed-13 grid has 27 runs. Eight run directories exist:

- Complete: all three `audio_egemaps` runs and
  `audio_w2vbert2_dyadic_shared`.
- Interrupted: `audio_w2vbert2_simple`, `audio_w2vbert2_attention`,
  `text_xlm_roberta_simple`, and `text_xlm_roberta_dyadic_shared`.
- Not started: the remaining 19 runs.

The four old interrupted runs do not contain `model_last.pt`, so they cannot
resume exactly and will restart once. Completed runs are skipped. New and
restarted runs save resumable last-state checkpoints after every epoch.

## Material Performance Findings

The original default data path processed W2V-BERT at only 1.7 windows/second.
The main cause was `torch.stack` invoking the host's 128-thread Torch CPU pool
for every batch. NumPy-based collation removes that overhead.

The normalized training tensors were also stored as compressed NPZ files.
Dense embedding tensors compress poorly and were decompressed again in every
epoch and architecture run. The new mmap cache stores the five training arrays
as separate uncompressed NPY files.

Measured W2V-BERT data-loading throughput on the CPU host:

- original compressed NPZ plus Torch collation: 1.7 windows/second
- fixed NumPy collation plus compressed NPZ: 105.4 windows/second
- fixed NumPy collation plus mmap cache: 202.9 windows/second

The combined improvement is approximately 119x for this loader benchmark.
Class-weight initialization fell from 45.20 seconds to 1.35 seconds and
produced identical weights.

The completed mmap cache contains all 1,008 normalized PinSoRo tensors and is
95 GiB. It was validated against 100 arrays sampled from the canonical NPZ
sources. `/work/ACM` had approximately 872 TiB available before the build.

## Other Performance and Reliability Changes

The trainer and launchers now:

- load each tensor once, rather than once per prediction head, for class weights
- avoid per-batch CUDA synchronization when calculating mean training loss
- check empty supervision before CUDA transfer
- use pinned-memory and non-blocking transfers on CUDA
- track reconstruction coverage with compact boolean arrays
- record train, validation, epoch, and throughput timings in `training_log.csv`
- save `model_last.pt` atomically after every epoch
- resume incomplete runs automatically, including optimizer, early-stopping,
  sampler, and Torch RNG state
- append launcher logs when resuming instead of overwriting them

Automatic mixed precision is intentionally not enabled by default. Benchmark
it on the GPU host before changing numerical behavior for the experiment grid.

## Completed Verification

- Full mmap cache build: 1,008 tensors, 95 GiB, 7.6 minutes
- Idempotent cache rebuild: all 1,008 tensors verified and skipped
- Random cache validation: 20 sources and 100 arrays exactly matched NPZ data
- Python and shell syntax checks
- Four-GPU launcher dry run: 4 complete and 23 pending runs
- End-to-end CPU mmap training smoke test
- Interrupted-run resume smoke test from epoch 1 to epoch 2

## Continue on the Four-GPU Host

The cache already exists at:

```text
/work/ACM/mltac-main/ACM/processed/pinsoro_mmap
```

If the GPU host does not share this filesystem, rebuild it after transferring
the canonical processed tensors:

```bash
python scripts/pinsoro_build_mmap_cache.py --workers 4
```

Inspect pending commands:

```bash
python scripts/run_pinsoro_training_4gpu.py \
  --python /path/to/cuda-enabled/python \
  --gpus 0,1,2,3 \
  --dry-run
```

Start or resume the grid by removing `--dry-run`. The launcher defaults to
`num_workers=0`, which was fastest in isolated loader benchmarks and avoids
large multiprocessing shared-memory batches. After a few epochs, compare GPU
utilization and timing fields before testing `--num-workers 1`.

Inspect `train_seconds`, `val_seconds`, `train_windows_per_second`, and
`val_windows_per_second` in each `training_log.csv`. These fields show whether
the remaining limit is GPU compute, validation reconstruction, or data supply.

## Remaining GPU-Host Questions

The following cannot be resolved on the CPU-only host and should be measured
before making further changes:

- GPU utilization and memory use for each architecture and modality
- whether full validation reconstruction every epoch dominates elapsed time
- whether `--num-workers 1` improves overlap or creates shared-memory pressure
- whether automatic mixed precision improves throughput without changing
  checkpoint selection or final metrics
- whether four concurrent runs saturate shared-storage throughput

Do not reduce validation frequency or enable mixed precision for the experiment
grid without first confirming that the resulting metric and checkpoint behavior
is acceptable.

## Additional Handoff

The selected multimodal workflow and organizer-format PinSoRo prediction tree
are documented in `docs/08_pinsoro_multimodal_and_submission_handoff_2026-06-08.md`.
