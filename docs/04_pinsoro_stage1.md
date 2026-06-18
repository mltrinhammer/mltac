# PinSoRo Stage 1

Stage 1 prepares PinSoRo for the ACM TCN experiments without changing the
existing NoXi pipeline.

## Contract

- Dataset: one combined `pinsoro` dataset with explicit `CC` or `CR` domain.
- Roles: `purple` and `yellow`.
- Targets: task engagement (4 classes) and social engagement (5 classes).
- Supervision: both CC roles; purple only for CR; no test labels.
- Features: aligned to 25 Hz and normalized using training-split statistics.
- Windows: 250 frames with stride 62 by default.
- Individual and dyadic manifests use identical session window boundaries.

## Entry Point

The runner resolves the repository and adjacent `PinSoRo/` data directory:

```bash
bash /work/ACM/mltac-main/ACM/scripts/run_pinsoro_stage1.sh
```

The full extraction and tensor export are storage- and I/O-heavy but do not
require a GPU.

## Outputs

- `outputs/pinsoro/raw_manifest.csv`
- `outputs/pinsoro/raw_stream_manifest.csv`
- `outputs/pinsoro/manifests/<feature_set>_25hz.csv`
- `outputs/pinsoro/manifests/<feature_set>_raw.csv`
- `outputs/pinsoro/windows/<feature_set>_w250_s62_individual.csv`
- `outputs/pinsoro/windows/<feature_set>_w250_s62_dyadic.csv`
- `outputs/pinsoro/windows/shared_w250_s62_canonical.csv`
- `outputs/pinsoro/validation/`
