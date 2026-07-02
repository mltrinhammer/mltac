# Training Pipeline Track

This track points to the source scripts needed to retrain the CR models after regenerating the prepared tensors from organizer-provided PinSoRo embedding streams.

1. Regenerate tensors/manifests from organizer archives:

```bash
DATA_ROOT=/path/to/PinSoRo \
PYTHON_BIN="$(command -v python)" \
bash scripts/run_cr_preprocessing_from_embeddings.sh
```

2. Run partner/encoder architecture ablations:

```bash
python MoE/pinsoro_noxi_settings/run_cr_task_clean_arch_queue.py --python "$(command -v python)" --gpu 0
python MoE/pinsoro_noxi_settings/run_cr_social_clean_arch_queue.py --python "$(command -v python)" --gpu 0
```

3. Run feature-family ablations:

```bash
python MoE/pinsoro_noxi_settings/run_cr_final_arch_modality_ablation_queue.py --python "$(command -v python)" --gpu 0
```

The head-adapter social variants produced by the clean social queue are not part of the packaged partner/encoder ablation artifact set.
