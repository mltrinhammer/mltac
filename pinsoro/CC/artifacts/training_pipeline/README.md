# Training Pipeline Track

This track documents full CC retraining. It does not include processed PinSoRo tensors.

Use the existing wrappers after placing or symlinking the processed tensor tree at `MoE/moe_data/processed`:

```bash
DEVICE=cuda bash scripts/train_cc_task_final.sh
DEVICE=cuda bash scripts/train_cc_social_final.sh
```

The bundled manifests are under `artifacts/manifests/windows_w2400_s1200/`.
