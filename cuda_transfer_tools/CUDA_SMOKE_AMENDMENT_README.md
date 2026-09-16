# CUDA smoke comparison amendment

This amendment finalizes the already completed three-epoch smoke without
retraining. The original smoke passed every model, scaler, hash, GPU, reload,
and determinism check, but falsely rejected the saved prediction CSV by
comparing parsed floats to their pre-serialization values. The independently
persisted prediction CSVs are exactly equal.

From `/home/jovyan`:

```bash
cd /home/jovyan/JOG_CUDA_SMOKE_HOTFIX_20260820_A/icepack-mlp
cp /home/jovyan/JOG_CUDA_SMOKE_AMENDMENT_20260820_B/finalize_cuda_smoke.py \
  production_training/finalize_cuda_smoke.py

python -m production_training.finalize_cuda_smoke \
  /home/jovyan/JOG_CUDA_RUNS/production_smoke/SQ01_CFG06_M01_E3
```

The command must report `status: complete`, exact row IDs, and zero maximum
prediction difference. It creates a new acceptance manifest and does not alter
the original failed diagnostic manifest.
