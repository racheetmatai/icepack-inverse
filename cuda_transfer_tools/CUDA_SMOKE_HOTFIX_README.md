# JOG exact-checkpoint CUDA smoke hotfix

This small package supplements immutable baseline bundle
`JOG_CUDA_BUNDLE_20260820_C`; it does not modify that bundle. It reloads the
exact `best_model.keras` checkpoint before validation predictions and portable
export, then runs a bounded three-epoch rehearsal of registered production job
`SQ01_CFG06_M01` with the globally frozen `lambda_L2=0`.

From `/home/jovyan`, verify the unpacked package:

```bash
python JOG_CUDA_SMOKE_HOTFIX_20260820_A/verify_cuda_smoke_hotfix.py \
  JOG_CUDA_SMOKE_HOTFIX_20260820_A
```

Run the focused regression test and smoke test with the already required XLA
runtime override:

```bash
cd /home/jovyan/JOG_CUDA_SMOKE_HOTFIX_20260820_A/icepack-mlp
python -m unittest tests.test_exact_checkpoint_reload -v

export XLA_FLAGS=--xla_gpu_cuda_data_dir=/srv/conda/envs/notebook
python -m production_training.cuda_smoke \
  --dataset-dir /home/jovyan/JOG_CUDA_BUNDLE_20260820_C/dataset \
  --split-bundle /home/jovyan/JOG_CUDA_BUNDLE_20260820_C/splits \
  --cuda-environment /home/jovyan/JOG_CUDA_RUNS/cuda_environment.json \
  --job-id SQ01_CFG06_M01 \
  --lambda-l2 0 \
  --epochs 3 \
  --output /home/jovyan/JOG_CUDA_RUNS/production_smoke/SQ01_CFG06_M01_E3
```

This is an operational gate, not an accepted model and not a test-region
evaluation. Do not place its model in the 660-job production registry.
