# JOG production launcher for LEAP Pangeo

This source-only package supplements immutable input bundle
`JOG_CUDA_BUNDLE_20260820_C`. It freezes `lambda_L2=0`, 660 registered jobs,
24 non-overlapping deterministic shards, one exact-best checkpoint per job,
and job-level restart behavior. Calibration and diagnostic smoke models are not
production models.

## Verify

From `/home/jovyan`:

```bash
python JOG_PRODUCTION_LAUNCHER_20260820_E/tools/verify_production_launcher_package.py \
  JOG_PRODUCTION_LAUNCHER_20260820_E \
  JOG_CUDA_BUNDLE_20260820_C
```

## Concurrency benchmark

Run this once on one T4 before selecting one or two workers per server. It
performs one single and one concurrent pair of three-epoch diagnostics, uses no
held-out test rows, and does not enter the production registry.

```python
import sys
sys.path.insert(0, "/home/jovyan/JOG_PRODUCTION_LAUNCHER_20260820_E/icepack-mlp")

from production_training.concurrency_benchmark import run_concurrency_benchmark

benchmark = run_concurrency_benchmark(
    bundle_root="/home/jovyan/JOG_CUDA_BUNDLE_20260820_C",
    output_root="/home/jovyan/JOG_CUDA_RUNS/concurrency_benchmark_20260820_a",
    cuda_environment="/home/jovyan/JOG_CUDA_RUNS/cuda_environment.json",
    runtime_override="/home/jovyan/JOG_CUDA_RUNS/cuda_runtime_override.json",
)
benchmark
```

Two workers are recommended only if all runs verify, repeated-job predictions
are exact, aggregate speedup is at least 1.65, and peak GPU memory is at most
90 percent.

## Production campaign

Do not run this until the benchmark result and server/shard assignment are
frozen. Call the same function again after interruption; it validates and skips
completed jobs and restarts only an incomplete current job.

```python
from production_training.campaign import run_production_campaign

summary = run_production_campaign(
    bundle_root="/home/jovyan/JOG_CUDA_BUNDLE_20260820_C",
    runs_root="/home/jovyan/JOG_CUDA_RUNS/production",
    cuda_environment="/home/jovyan/JOG_CUDA_RUNS/cuda_environment.json",
    runtime_override="/home/jovyan/JOG_CUDA_RUNS/cuda_runtime_override.json",
    shard_ids=[0],
    workers=1,
)
summary
```

Each server must receive explicit, non-overlapping shard IDs. A fresh heartbeat
prevents accidental duplicate claims on shared storage. Shards with stale or
completed state are resume-verified. Never manually copy a diagnostic model
into the production run directory.
