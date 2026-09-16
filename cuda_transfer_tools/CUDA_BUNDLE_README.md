# JOG revision CUDA bundle

This is the immutable input package for Gate 3. Do not edit files inside the bundle. Copy the entire directory to the CUDA computer, then run from its parent directory:

```bash
python JOG_CUDA_BUNDLE_20260820_C/tools/verify_cuda_bundle.py JOG_CUDA_BUNDLE_20260820_C
python JOG_CUDA_BUNDLE_20260820_C/tools/capture_cuda_environment.py \
  --output JOG_CUDA_RUNS/cuda_environment.json
```

Both commands must pass before launching the L2 pilot. Environment output is deliberately outside the immutable bundle.

Run the packaged unit tests:

```bash
cd JOG_CUDA_BUNDLE_20260820_C/icepack-mlp
python -m unittest discover -s tests -v
```

The next scientific computation is the frozen four-fit `CFG06` calibration in `l2_calibration/l2_pilot_registry.csv`. It uses one deterministic 70/20/10 random split of common-eligible rows after excluding only the ten central 50 km square masks. PIG, both inter-catchment corridors, and all buffers remain included. The 10% calibration test mask is sealed during fitting and selection.

Run all four candidates sequentially and freeze L2:

```bash
# Run from the directory containing JOG_CUDA_BUNDLE_20260820_C.
python JOG_CUDA_BUNDLE_20260820_C/tools/run_l2_calibration.py \
  --bundle-root JOG_CUDA_BUNDLE_20260820_C \
  --runs-root JOG_CUDA_RUNS/l2_calibration \
  --cuda-environment JOG_CUDA_RUNS/cuda_environment.json
```

The launcher verifies/resumes completed jobs, runs one GPU training process at a time, and writes `global_l2_selection.json`. Selection uses only best-restored unpenalized validation `data_mse`. A nonzero value must improve `L2=0` by at least `1e-4`; otherwise zero is retained. Exact ties go to the smaller value.

Only after selection is written, evaluate the sealed 10% calibration test once:

```bash
cd JOG_CUDA_BUNDLE_20260820_C/icepack-mlp
python -m production_training.evaluate_calibration_test \
  --dataset-dir ../dataset --calibration-bundle ../l2_calibration \
  --selection ../../JOG_CUDA_RUNS/l2_calibration/global_l2_selection.json \
  --runs-root ../../JOG_CUDA_RUNS/l2_calibration \
  --output ../../JOG_CUDA_RUNS/l2_calibration/calibration_test
```

This four-fit protocol supersedes the earlier 36-fit square/configuration pilot. No broad architecture optimization is performed. The chosen single L2 value is subsequently frozen for all 660 production jobs.

Do not use `--max-epochs`, `--allow-cpu-training`, `--prepare-only`, or `--skip-full-hash-check` for pilot or production jobs.
