# Portable production training workflow

This directory is the accepted training path for the JOG revision. It consumes the frozen canonical-dataset and exact split-bundle manifests; it does not create a new split and it never makes held-out rows available to `fit`.

The implementation freezes the six approved feature configurations, 10×200 Dense–BatchNorm–SiLU hidden architecture, linear output, hidden-kernel-only L2 regularization, Adam at 0.001, batch size 1024, and the 1500-epoch callback policy. Both predictor and target `RobustScaler` objects are fit on the selected member's training rows only. Validation is the exact frozen member validation mask. Epoch permutations use the separate frozen shuffle seed.

Run one accepted job on the CUDA computer from this repository root:

```bash
python -m production_training.train \
  --dataset-dir /path/to/gate2_canonical_dataset_20260820_c \
  --split-bundle /path/to/gate2_split_manifests_20260820_a \
  --job-id SQ01_CFG01_M01 --lambda-l2 1e-5 \
  --output /new/empty/output/SQ01_CFG01_M01
python -m production_training.verify_run /new/empty/output/SQ01_CFG01_M01
```

`--max-epochs` may only lower the frozen cap and is intended for the later explicitly labelled CUDA smoke test. Production and L2-pilot runs omit it. CPU training is refused unless `--allow-cpu-training` is explicitly supplied for a bounded smoke test. `--prepare-only` verifies and prepares real data without importing TensorFlow or training a model.

Every completed run records the dataset and split identities, exact row-membership identifiers, feature list, seeds, scalers, environment, histories, best epoch, validation predictions, and hashes of its artifacts. Finite completed jobs are retained; held-out results never select or reject individual members.

The revised L2 calibration uses four `CFG06` fits on one dedicated deterministic 70/20/10 random split. Its population contains all common-eligible rows except the ten central 50 km square evaluation masks; all buffers, PIG, and both inter-catchment corridors remain included. Candidates are `{0, 1e-6, 1e-5, 1e-4}` with exactly matched rows and seeds. Selection uses only best-restored unpenalized validation `data_mse`; a nonzero value must improve zero by at least `1e-4`, otherwise zero is retained. The sealed 10% calibration test is accessed once only after selection.

Calibration launches use the exact `pilot_job_id` in the dedicated `l2_calibration/l2_pilot_registry.csv`; the CLI verifies that `--lambda-l2` equals the registered value. One selected L2 is frozen globally. Production launches use `job_id` from the main `splits/job_registry.csv`. The former 36-fit square/configuration pilot is superseded and must not be run.
