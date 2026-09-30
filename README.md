# Point-wise inference of basal friction in the Amundsen Sea sector

This repository contains the code and the fixed experimental design for the
Journal of Glaciology manuscript *Do ice and subglacial observables determine
basal friction? A point-wise test in the Amundsen Sea sector*.

The workflow tests whether local observable predictors can reproduce the
dimensionless basal-friction control `C` from a sector-wide Icepack inversion
and whether the predicted controls improve modeled velocity in spatially
withheld regions.

The paper calls the two predictor groups ice-product and subglacial-product.
Identifiers in the code such as `CFG04_best_geophysical` keep an earlier name
for the subglacial-product group, because they name files and keys in the
archived artifacts.

## Reproducibility levels

Two supported routes are provided:

1. **Reproduce the paper.** Download the archived analysis artifacts, verify
   their hashes, and regenerate the manuscript figures and analysis tables.
   This route does not repeat inversion, MLP training, or
   Icepack simulations.
2. **Recompute the experiment.** Starting from the scientific inputs, repeat
   the inversion and L-curve selection, build the dataset and
   spatial splits, train 660 MLPs, construct 66 vertex-wise median-`C` fields,
   run the 76 controlled Icepack simulations used for velocity comparisons,
   and rebuild the analyses.

The first route is intended for most readers. The second requires a Firedrake
environment, CUDA-capable training resources, and substantial runtime.

## Repository layout

- `production_workflow/`: inversion, dataset, prediction, Icepack simulation,
  evaluation, diagnostics, and figure-generation code.
- `production_workflow/controlled_replacement/`: construction, execution,
  evaluation, and integrity checks for the controlled replacement design.
- `icepack-mlp/production_training/`: reproducible CUDA MLP training code.
- `cuda_transfer_tools/`: bundle verification and restartable campaign tools.
- `configs/`: configuration and artifact metadata.
- `environments/`: Icepack and CUDA environment records.
- `patches/`: the documented patch applied to the pinned Icepack revision.
- `scripts/`: artifact verification and paper-reproduction entry
  points.
- `manuscript/`: the manuscript source and its figures.
- `tests/`: lightweight public-package tests.

Large scientific inputs, trained checkpoints, predicted controls, modeled
velocities, and analysis-ready outputs are stored in the associated Zenodo
record. They are not stored in Git.

## Quick paper reproduction

After downloading the Zenodo files into an empty artifact directory:

Use Python 3.10 in a separate environment for the tested plotting dependencies.

```bash
python -m pip install -r environments/paper_requirements.txt
python scripts/verify_artifacts.py --profile paper --artifact-dir /path/to/artifacts
python scripts/unpack_artifacts.py --profile paper --artifact-dir /path/to/artifacts
python scripts/reproduce_paper.py --artifact-dir /path/to/artifacts/unpacked
```

The manuscript figures are in `manuscript/figures`. To compare rendered outputs:

```bash
python scripts/compare_figure_rendering.py --reference manuscript/figures \
  --candidate reproduced_paper/figures --output reproduced_paper/comparison
```

The Zenodo DOI (10.5281/zenodo.22839669) is recorded in
`configs/artifacts.json`.

A few numbers quoted in the text come from short read-only checks in
`production_workflow/paper_checks/`, run on the same unpacked artifacts:

- `check_blend_bound.py`: the appendix bound "at most 0.004" on how the edge
  of the replaced region changes the C RMSE;
- `check_speed_classes.py`: the SQ06 speed-class shares (91% and 4%);
- `training_density_sensitivity.py`: the training-density check with the
  100th/500th neighbour and a 40 km separation;
- `regional_validation_c.py`: the median validation C RMSE / R_C^2 of the two
  regional tests (reads the unpacked training-run archive 02).

## Full recomputation

The full workflow is described in `FULL_RECOMPUTATION.md`. In outline:

1. build the pinned Icepack environment and apply `patches/icepack.patch`;
2. verify the inputs and run the inversion/L-curve workflow;
3. export and verify the dataset and exact spatial splits;
4. train the registered CUDA jobs and verify every run;
5. construct the 66 median controls and run the corresponding Icepack cases;
6. evaluate the saved fields and regenerate the paper outputs.

Every stage is restartable and verifies input identities before reusing an
existing result.

The manuscript scenario replaces the inversion-reference control only inside
each withheld geography and evaluates modeled velocity against the original
paired MEaSUREs raster components. The earlier sector-wide replacement is
retained as a separately named sensitivity analysis.

## External inputs

BedMachine Antarctica and MEaSUREs velocity are obtained from NSIDC using an
Earthdata account. Their expected SHA-256 hashes are stored in the
configuration. The remaining exact input rasters, mesh, regional boundaries,
and design files are included in the Zenodo input archive.

## Tests

The package tests run in the paper environment:

```bash
python -m pytest tests
```

The workflow tests need the Firedrake environment:

```bash
python -m pytest production_workflow/tests
```

`pytest.ini` sets the import paths and turns off pytest's output capture,
which one L-curve test needs because it reads the solver's own log.

## License

This repository is licensed GPL-3.0-or-later (see `LICENSE`), matching the
license of Icepack, which it patches (`patches/icepack.patch`) and directly
builds on. This applies to the code in this repository; it does not set the
license of the separately staged Zenodo data archives.

## Citation

Please cite the manuscript and the associated Zenodo record. Repository
citation metadata are provided in `CITATION.cff`.
