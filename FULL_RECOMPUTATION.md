# Full recomputation

The complete experiment has three computational environments.

## 1. Icepack inversion and modeled velocity

Build `environments/Dockerfile.icepack`. It pins the Firedrake base image and
Icepack commit used for the study and applies `patches/icepack.patch`.

Extract the input archive at the repository root, then obtain the two NSIDC
files identified in `configs/artifacts.json`. Verify all inputs before running:

```bash
bash production_workflow/run_production.sh preflight \
  --repo-root "$PWD" --output-root "$PWD/production_runs" \
  --run-id preflight --level full
```

The remaining inversion and L-curve commands, convergence rules, and resume
behavior are documented in `production_workflow/README.md`. The accepted
regularization coefficient is recorded in the archived selection bundle; a
new calculation should reproduce the frozen selection procedure rather than
hard-code its answer.

## 2. CUDA training

Create the CUDA environment described by
`environments/cuda_environment.yml`. The production registry contains 660
jobs: 66 experiment/configuration pairs with ten reproducible training
realizations each.

Training commands and checkpoint verification are documented in
`icepack-mlp/README_PRODUCTION_TRAINING.md`. The campaign runner is restartable:
completed, verified jobs are skipped, while incomplete jobs are retried
without overwriting previous evidence.

## 3. Predictions, modeled velocity, and evaluation

Use `production_workflow/full_mesh_ensemble_predictions.py` to construct each
vertex-wise median control. Use `production_workflow/forward_solve_campaign.py`
for the 66 primary Icepack runs and
`production_workflow/uniform_baseline_campaign.py` for the spatially uniform
controls. Evaluate the saved fields with the corrected row-ID-aligned
`production_workflow/evaluate_forward_campaign.py`.

The public artifact manifest identifies the exact accepted outputs for each
stage. Verification must pass before a downstream stage is run.

## Expected scale

- 1,530,992 common eligible observation rows.
- 660 MLP training jobs.
- 66 vertex-wise median controls.
- 66 primary modeled-velocity evaluations.
- Ten central-square experiments are the independent geographic replicates;
  the regional experiments are secondary tests.

The exact elapsed time depends on the GPU allocation and Icepack solver
performance. The original CUDA campaign was distributed across multiple
Tesla T4 workers.

