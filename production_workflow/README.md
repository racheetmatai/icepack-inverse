# Amundsen production workflow

This directory is the controlled, non-notebook entry point for the revised
whole-sector Icepack workflow. It reads the frozen mesh and canonical inputs;
it never regenerates the mesh. The input preflight, zero-control forward smoke
test, independently initialized L-curve points, and resumable L-curve study
controller all reuse one frozen configuration and provenance contract.

## Frozen scientific policy

- Mesh: `amundsen.msh`, read with `read_mesh=True`; CG2 fields.
- Boundary tags: Dirichlet `[1,3,5,6,7,8,9,10,11]`; stress/calving fronts
  `[2,4]`.
- Velocity observations: finite paired `VX/VY`, `SOURCE > 0`, true 450 m
  pixel centers, and `locate_cell(...) is not None`. `ERRX/ERRY` are
  diagnostics for the unweighted inversion.
- Updated geophysics: ADMAP2S `z` and AntGG2021 surface `grav_dist` NetCDF
  fields in EPSG:3031.
- Bouguer anomaly and snow accumulation are omitted. Bed class and BedMachine
  `source/errbed` are auxiliary only and never affect eligibility.
- Exact predictor order:
  `s,h,mag_s,mag_h,driving_stress,surface_air_temp,b,mag_b,heatflux,gravity_disturbance,mag_anomaly,cos_theta_bs`.
- Driving stress is Icepack-native MPa. `cos_theta_bs` uses the frozen
  surface/bed slope floors and is finite and clipped to `[-1,1]`.

Centered linear MEaSUREs sampling is unavailable at 65 of 35,797 full-mesh
DOFs. These are not inversion observations. To give the nonlinear solver a
finite initial/Dirichlet field, only those 65 solver DOFs are filled from the
nearest valid centered MEaSUREs pixel (11 are Dirichlet DOFs). This does not
alter the observation mesh, unweighted loss, CSV/training population, or
evaluation population; the manifest records the fill count and distances.

## Running in Docker

The launcher activates Firedrake so MPI compilers and PETSc are available:

```bash
cd /home/firedrake/icepack/icepack-inverse
bash production_workflow/run_production.sh preflight \
  --repo-root /home/firedrake/icepack/icepack-inverse \
  --output-root /home/firedrake/icepack/icepack-inverse/production_runs \
  --run-id <immutable-run-id> \
  --level full
```

The output directory is immutable: the entry point refuses to overwrite an
existing run ID. A full run hashes all scientific inputs and frozen design
artifacts, constructs the Firedrake fields without optimization, and checks
mesh/boundary policy, raster metadata, centered coordinates, observation
support, predictor equality, stable row IDs, square masks, region partition,
and auxiliary-field linkage.

Before optimization, run the whole-sector model once at exactly `C=theta=0`.
Here `C=0` is the zero logarithmic control, corresponding to the frozen
constant friction scale `c0=0.01`; it is not zero physical basal drag:

```bash
bash production_workflow/run_production.sh forward-smoke \
  --repo-root /home/firedrake/icepack/icepack-inverse \
  --output-root /home/firedrake/icepack/icepack-inverse/production_runs \
  --run-id <immutable-forward-smoke-id>
```

The smoke test requires finite modeled velocity and observation misfit, zero
gradient roughness and weighted penalty, and exact equality of objective and
misfit. It performs no optimization.

After that gate passes, start the frozen adaptive-v2 L-curve study serially.
The approved base grid is `reg_C={0.01,0.02,0.05,0.1,0.2}`. The current study
was launched on 14 August 2026 as `gate1_lcurve_v2_20260814_a`, using verified
source-matched smoke `gate1_forward_smoke_v2_20260814_a`; substitute these IDs
when resuming that exact run:

```bash
bash production_workflow/run_production.sh lcurve \
  --repo-root /home/firedrake/icepack/icepack-inverse \
  --output-root /home/firedrake/icepack/icepack-inverse/production_runs \
  --forward-smoke-dir /home/firedrake/icepack/icepack-inverse/production_runs/<immutable-forward-smoke-id> \
  --run-id <immutable-lcurve-study-id>
```

If the container or host is interrupted, rerun the same command with
`--resume` (the smoke directory is then recovered from the saved contract).
The L-curve refuses to start unless the smoke manifest passed under exactly the
same config, scientific source hashes, input/design hashes, Docker image ID,
and stable runtime identity. A point lacking a completed manifest is preserved
and retried under a new immutable attempt directory; a verified completed
point is never silently rerun or overwritten. Resume also refuses to duplicate
a still-active point or orchestrator process.

Every newly introduced base, extension, or midpoint point is a separate
process and starts from exact `C=0`; confirmation blocks continue the relevant
same-`reg_C` saved controls. Optimization is run in 50-iteration blocks, with a
minimum of four blocks and a maximum of six ordinary blocks. A point may stop
only after both misfit `E` and unweighted
roughness `R` change by no more than 0.5% in each of two consecutive block
comparisons and its reduced-gradient norm is at most the `1e-3` safety ceiling.
Exposed ROL state and the native termination record must agree. Failed,
nonfinite, unsafe-gradient, or unstabilized points remain in the record and are
excluded from L-curve geometry.

Selection uses the preregistered maximum Menger curvature after independent
min-max normalization of `log10(E)` and `log10(R)`. A boundary-adjacent
provisional corner triggers exactly one extension: `reg_C=0.005` on the low
side or `reg_C=0.5` on the high side. If the largest-to-second-largest eligible
curvature ratio is below `1.25`, run one geometric-midpoint refinement round
around the leading candidate. Exact curvature ties favor the smaller `reg_C`.

Finally, give the selected point and its immediate evaluated neighbors one
additional 50-iteration confirmation block each and recompute the selection.
If selection changes or a confirmation delta fails the frozen stability
tolerance, confirm the newly selected/current triple once more; permit at most
two confirmation rounds total. An unresolved second round prevents automatic
selection. The verified final selected point is designated as the definitive
inversion without a redundant fresh inversion.

The earlier study `gate1_lcurve_20260813_a` was deliberately stopped at
`2026-08-14T22:03:37Z` and is preserved but superseded. Its completed
`reg_C=0.005` point is a historical diagnostic excluded from v2, and its
interrupted `reg_C=0.01` point is incomplete and cannot be reused. Do not
resume or modify that study. See `L_CURVE_RUN_STATUS.md` for the frozen v2
decision and current launch status.

Run all focused tests with:

```bash
source /home/firedrake/firedrake/bin/activate
cd /home/firedrake/icepack/icepack-inverse
python -m unittest discover -s production_workflow/tests -p 'test_*.py' -v
```

## Outputs

Each preflight run contains:

- `resolved_config.json`
- `preflight_manifest.json`
- `checks.csv`
- `input_metadata.json`
- `field_summary.csv`
- `environment.txt`
- `source_snapshot/`
- `logs/preflight.log`

No L-curve point, inversion, dataset export, or ML training is performed by
the `preflight` command.

The accepted revision training implementation lives at
`icepack-mlp/production_training`, with launch instructions in
`icepack-mlp/README_PRODUCTION_TRAINING.md`. It consumes this workflow's frozen
canonical dataset and split-bundle identities; historical notebook training is
formally superseded. Gate 2 data preparation has passed, while TensorFlow/GPU
execution remains reserved for the mandatory CUDA smoke gate.

Each L-curve point contains its own input preflight, native solver log,
attempt/status table, model-state DOF arrays with exact mesh hash, complete
objective decomposition, environment/source hashes, and self-hashed manifest.
The parent study contains the immutable contract and source snapshot,
incremental resume state, every point reference, curvature evidence,
normalization bounds, diagnostics, appendix plot, and
`definitive_inversion.json`.

## Canonical dataset export

The accepted revised dataset is `gate2_canonical_dataset_20260820_c`, generated
from the verified adopted endpoint at `reg_C=0.01414213562`. It retains all
1,622,598 selected observations and marks the single common population of
1,530,992 grounded rows finite for the union of all twelve predictors. Its
training target is logarithmic control `reference_log_C`; observed and inversion
velocities are evaluation-only. Row-level BedMachine `source`/`errbed`, bed
class, MEaSUREs diagnostics, exact square masks, and direct frozen-mesh regional
labels are auxiliary.

Verify an export with:

```bash
source /home/firedrake/firedrake/bin/activate
cd /home/firedrake/icepack/icepack-inverse
python -B production_workflow/tools/verify_canonical_dataset.py \
  production_runs/gate2_canonical_dataset_20260820_c
```

The accepted manifest ID is
`sha256-json-v1-496a391df29fc4d64ba1b134fc8e12fd808b2bb1194935e60981d980767dfd8e`.

## Held-out distribution diagnostics

The accepted descriptive bundle is
`gate2_distribution_diagnostics_20260820_c`. It reads only the accepted
canonical dataset and frozen 5 km support evidence. It cannot move locations,
change masks/features, or remove unsupported rows. It reports all four
point-support categories and treats inversion-reference `C` as a separate
descriptive diagnostic.

Run and verify with:

```bash
source /home/firedrake/firedrake/bin/activate
cd /home/firedrake/icepack/icepack-inverse
python production_workflow/describe_heldout_distributions.py \
  --dataset-dir production_runs/gate2_canonical_dataset_20260820_c \
  --reference-grid production_workflow/frozen_design/amundsen_input_support_grid_5km.npz \
  --partition production_workflow/frozen_design/five_region_partition_5km.npz \
  --selected-squares production_workflow/frozen_design/selected_squares.csv \
  --support-evidence production_workflow/frozen_design/five_region_partition_and_support.json \
  --output production_runs/<immutable-diagnostic-id>
python production_workflow/tools/verify_heldout_distributions.py \
  --bundle production_runs/<immutable-diagnostic-id> \
  --dataset-dir production_runs/gate2_canonical_dataset_20260820_c
```

Accepted manifest ID:
`sha256-json-v1-a27e564c9b5456ae52fbbe917e5aa6dbe30a9e24591f4d573d69292a988d3bca`.

## Exact split manifests

The accepted row-partition bundle is `gate2_split_manifests_20260820_a`.
It stores the sorted stable eligible row IDs once and uses lossless bit-packed
masks for every held-out/development population and all 120 member
training/validation splits. This is an exact row-ID representation, not a
probabilistic index or regenerated split. Matched configurations reference one
unchanged experiment/member split.

Run and verify with:

```bash
source /home/firedrake/firedrake/bin/activate
cd /home/firedrake/icepack/icepack-inverse
python production_workflow/create_split_manifests.py \
  --dataset-dir production_runs/gate2_canonical_dataset_20260820_c \
  --output production_runs/<immutable-split-bundle-id>
python production_workflow/tools/verify_split_manifests.py \
  --bundle production_runs/<immutable-split-bundle-id> \
  --dataset-dir production_runs/gate2_canonical_dataset_20260820_c
```

Accepted manifest ID:
`sha256-json-v1-b838631dfde2849f84ee2165749fe9ce44c01e71513e53fb66afe1d8785281b8`.
