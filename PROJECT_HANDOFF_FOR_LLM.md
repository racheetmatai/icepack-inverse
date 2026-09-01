# JOG paper revision — handoff for another LLM

Last consolidated: 1 September 2026

This is the entry point for any LLM or researcher resuming the revision of the
Journal of Glaciology manuscript on learning a velocity-independent functional
relationship for the basal-friction control `C` in the Amundsen Sea sector.
Do not infer decisions from old notebooks or chat logs. The author-approved
decisions and verified manifests named below are the source of truth.

## Where the project lives

- Windows working root: `F:\Codex\JOG`
- Icepack Docker container: `xenodochial_cerf`
- Docker repository: `/home/firedrake/icepack/icepack-inverse`
- Docker Git remote: `git@github.com:racheetmatai/icepack-inverse.git`
- Windows CUDA/training source: `F:\Codex\JOG\icepack-mlp`
- Paper/LaTeX working directory: `F:\Codex\JOG\latex_revision`
- Accepted post-training workflow: `F:\Codex\JOG\production_workflow`
- Verified CUDA production results: `F:\Codex\JOG\cuda_results`
- CUDA transfer packages and provenance: `F:\Codex\JOG\cuda_transfer`
- Work-package audits: `F:\Codex\JOG\work_package_1` and
  `F:\Codex\JOG\work_package_2`

The Windows root is deliberately not a Git repository. The scientific Python,
Firedrake/Icepack, notebooks, and Docker-side workflow are versioned in the
Docker Git repository. Large result archives, model checkpoints, generated
fields, external datasets, manuscript assets, and audit records are preserved
in the dated `D:` backup described by its own `BACKUP_README.md`.

## First files to read, in order

1. `REVISION_PROJECT_MEMORY.md` — durable author-approved scientific and
   methodological decisions.
2. `REVISION_MASTER_CHECKLIST.md` — completed and remaining work.
3. `REVISION_REMAINING_WORK_ORDER.md` — current execution order.
4. `latex_revision/MANUSCRIPT_REVISION_OUTLINE_20260830.md` — results/discussion
   story and manuscript structure.
5. `latex_revision/FINAL_FIGURE_PLAN_20260830.md` — accepted main/appendix
   figure roles and style rules.
6. `production_workflow/final_figures_20260830_a/README.md` — generated figure
   files and provenance.
7. `cuda_results/PRODUCTION_TRANSFER_VERIFICATION_20260828.md` — independent
   verification of the 660-model archive.
8. `work_package_1/WORK_PACKAGE_1_REPORT.md` and
   `work_package_2/AUTHOR_DECISION_LOG.md` — data-pipeline fixes and experiment
   freeze.

## Scientific question and interpretation

The primary test is not whether an MLP reproduces the inversion-derived `C`.
The inverse problem is non-unique and the inverted `C` is one regularized,
model-dependent reference solution. The primary question is whether a
velocity-independent mapping from supported ice/geophysical predictors to `C`
produces an Icepack forward velocity that agrees with observed MEaSUREs surface
velocity in spatially held-out regions.

Consequences:

- Evaluate forward velocity against observed satellite velocity.
- Treat agreement with inversion-reference `C` as a secondary diagnostic.
- A different `C` with good velocity can reflect equifinality/non-uniqueness.
- A local `C` discrepancy need not cause a collocated velocity error because
  the forward response is spatially non-local.
- Do not call ensemble-member spread calibrated uncertainty. It is
  training-realization variability caused by deterministic changes in split,
  initialization, and shuffle seed.
- Do not use velocity, velocity direction, basal shear stress, or velocity
  errors as MLP predictors. Basal shear stress is internal to Icepack's sliding
  law.
- Do not frame every failure as out-of-distribution. Separate marginal support,
  joint/density support, target non-uniqueness, missing predictors, resolution,
  and model/physics limitations.

## Corrected data and inversion path

- Magnetic predictor: ADMAP2S EPSG:3031 NetCDF.
- Gravity predictor: AntGG2021 surface gravity disturbance NetCDF.
- The raster loader accepts coordinate-aware NetCDF and TIFF.
- Sample raster values at true cell centers.
- Velocity support requires finite paired `VX` and `VY`, valid source/domain,
  and grounded `phi > 0.1`; `ERRX/ERRY` are diagnostics, not a row mask for the
  unweighted inversion.
- The accidental Bouguer-anomaly filter is removed. Bouguer anomaly is not a
  revised predictor or selection variable.
- Driving stress is `rho_i g h |grad(s)|`, passed/stored in Icepack-native MPa.
- BedMachine v2 is retained. `source` and `errbed` are auxiliary diagnostics,
  not predictors or row filters.
- Snow accumulation is omitted from the revised data and predictors.
- Definitive whole-sector L-curve selection froze `reg_C = 0.02`; the fresh
  definitive inversion started from exact logarithmic `C = 0` and its archived
  outputs/manifests are the only revised inversion source.

## Frozen experiments

### Ten primary spatial tests

- Ten deterministic, non-overlapping 50 km by 50 km central test squares.
- A 40 km buffer on every side makes a 130 km by 130 km training-exclusion
  footprint.
- Central square is the independent primary evaluation region; buffer and full
  footprint are nested diagnostics.
- Ten squares are the independent spatial replicates.
- Each square/configuration has ten MLP members with distinct reproducible
  90/10 train/validation splits of the remaining eligible population.
- One common row population is finite for the union of all twelve predictors.

### Regional stress tests

- PIG holdout: train on the rest of the exhaustive grounded-sector partition;
  evaluate complete PIG. Configurations CFG02, CFG01, CFG03.
- Inter-catchment holdout: train on all three catchments; evaluate both
  inter-catchment corridors. Configurations CFG04, CFG05, CFG06.
- These are secondary partial-support transfer tests, not additional spatial
  replicates and not universal in-distribution demonstrations.

### Six feature configurations

1. CFG01 — all ice predictors.
2. CFG02 — historically selected ice subset.
3. CFG03 — all geophysical predictors.
4. CFG04 — historically selected geophysical subset.
5. CFG05 — selected ice plus selected geophysical predictors.
6. CFG06 — CFG05 plus the velocity-independent bed/surface gradient-alignment
   feature `cos_theta_bs`.

Exact predictor lists and units are frozen in
`work_package_2/AUTHOR_DECISION_LOG.md`. The selected subsets came from the
legacy dynamical `% explained` screening, not the validation-R2 heatmap. The
defective exhaustive legacy catchment heatmap is provenance only and must not
be reused as revised performance evidence.

## Training and completed production campaign

- Common architecture: ten hidden layers, 200 neurons each, batch
  normalization plus Swish/SiLU, linear output, Adam/MSE, batch size 1024,
  maximum 1500 epochs.
- Train-only `RobustScaler` objects for inputs and target `C`.
- Global L2 calibration tested `{0, 1e-6, 1e-5, 1e-4}` on a dedicated
  deterministic CFG06 70/20/10 split. Selected `lambda_L2 = 0` because every
  nonzero candidate worsened unpenalized validation data MSE.
- CUDA campaign: 600 square models plus 60 regional models = 660 MLPs.
- Production completion audit: 660/660 verified; 24/24 shards complete;
  historical failed attempts retained as provenance.
- Production audit manifest:
  `sha256-json-v1-592f9e3753030c2eadab0137c365a11c337a77d6a8fc325a05a46a405638387c`.
- Pangeo archive SHA256:
  `c19f999d62b42177a0d5b6d69573a647f7470d13c1aa2a9b2305f4cc82ebe6d2`.
- All 660 member `C` fields and 66 vertex-wise ensemble-median `C` fields were
  run through Icepack: 726 forward runs total.
- The headline ensemble field is the vertex-wise median of the ten predicted
  `C` fields, followed by a new nonlinear forward solve. It is not the median
  of member velocity metrics.

## Metrics and statistical hierarchy

- Primary metric: finite-element-area-weighted vector velocity RMSE against
  MEaSUREs, in m a^-1.
- Relative physical skill: `RMSE_ML / RMSE_uniform-C`; values below one improve
  on the frozen baseline.
- `P_exp` is retained only in archived provenance and is removed from the
  revised manuscript.
- Report square-level median/IQR and all ten square values. Ensemble members are
  not independent replicates.
- Prespecified paired square contrasts and sign tests use squares as the ten
  paired observations; exact Holm-adjusted results belong in the appendix.
- Predictor support is diagnostic and computed on held-out populations against
  their corresponding training pool. Do not remove unsupported held-out rows.

## Accepted manuscript story and figure plan

The narrative is: geography/design -> inversion reference and caveat -> input
support versus physical skill -> repeated quantitative outcomes -> spatial
square errors -> catchment-transfer behavior -> `C`/velocity diagnostic and
non-uniqueness/non-locality interpretation.

Main figures currently planned:

1. Study region and experimental geometry.
2. Definitive whole-sector inversion (observed speed, inversion `C`, velocity
   residual). The workflow flowchart is appendix material, not main Figure 2.
3. Predictor support and relative physical skill across ten squares and six
   configurations.
4. Quantitative performance: ten-square results plus the complete PIG stress
   test.
5. Spatial square error mosaics for CFG02 and CFG01 over complete footprints.
6. Complete PIG transfer diagnostic, planned as a 2x3 layout: signed speed bias
   on the top row and vector velocity error on the bottom row for CFG02, CFG01,
   and CFG03.
7. For CFG02 and CFG01, continuous triangulated signed `C` difference alongside
   normalized local velocity error over the ten held-out footprints.

All map figures use the model-domain/terminus outline and the same Antarctica
locator at top right. Multi-panel manuscript artwork is stored as separate,
title-free panels; LaTeX supplies subfigure letters and titles. Legends and
labels must remain readable at final journal size and must not cover data.

The current final-figure bundle manifest is:
`sha256-json-v1-2696f40bd12c60a67bfd6fef521b8e8773b08ca893d4a99b63a0489b850bdb57`.
The latest spatial update changed signed `C` maps from hexagonal scatter markers
to continuous deterministic mesh triangulation and added PIG and
inter-catchment transfer maps. The accepted 2x3 PIG signed-bias/vector-error
layout and its appendix support/prediction panels are the next artwork change;
they have not yet been implemented.

Appendix evidence includes workflow, mesh/L-curve, twelve predictor maps,
detailed support diagnostics, all six square mosaics, PIG supporting maps,
inter-catchment results, `C` distributions/diagnostics, ensemble variability,
BedMachine source/errbed, feature-selection provenance, and training QA.

## Immediate next work

1. Implement and visually review the accepted PIG 2x3 figure and supporting
   appendix panels.
2. Review every main and appendix figure with the author, one numbered figure
   at a time.
3. Freeze final artwork and regenerate the output-hash manifest.
4. Rewrite the manuscript around the accepted outline while preserving the
   original scientific motivation and integrating reviewer responses into the
   natural argument.
5. Build LaTeX, inspect every page, update tables/captions/references, and run a
   claim-to-evidence audit.
6. Draft the point-by-point reviewer response only after the revised text,
   figures, tables, and appendix locations are stable.

## Non-negotiable cautions

- Preserve legacy files for provenance; do not silently overwrite or relabel
  them as revised outputs.
- Never mix legacy datasets/scalers/models with revised rows or manifests.
- Do not relocate holdout squares after looking at outcomes.
- Do not tune architecture, L2, support thresholds, feature lists, or reporting
  rules against held-out performance.
- Do not present the 660 trained MLPs or their 660 forward runs as 660
  independent scientific tests.
- Do not call training-realization spread parameter uncertainty or predictive
  uncertainty without calibration.
- Do not use the local squared-error-improvement/P_exp maps in the manuscript.
- Every final figure must be generated from accepted manifests and hash
  verified before LaTeX inclusion.

