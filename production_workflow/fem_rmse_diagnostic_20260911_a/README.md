# Observation-grid and FEM velocity RMSE diagnostic

This saved-field-only diagnostic compares the manuscript's velocity RMSE on
the retained regular 450 m MEaSUREs rows with an area-integrated calculation
on the unchanged production finite-element mesh.  It includes all six median
controls for SQ01--SQ10 and CFG02 for the complete PIG holdout.  It did not run
an inversion, train a model, or solve the ice-flow equations.

## FEM definition and coverage

MEaSUREs observations are stored for inversion on a vertex-only observation
mesh, which has no cells and cannot be integrated over area.  The FEM
diagnostic therefore uses the existing CG2 MEaSUREs velocity fields on the
production mesh.  It retains whole production triangles when the triangle
barycentre (1) lies in the test geography, (2) maps to a 450 m pixel in the
frozen common eligible observation population, and (3) has finite observed
velocity at every local CG2 node.  No missing observation inside a retained
triangle is filled.

The squared CG2 vector residual is integrated by Firedrake over this DG0-masked
triangle population and divided by its assembled area.  Triangle selection at
boundaries is a cell-barycentre approximation, so this is not exactly the same
population as the retained observation rows.  Across the ten squares the
median FEM-to-row-pixel area ratio is 0.995; the largest departure from one is
4.40%.  PIG retains 92.6% of the area represented by its eligible 450 m rows.

The mesh representation sampled back at eligible observation rows is finite
for 99.488% of rows.  On those rows its RMSE against the original observations
is 1.92e-14 m a-1.  The remaining rows are near missing-data interpolation and
their intersecting FEM triangles are excluded.

## Result

The evaluation choice does not change the scientific conclusions.

- All 61 observation-grid RMSEs reproduce the corrected authoritative results.
- For square cases, the median absolute change in ML RMSE is 0.61%; the largest
  is 9.14% (SQ05 CFG04: 134.11 to 121.85 m a-1).
- The ML control improves on uniform C in exactly the same 47 of 60 square
  configuration cases.  Every configuration retains its original count of
  successful squares.
- The ordering by median square RMSE remains CFG02, CFG01, CFG04, CFG05,
  CFG06, CFG03.  Only small within-square swaps occur for SQ01 and SQ02.
- PIG CFG02 changes from 355.67 to 366.30 m a-1.  Its relative RMSE changes
  from 0.920 to 0.932, so it still improves on uniform C (8.03% versus 6.79%).
- Degree-4 and degree-6 quadrature differ by at most 5.7e-13 m a-1.  Zero and
  constant 3--4 vector-residual tests return exactly 0 and 5 m a-1.

The current observation-grid metric should remain the manuscript metric.  It
uses the complete declared observation population directly and avoids the
additional triangle-boundary and finite-interpolation exclusions required by
the FEM diagnostic.  FEM integration is a useful robustness check, but it is
not a strictly superior evaluation of the identical population.

## Files

- `compare_observation_grid_fem_rmse.py`: restartable calculation.
- `rmse_comparison.csv`: all 61 case-level results.
- `manifest.json`: input identities, method, software and summary checks.
- `verify_results.py` and `verification.json`: independent artifact checks.
