# Training representation and inversion-control error

## Question

This extension asks whether predictor combinations that are sparse among the actual MLP training rows have larger errors in the predicted inversion control (C). It uses exactly the rows and representation percentiles from the completed parent analysis: SQ01--SQ10 central 50 km squares for CFG02 and CFG04, plus the complete eligible PIG holdout for CFG02.

This is descriptive. Agreement with the inversion-reference (C) measures reproduction of one regularized, model-dependent inverse solution. Forward velocity remains the primary physical test.

## Authoritative alignment

The saved vertex-wise median ensemble control used in each forward simulation was loaded from `gate3_full_mesh_ensemble_predictions_20260828_a`. It was not recomputed from observation-grid member predictions. Each exact saved CG2 median field and the adopted reference field were interpolated to the canonical 450 m observation mesh using the same `icepack.interpolate(field, Delta)` operation used by the canonical dataset exporter. Stable row IDs then joined these values to the existing representation and velocity diagnostics.

The interpolated reference C agrees with the canonical exported target to a maximum absolute difference of 3.55e-15. All 474,327 requested case rows aligned one-to-one. Observation-grid C RMSEs agree with the independent exact finite-element-area diagnostics to within 2.37%; the small difference is expected because the two diagnostics use different discretizations.

## Findings

### Ten square holdouts

Sparse representation does not produce a general increase in (C) error across the ten squares.

- For CFG02, the median square C RMSE is 0.299, 0.297 and 0.301 in the `<=50`, `50-95` and `>95` percentile categories. Seven squares increase monotonically, but the changes are generally small and SQ05 and SQ10 show the opposite pattern.
- For CFG04, the corresponding medians are 0.492, 0.499 and 0.444. Only one square increases monotonically; several have smaller C error in the sparsest category.
- Point-level associations between representation percentile and absolute C error are weak and inconsistent. Across squares, the median Spearman correlation is 0.069 for CFG02 and -0.052 for CFG04.
- The result persists within observed-speed classes. Below 100 m a-1, the median square correlation is 0.080 for CFG02 and -0.113 for CFG04. Only five squares contribute to 100--500 m a-1 and only one contributes to 500--1000 m a-1, so those classes cannot establish a repeated spatial pattern.

High-support failures are not explained by C error increasing in the sparse category. For example, within the jointly and marginally supported population, SQ02 CFG04 has a velocity RMSE ratio of 1.50 in the `<=50` category and 1.28 in the `>95` category while C RMSE decreases from 0.608 to 0.448. SQ06 CFG02 remains near or above the uniform-C velocity error across categories although its C RMSE changes only from 0.493 to 0.525.

The C and velocity diagnostics also separate in individual cases. In SQ05 CFG02, C RMSE decreases from 0.416 in the best-represented category to 0.337 in the sparsest category, while the velocity RMSE ratio increases from 0.36 to 0.80. SQ01 CFG04 has the largest overall C RMSE (1.205) but still slightly improves on uniform C in velocity (ratio 0.938). Conversely, SQ06 CFG02 has a moderate C RMSE (0.504) but does not improve the velocity baseline (ratio 1.073).

### PIG

PIG shows a clearer aggregate increase: C RMSE rises from 0.742 to 0.877 to 0.898 across the three representation categories, while the velocity RMSE ratio rises from 0.800 to 0.882 to 1.009. However, the row-level association between representation percentile and absolute C error is only 0.031.

Observed speed is the stronger stratification in PIG. C RMSE is 0.502 below 100 m a-1, 1.019 at 100--500 m a-1, 1.681 at 500--1000 m a-1 and 1.577 at or above 1000 m a-1. Within those speed classes, the representation--absolute-C-error correlations are -0.068, 0.171, -0.085 and 0.067. Thus, fast PIG flow coincides with both larger C error and relatively sparse representation, but sparse representation alone does not order C error after speed is separated.

Absolute C error and absolute ML velocity error are often positively associated locally, including PIG (Spearman correlation 0.675). That does not imply that C error determines comparative velocity skill: in PIG, the association between absolute C error and the local ML-minus-uniform velocity error is -0.100. Across the squares, both associations vary widely. This is consistent with the model-dependent nature of the inverse target and nonlocal ice-flow response.

## Retraining assessment

This diagnostic does not justify retraining by itself. The square experiments show no general representation--C-error relationship, CFG04 often behaves in the opposite direction, and the apparent PIG pattern weakens or reverses within fixed speed classes. A controlled reweighting or resampling experiment could test a narrower question--whether increasing exposure to sparsely represented fast-flow predictor combinations improves their C fit and downstream velocity--but the present evidence does not show that this would resolve the observed failures. The case for retraining is therefore unchanged rather than strengthened.

## Files

- `export_observation_grid_controls.py`: exact interpolation-only export of saved median controls to the canonical observation mesh.
- `analyze_c_representation.py`: reproducible analysis and figure generation.
- `observation_grid_c.csv.gz`: interpolated reference and median-predicted C values for the requested rows.
- `point_c_diagnostics.csv.gz`: row-level representation, C, velocity, speed and support diagnostics.
- `c_overall_metrics.csv`: whole-case C and velocity summaries.
- `c_category_metrics.csv`: summaries by representation category.
- `c_speed_stratified_metrics.csv`: category summaries within observed-speed classes.
- `c_speed_overall_metrics.csv`: speed-class summaries without representation subdivision.
- `c_support_stratified_metrics.csv`: summaries separated by whether frozen marginal and joint support are both satisfied.
- `c_associations.csv`: representation--C associations.
- `c_velocity_associations.csv`: C-error--velocity-error associations.
- `c_authoritative_metric_checks.csv`: comparison with exact finite-element C diagnostics.
- `figure_C_C_rmse_by_representation.pdf` and `.png`: principal figure.
- `verification.json` and `manifest.json`: checks, identities and hashes.
- `LATEX_SUGGESTIONS.md`: optional manuscript and reviewer-response wording; not inserted anywhere.

## Limitations

The millions of nearby rows are not independent replicates. The ten squares are the geographic replicates and PIG is one secondary test. Representation percentile is a neighbor-distance rank relative to actual training rows, not a density, probability or effective sample size. Association does not establish causation. The analysis does not treat inversion-reference (C) as physical truth and does not assume that local C agreement must map locally to velocity agreement.
