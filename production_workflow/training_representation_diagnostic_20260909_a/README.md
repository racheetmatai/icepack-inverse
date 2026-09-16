# Training-data representation diagnostic

## Question

This analysis tests whether held-out predictor combinations that are sparsely represented among the actual MLP training rows tend to have larger forward-velocity errors. It covers the central 50 km regions of SQ01--SQ10 for CFG02 and CFG04, and the complete eligible PIG holdout for CFG02. It is descriptive and does not identify a causal mechanism.

## Authoritative inputs

- Canonical dataset: `gate2_canonical_dataset_20260820_c`, dataset manifest `sha256-json-v1-496a391df29fc4d64ba1b134fc8e12fd808b2bb1194935e60981d980767dfd8e`.
- Exact member training rows: `gate2_split_manifests_20260820_a`. Each calculation uses the corresponding member's frozen training mask.
- Predictor configurations: CFG02 = `s`, `h`, `mag_s`, `mag_h`, and `surface_air_temp`; CFG04 = `b`, `mag_b`, and `heatflux`.
- Joint-support transform: rank-Gaussian quantile transformation followed by whitened PCA retaining at least 99% of the variance. It was fitted to the frozen eligible 5 km sector reference grid exactly as in `describe_heldout_distributions.py`. This is not the MLP RobustScaler.
- Frozen support categories: `gate2_distribution_diagnostics_20260820_c`.
- Median-control forward velocities, observed velocities, and uniform-C comparisons: `gate4_forward_evaluation_20260829_a`.

The input SHA-256 hashes and software versions are recorded in `manifest.json`.

## Method

For each held-out row and ensemble member, the script calculated the distance to the 20th-nearest row in that member's actual training set in the frozen transformed predictor space. Training rows were not thinned, balanced, deduplicated, or given an additional spatial exclusion.

For each member, 10,000 training rows were selected uniformly with seed 20260909. Their self-excluded 20th-neighbor distances provided the training reference. A held-out distance was converted to a midrank empirical percentile of that reference, with ties receiving half weight. Each held-out row was summarized by the median percentile across the ten members. High percentiles therefore mean sparse representation relative to combinations encountered during training; they are not probabilities, density estimates, or independent sample sizes.

All central-square rows and all 227,685 eligible PIG rows were evaluated. Equal-area row means reproduce the established observation-grid weighting. Category RMSEs were calculated from mean squared vector errors before taking the square root.

The frozen support archive stores positions in the original canonical eligible-row order, while the split and forward tables are sorted by stable row ID. Support categories were therefore joined by stable row ID. This reproduces the frozen square support fractions to numerical precision.

## Main findings

### Where representation is sparse

The largest fractions above the training-reference 95th percentile are:

- CFG02: SQ10 20.8%, PIG 19.2%, SQ08 17.7%, SQ09 17.0%, and SQ01 11.1%.
- CFG04: SQ08 35.8%, SQ04 29.1%, SQ01 23.7%, SQ09 13.9%, and SQ10 11.7%.

No case has a material concentration at zero 20th-neighbor distance.

### Relation to forward-velocity performance

Across the ten squares, overall performance is not ordered by representation. The Spearman correlations between each square's median representation percentile and its ML/uniform-C RMSE ratio are -0.006 for CFG02 and -0.030 for CFG04. The corresponding correlations using the fraction above the 95th percentile are -0.176 and -0.164. These are descriptive values for ten geographic replicates; no row-level significance test was used.

The category summaries contain one partial pattern and several clear exceptions. For CFG02, the median RMSE ratio across squares increases from 0.405 to 0.536 to 0.733 across the <=50th, 50th--95th, and >95th percentile categories. Individual squares do not follow a common monotonic pattern. CFG04 is essentially flat across those categories, with median ratios of 0.901, 0.899, and 0.899.

Sparse representation is neither necessary nor sufficient for failure. CFG02 fails to improve on uniform C in SQ06 even though only 1.6% of the square is above the 95th percentile, while SQ10 has 20.8% above the 95th percentile and a strong overall improvement (RMSE ratio 0.205). For CFG04, SQ02, SQ06, and SQ07 fail with only 7.8%, 3.0%, and 7.8% above the 95th percentile, whereas SQ04 improves strongly despite 29.1% above the 95th percentile.

The same conclusion holds within the frozen `both supported` population. Both-support coverage is 93.3--99.3% for the square/configuration cases. The failing cases remain failures inside that population, including SQ06 CFG02 (ratio 1.074), SQ09 CFG02 (1.140), SQ02 CFG04 (1.479), SQ06 CFG04 (1.037), and SQ07 CFG04 (1.209). Training representation therefore does not resolve the previously identified high-support failures.

### Observed-speed classes and PIG

Within-square associations between representation percentile and local ML error vary in sign. Across squares, their median Spearman correlation is 0.043 for CFG02 and -0.009 for CFG04 below 100 m a-1, and 0.017 and -0.118 from 100 to 500 m a-1. Only one square contributes to the 500--1000 m a-1 class, so it cannot establish a repeated pattern. No square contains an eligible >=1000 m a-1 population in this table.

PIG adds a catchment-scale example. Its median percentile is 79.0 and 19.2% of its eligible area lies above the 95th percentile. The CFG02 RMSE ratio changes from 0.800 to 0.882 to 1.009 across the three representation categories. The loss of relative skill in the sparse tail is concentrated in faster flow: the ratios above 500 m a-1 increase with sparser representation, reaching 1.038 in the >=1000 m a-1, >95th-percentile category. The pattern is absent below 500 m a-1, where category ratios are flat or non-monotonic. The point-level PIG association is also weak (Spearman rho = -0.041 for ML error and 0.020 for ML-minus-uniform local error).

PIG therefore shows that sparse training representation can accompany poor relative performance in fast flow, but the ten-square comparison and the PIG speed classes do not support a general representation--error relationship. Predictor frequency, observed speed, nonlocal stress transmission, and the model-dependent inversion target remain entangled in this diagnostic.

## Files

- `point_diagnostics.csv.gz`: row-level neighbor, velocity-error, speed, and support results.
- `representation_summary.csv`: percentile-distribution summaries.
- `category_metrics.csv`: requested representation-category metrics.
- `speed_stratified_metrics.csv`: category metrics within the fixed speed classes.
- `associations.csv`: point-level descriptive Spearman associations.
- `square_level_associations.csv`: associations across the ten geographic replicates.
- `support_stratified_metrics.csv`: results inside and outside the frozen both-support category.
- `case_overall_metrics.csv`: one overall row per experiment/configuration.
- `sampled_row_ids.csv.gz`: held-out IDs and training-reference IDs. PIG held-out rows are complete, not sampled.
- `benchmark.json`, `verification.json`, and `manifest.json`: timing, correctness, identity, and provenance records.
- `figure_A_training_representation_percentiles.*` and `figure_B_rmse_ratio_by_representation.*`: the two principal figures.
- `LATEX_SUGGESTIONS.md`: optional wording; the manuscript was not edited.

## Limitations

The analysis describes representation in the predictor space used by the support diagnostic. It is not an effective-sample-size calculation, a calibrated density estimate, or a causal test. Correlated and geographically adjacent rows were retained because they contributed to MLP training. Median-ensemble velocity responds nonlocally to the predicted control, so a row-level neighbor distance cannot isolate a local physical cause of error. The ten squares are the independent geographic replicates; PIG is one secondary catchment test.
