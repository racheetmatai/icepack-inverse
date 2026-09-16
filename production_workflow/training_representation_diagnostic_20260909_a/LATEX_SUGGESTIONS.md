# Optional manuscript wording

The following text was not inserted into the manuscript.

## Concise appendix text

```latex
We tested whether poor forward performance was associated with sparse
representation of held-out predictor combinations in the actual MLP training
rows. For each held-out row, we calculated the distance to the twentieth
nearest training row in the rank-transformed, whitened principal-component
space used for the joint-support diagnostic. The calculation was repeated for
each ensemble member using its exact training rows. Distances were expressed
as percentiles of self-excluded distances for 10,000 randomly selected training
rows from the same member, and the median percentile across members was used
for each held-out row. High percentiles therefore indicate sparse training
representation, not a calibrated probability density.

Representation varied among the spatial tests, but it did not generally order
their forward performance. Across the ten squares, the Spearman correlation
between median representation percentile and the ML-to-uniform-$C$ RMSE ratio
was $-0.006$ for CFG02 and $-0.030$ for CFG04. Several poorly represented
cases performed well, while high-support failures also occurred where sparse
representation was uncommon. PIG contained a larger sparse tail: 19.2\% of
its eligible area exceeded the training-reference 95th percentile. CFG02's
RMSE ratio rose from 0.800 in the lowest category to 1.009 above the 95th
percentile, with the loss of relative skill concentrated above
$500\,\mathrm{m\,a^{-1}}$. This catchment result suggests that sparse
representation can matter in fast flow, but the mixed square results and weak
point-level associations do not support a general causal explanation of model
failure.
```

## Concise response to the training-imbalance concern

```latex
We added a diagnostic based on the predictor combinations encountered by each
trained MLP. For every held-out row, we measured the twentieth-neighbor
distance to each ensemble member's exact training rows and normalized it by a
self-excluded training-reference distribution. This retains the correlated
rows that contributed to the training loss and avoids assuming independent or
uniform predictors. Sparse representation did not explain performance across
the ten spatial replicates: the square-level correlations with relative RMSE
were near zero for both CFG02 and CFG04, and several failures occurred where
the sparse tail was small. In PIG, however, the sparsest predictor combinations
were associated with reduced relative skill in flow faster than
$500\,\mathrm{m\,a^{-1}}$. We therefore report training representation as a
descriptive limitation rather than as a general cause of success or failure.
```
