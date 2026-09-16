# Optional LaTeX wording

## Manuscript

```latex
As a secondary diagnostic, we asked whether errors in the predicted inversion
control increased where the withheld predictor combinations were sparsely
represented among the actual training rows. Across the ten square tests, the
median CFG02 $C$ RMSE was 0.299, 0.297 and 0.301 in the training-reference
$\leq 50$th, 50th--95th and $>95$th percentile categories, respectively. The
corresponding CFG04 medians were 0.492, 0.499 and 0.444. Individual squares
varied substantially, and the median point-level association between
representation percentile and absolute $C$ error was weak (Spearman $\rho=0.069$
for CFG02 and $-0.052$ for CFG04). PIG showed increasing $C$ error across the
three categories (0.742, 0.877 and 0.898), but its row-level association was
also weak ($\rho=0.031$) and did not persist consistently after observed speed
was separated. Sparse training representation therefore does not provide a
general explanation for either inversion-control error or forward-velocity
failure in these tests. This comparison concerns reproduction of one
regularized inversion solution; forward velocity remains the primary physical
test.
```

## Response to a training-imbalance concern

```latex
We added a diagnostic based on the actual member-specific training rows. For
each withheld point, we related its training-representation percentile to the
error in the vertex-wise median prediction of the inversion control $C$. The
saved median control used in the forward simulation was interpolated to the
canonical observation grid with the same Icepack interpolation used for the
training target. Across the ten spatial replicates, sparse representation did
not consistently produce larger $C$ error: the median category RMSE was nearly
flat for CFG02 and decreased in the sparsest category for CFG04. PIG showed an
aggregate increase, but the point-level association was weak and the ordering
did not persist consistently within observed-speed classes. We therefore report
training representation as a limitation and diagnostic, but the evidence does
not identify imbalance as a general cause of the observed failures or justify
retraining solely on that basis.
```
