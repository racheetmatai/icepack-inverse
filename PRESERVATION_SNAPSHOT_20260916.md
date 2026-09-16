# Preservation snapshot — 16 September 2026

This commit preserves the code state used for the revised Journal of
Glaciology manuscript before code streamlining begins.

It includes the inversion and L-curve workflow, the frozen experimental
design, CUDA MLP training and campaign launch code, forward-simulation and
evaluation code, diagnostic analyses, paper figure/table generators, tests,
the current manuscript package, and the two locally modified Icepack modules.

Large inputs, trained models, full run directories, and other regenerated
artifacts remain outside Git under the repository's existing artifact policy.
The preservation commit is a source snapshot; it is not yet the streamlined
public reproduction package.

The surrounding Icepack checkout was at commit
`28eed36fe652da79769d0130822037f903b23ed3`. Its locally modified
`datasets.py` and `statistics.py` are copied under
`icepack_overrides/src/icepack/` so these changes are preserved in this
repository.

The manuscript snapshot is under `manuscript_20260916/`. Its figures, tables,
equations, and reported numerical values are frozen by author decision.
