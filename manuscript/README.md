# Canonical Journal of Glaciology manuscript

## Controlled-replacement correction — 17 September 2026

This candidate preserves the 15 September author-approved package and applies
the verified evaluation correction requested on 17 September. Modeled
velocities are compared with the original paired MEaSUREs raster components at
their 450 m pixel centres. In the square, PIG, and inter-catchment
comparisons, predicted and uniform controls now replace the inversion
reference only inside the same withheld footprint or catchment; the reference
control is retained elsewhere. The controlled campaign comprises 76 verified
simulations: 60 square median-C, 10 square uniform-C, one PIG CFG02 and its
uniform baseline, and the three inter-catchment configurations (CFG04-CFG06)
with their shared uniform baseline.

The primary square conclusions are unchanged: no improve/worse classification
or per-square winning configuration changed, in the squares or in the
inter-catchment ordering. Numerical tables, affected maps, the PIG evaluation,
the inter-catchment evaluation, and the comparable C-error/velocity-error
correlations were updated from the corrected evaluation. A traced numerical
audit corrected the SQ02 CFG02 central relative RMSE, the SQ03 local-error
improvement percentages, one class-5 square RMSE range endpoint, and the six
BedMachine error-correlation summaries; see `CHANGELOG.md` and
`production_workflow/controlled_replacement_20260917_a/MANUSCRIPT_NUMERICAL_AUDIT.md`.
Appendix Figure 8 also shows the valid but unconverged r_C=0.05 L-curve point
as an open marker; it was not used to select the adopted regularization. A new
appendix subsection, "Sensitivity to the replacement region," compares the
controlled design with the earlier sector-wide replacement and explains the
finite-element replacement boundary, with one cross-reference from the
main-text non-local-response discussion. The previous author-approved
directory was not modified, and no other wording was changed beyond the
values and additions above.

Compile `manuscript.tex` as the root document. The package is self-contained
and includes the class, bibliography style, bibliography, and all active figure
assets (17 figures; unchanged from the 16 September package — this correction
did not introduce a new figure).

## Current package — 16 September 2026

Compile manuscript.tex as the root document with the supplied bibliography,
igs.cls, igs.bst and figures. The package includes 17 referenced figure assets.
This update adds the horizontal experimental overview in Methods and the
author-approved terminology and training-history clarifications. The
conclusion, numerical results, equations and existing figure artwork remain
unchanged. Local build: tectonic -X compile manuscript.tex.
The following section records the previous release.

## Current author-approved release — 15 September 2026

The authoritative source is now this manuscript.tex, revised from the author's
Testing_point_wise_inference_of_basal_friction__5_.zip, as displayed in
JoG_Full_pass.pdf. The approvals and amendments were supplied in
consistency_updates.docx. The older history below does not describe the
current active figure set.

Use manuscript.tex as the Overleaf root, with the supplied igs class/style.
The upload ZIP contains the source, bibliography, class/style, all 16 active
figure assets, this README, and the newly compiled PDF. No pass files are needed.
The 46-page PDF replaces the stale PDF that was inside the input ZIP.

The approved 90-item terminology review and author amendments were applied.
C remains the dimensionless basal-friction control; modeled velocity is the
standard simulation-output term. R_C^2 is now defined at first use.
Inter-catchment results remain in the appendix with brief context.
The englacial-temperature attribution is retained as confirmed by the author.
All 15 paired sign tests were repeated from existing corrected median-field
results and independently checked by enumerating the possible sign patterns.
All 15 Holm-adjusted p-values are 1.0.

All 35 labels, 12 displayed equations, table numbers and 16 figure assets were
preserved. No inversion, training or forward simulation was rerun.
Two approved method citations were added from verified publication metadata.
Google Scholar citation export was inaccessible; the entries are not represented
as having been exported from Google Scholar.

Verification, the exact source diff and the statistical check are retained one
directory above this manuscript folder. Build intermediates are not in the ZIP.
Older records below and the copied older audit files are historical only.

## Results-figure update, 12 September 2026

This copy starts from the author's latest Overleaf export,
`Testing_point_wise_inference_of_basal_friction__4_.zip`.
Figure 3 now contains the square comparison with the inversion reference,
uniform C and all six configurations on a logarithmic axis. The former PIG
panel is retained unchanged as Appendix Figure 12. Results references and
the corresponding captions were updated. All other scientific text is
unchanged. Historical release notes below describe earlier work.

Use `manuscript.tex` as the Overleaf main document. The accompanying
`results_figure_revision_verification.json` records the restricted edit check.

This directory is the self-contained submission package. `manuscript.tex` is
the only manuscript source and is the root document. It contains the latest
active Overleaf manuscript in compiled order; no pass file is required.

## Overleaf

Upload the contents of this directory and select `manuscript.tex` as the main
document. The Journal of Glaciology class, bibliography style, bibliography,
and every used figure are included.

## Local build

From this directory, run:

```text
tectonic manuscript.tex
```

The figures are organized under `figures/introduction_methods`,
`figures/results`, and `figures/appendix`. All paths are relative and portable.

See `CHANGELOG.md`, `VERIFICATION_REPORT.md`, and `TODO.md` for the exact scope
of this consolidation.

Current revision: the seven-item support-audit correction of 10 September 2026.
Whole-population results and all figure assets are unchanged. Support-stratified
diagnostics were repaired separately; the manuscript now describes metric
weighting, support reference populations, and statistical results accurately.
