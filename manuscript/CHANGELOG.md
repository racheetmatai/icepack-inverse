# Change log

## Transfer-predictability appendix — 19 September 2026

- New appendix section "Predicting where transfer succeeds" with Fig. 13 (two
  panels), responding to the reviewer's question of how a user would know
  where a point-wise prediction is useful. A random forest classifier is
  trained to predict, row by row, whether the predicted C gives lower local
  velocity error than uniform C (and, more strictly, halves it), using observed
  speed alone, the twelve predictors, or both. Squares: leave-one-square-out
  over all six configurations. PIG: CFG02, trained on the ten central squares
  with the 18,409 rows inside PIG removed.
- Result: the combined classifier is not consistently better than observed
  speed alone (higher AUC in 23 of 50 square tests), is below AUC 0.5 in 17
  of 50 (25 of 48 for the stricter criterion), and does not transfer to PIG
  (AUC 0.32 with speed, 0.55 without).
- One sentence added in "Interpreting the point-wise relationship" pointing to
  the new section. One reference added (Breiman, 2001).
- No inversion, MLP training or forward simulation was run. Per-row errors for
  CFG01, CFG03, CFG05 and CFG06 were exported by interpolating the saved
  controlled-replacement velocities onto the observation rows; re-exporting
  CFG02 and CFG04 the same way reproduced the archived arrays exactly.
- Recompiled: 52 pages (was 49), no undefined references. Figures 1-12 keep
  their numbers.

## Figures, joint-support text and Zenodo DOI — 18 September 2026

Recorded here for completeness; made in a separate session (see
`README_FOR_NEXT_LLM.md`). Figure 3, 5a and 7 panels regenerated with larger
axis labels; the unconverged L-curve point removed from Fig. 8, its caption
and the text; the joint-support appendix text corrected to state the 40 km
separation as a square exclusion box and to describe the two stages
(twelve-predictor square selection and per-configuration reporting); Figure 5
float placement and Figure 3 subfigure widths adjusted; Code and data
availability given the Zenodo DOI 10.5281/zenodo.22839669. In the same pass a
truncated equation reference (`\ref{eq}`, rendered "Eqn ??") was restored to
`eq:functional_relationship`, and two cited entries missing from the
repository bibliography (Meierbachtol and others, 2015; Larour and others,
2012) were restored.

## Appendix figure addition — 17 September 2026 (post-audit pass)

- Added Appendix Figure 9 ("The modeled domain and the eligible region used
  for training and for every reported metric"), closing reviewer comment
  R-DET-25a (`review_text.txt` L266): "Metrics are only reported when
  phi > 0.1, but that region is not shown anywhere... There should be a
  figure explicitly showing the region." The tracked response
  (`REVIEW_TRACKER.md`) had added descriptive text but never the requested
  figure; every existing map already restricted display to the eligible
  region alone, so the excluded fringe was never shown for contrast.
- The figure is built from the frozen 5 km support grid
  (`frozen_design/amundsen_input_support_grid_5km.npz`) every other figure
  already masks by: no new computation, inversion, or simulation.
- Added one main-text cross-reference at the first statement of the
  $\phi>0.1$ criterion.
- Recompiled (50 pages, was 49); clean-extraction ZIP rebuild verified
  identical. No other manuscript text, equation, citation, label, or
  numerical value changed.

## Code and data availability correction — 17 September 2026 (later pass)

- Corrected the Code and data availability statement, which cited a separate
  `racheetmatai/icepack-mlp` repository for model-training code. That
  repository holds only the legacy notebook-based training this project's
  own docs call formally superseded; the accepted training implementation
  (`icepack-mlp/production_training/`) already lives inside
  `racheetmatai/icepack-inverse`, alongside the inversion and forward-solve
  code, since the reproducibility-package consolidation. Now cites a single
  repository. No other text, equation, citation, label, or numerical value
  changed. Recompiled (49 pages, no undefined references); clean-extraction
  ZIP rebuild verified identical.

## Controlled-replacement correction — 17 September 2026

- Evaluated modeled velocities against the original paired MEaSUREs raster
  components at verified 450 m pixel centres.
- Replaced inversion-reference C only inside each square's complete 130 km
  withheld footprint, the PIG holdout, or the combined inter-catchment
  corridor geography, using identical masks for the median predicted and
  uniform controls and retaining reference C elsewhere.
- Ran and verified 76 controlled simulations: 60 square median-C, 10 square
  uniform-C, PIG CFG02 and its uniform baseline, and the three inter-catchment
  configurations (CFG04-CFG06) with their shared uniform baseline; no MLP
  training or inversion was rerun.
- Traced every affected numerical claim against the corrected tables and
  corrected: the SQ02 CFG02 central relative RMSE (0.47 to 0.49, unrounded
  0.4884288); the SQ03 local-error improvement area (CFG02 90.9% to 90.4%,
  CFG04 77.4% to 77.5%); a class-5-only square RMSE range endpoint (135.8 to
  136.3 m a-1); and the six BedMachine error-correlation median/range
  summaries, recomputed from the controlled velocities and original MEaSUREs
  components. See
  `production_workflow/controlled_replacement_20260917_a/MANUSCRIPT_NUMERICAL_AUDIT.md`
  for the complete audit trace.
- Updated the square and PIG performance values and affected figures. The
  primary improve/worse classifications, per-square configuration winners, and
  overall conclusions did not change.
- Recomputed comparable C-error/velocity-error correlations. Rounded manuscript
  summaries remain 0.32 for CFG02, 0.42 for CFG04, and 0.67 for PIG CFG02.
- Extended the existing L-curve figure with the valid finite r_C=0.05 run as an
  open unconverged point; the selected r_C remains 0.01414213562.
- Added the appendix subsection "Sensitivity to the replacement region"
  (`app:replacement_sensitivity`), immediately after the replacement-mask and
  grounding description, comparing the earlier sector-wide replacement design
  with the controlled design, separating the effect of the observational
  correction from the effect of the replacement geometry, and explaining the
  quadratic-element replacement boundary (no taper, not a literal
  discontinuity, 40 km buffer separation, convergence not evidence of
  negligible boundary influence). Added one cross-reference to this
  subsection in the main-text non-local-response discussion.
- Reran the inter-catchment holdout as four controlled forward solves
  (CFG04-CFG06 plus the shared uniform baseline), replacing reference C only
  inside the combined corridor geography. Replaced the appendix's earlier
  sector-wide inter-catchment velocity values and temporary "old-design"
  note with the controlled results (CFG04-CFG06: 470.3, 119.0, 109.3 m a-1;
  uniform 578.4 m a-1; inversion reference 28.1 m a-1); the configuration
  ordering is unchanged. Target-C diagnostics and distributions were not
  altered.
- Recompiled the manuscript with Tectonic 0.17.0 (49 pages; previously 48
  before the appendix addition) and rebuilt the Overleaf upload ZIP
  (`JOG_Overleaf_CONTROLLED_20260917.zip`) and its clean-extracted copy from
  the corrected source; both compile identically with no undefined citations
  or cross-references.
- Preserved the 15 September author-approved source unchanged in its original
  directory. No other wording, equation, citation, label, or structure was
  changed; corrections were limited to the numerical values above, the new
  appendix subsection, its single cross-reference, and the inter-catchment
  update.

## Approved consistency pass — 15 September 2026

- Applied the 90 approved proposals, amended by consistency_updates.docx.
- Preserved the author's request to call measured error reductions improvements.
- Standardized objective function and held-out regions; clarified relative
  velocity RMSE, reference C, sample-location matching and buffer correlation.
- Defined R_C^2 before the first table using it.
- Kept the inter-catchment experiment in the appendix and added its geographic
  motivation and measured interpretation.
- Retained the author-confirmed englacial-temperature reference.
- Repeated all 15 paired sign tests; all Holm-adjusted p-values are 1.0.
- Added the two verified method references; Google Scholar export unavailable.
- Wrapped Table 2 headers to accommodate the approved explicit metric names.
- Recompiled the PDF without unresolved references or overfull boxes.
- Preserved equations, table values, figures and all labels.


## Seven-item support-audit correction — 10 September 2026

- Repaired support labels by stable row ID and rebuilt affected diagnostics in
  a separate versioned bundle. Whole-population results and manuscript numerical
  findings remain exactly unchanged.
- Corrected velocity RMSE to equal-area 450 m observation-grid sampling;
  distinguished the uniform-C cell-area mean and cell-center membership.
- Corrected marginal/joint support descriptions and relevant captions;
  distinguished the actual-training-row representation diagnostic.
- Replaced the driving-stress information claim with the measured comparison,
  reported five Holm-adjusted p-values of 1.0, and made the PIG conclusion
  quantitative (approximately 8% lower RMSE; 78.1% lower local error).
- Used neutral configuration/support headings, consistent predictor/metric
  names, and existing cited examples of potentially unrepresented basal controls.
  Removed only repeated qualifications; no general rewrite was performed.
- Reconciled reviewer records and corrected the handoff's inversion parameter
  to 0.01414213562. Legacy training code was not changed.
- Removed the sign-test minipage wrapper to restore normal spacing and review
  line numbers. No figure asset, equation, or numerical table changed;
  the predictor table only uses the consistent alignment name.

## 10 September 2026: targeted corrections only

- Replaced uniqueness conclusions with demonstrated performance of fitted mappings.
- Corrected interpretation of loss decline, common architecture, robust scaling,
  and clipping of the alignment value rather than its denominator.
- Removed repeated qualifications without removing numerical examples.
- Restored the original Parish and Duraisamy (2016) FIML acknowledgment and
  reference; documented mesh interpolation of prescribed englacial temperature
  and its use in Icepack's rate-factor function.
- Traced historical subsets to the 11 August 2026 author decision; standardized
  selected predictor-group terminology in the affected narrative.
- Updated JOG/JGR response records and coverage boundaries; no reweighting or
  alternative-regression experiment is claimed.
- Figures, captions, tables, equations, section order and numerical results unchanged.


## 9 September 2026: training representation in the ten squares

- Added two Results paragraphs after the predictor-support analysis, reporting
  the CFG02/CFG04 patterns and the SQ06 and SQ05 counterexamples.
- Added "Training representation and control error" within the existing
  target-distribution appendix: method, percentile definition, interpolation
  order, evaluation weights, speed-specific associations and Table 7.
- Reused the completed diagnostic. This addition concerns only the ten central
  squares; it adds no PIG representation argument. Existing figures, other
  manuscript wording and scientific results were retained.
- Updated ZIP: Testing_point_wise_inference_of_basal_friction_training_representation_20260909.zip.

- Consolidated the latest active Overleaf root and its compiled pass files into
  one authoritative `manuscript.tex`; the source pass files were not changed.
- Collected only the 18 figures used by the manuscript into a portable,
  section-organized figure tree and corrected the four non-portable Figure 5
  and Figure 6 paths in the source package.
- Revised only the approved parts of the Introduction and Conclusion: defined
  the two predictor groups, clarified the role of surface air temperature,
  corrected the geothermal-heat-flux explanation, simplified the experimental
  statement, bounded the final inference, and replaced ``spatially independent''
  with ``spatially withheld.'' The existing MLP justification was retained.
- Simplified the main-text descriptions of the common eligible rows and
  the 130 km exclusion/50 km test/40 km buffer design; moved quantitative and
  bookkeeping detail to the appendix.
- Corrected `CFG01--CFG01` to `CFG01--CFG06` and added concise appendix pointers
  for predictor construction and the regularization decision.
- Replaced the former appendix with a verified reproducibility appendix covering
  the SSA implementation, basal law and grounding ramp, boundary treatment,
  buffer selection, common rows, predictor construction, target-
  distribution evidence, training/L2 settings, complete set of MLP runs, and
  statistical hierarchy.
- Added compact inversion-reference-C distribution figures for all ten square
  experiments and the two regional experiments, plus verified numerical
  containment evidence. No target values were used for holdout selection or
  row removal.
- Reused the verified L-curve, workflow, convergence, distribution, and result
  data. No inversion, training, or Icepack simulation was rerun.
- Applied two layout-only corrections: an appendix heading was made long enough
  to avoid the class's run-in rendering, and the terminal sign-test paragraph
  was kept together ahead of the convergence figure.
- Repaired the four appendix-section references with title-based LaTeX
  cross-references, because this JOG class leaves appendix section counters
  unnumbered.
- Restored the Roberts et al. (2017) and Valavi et al. (2019) citations supporting
  the rationale for spatially separated validation; their BibTeX records were
  already present in the canonical bibliography.
- Removed the requested defensive paragraph at the end of the transfer-test
  interpretation.
- Replaced audit-like wording in the supplementary workflow with direct labels
  for the common dataset, fixed training and validation rows, predicted (C),
  predictor support, agreement with reference (C), and velocity error.
- Added the verified validation-MSE contrast between CFG02 and CFG04 to the
  square-results interpretation, clarifying why CFG04 remains useful as the
  selected geophysical comparison despite its shallow fitted relationship.
- Replotted the ten square target-density panels with independent display
  windows spanning the 5th--95th-percentile extent of the three row sets
  in each experiment. The underlying rows and numerical 1st--99th-percentile
  diagnostics were not changed.
- Completed a line-by-line plain-language pass through the active manuscript.
  Names for training, validation, buffer, and evaluation rows; MLP predictions;
  regional experiments; predictor support; and velocity performance are now
  used consistently. This pass did not change any reported result.
- Replaced the unexplained regularized-cosine notation
  `\widetilde{\cos\theta}_{bs}` with (a_{bs}). The text now states directly
  that (a_{bs}) approaches (cos\theta_{bs}) when both slopes exceed the
  stated denominator floors.
- Audited all active citations against Crossref, DataCite, publisher, or
  official dataset metadata. Corrected final-publication records, author
  names, title capitalization, page ranges, journal metadata, and DOIs;
  removed 63 uncited legacy entries so `bibliography.bib` now contains exactly
  the 57 sources cited by the manuscript.
- Added direct support for the BedMachine construction, model dependence-based
  variation in inversion products, and L-curve use in ice-sheet inversions.
  Realigned the geophysical citations with the claims they support and removed
  an unrelated hydrology citation from the grounding-ramp sentence.
