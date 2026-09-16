# Change log

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
