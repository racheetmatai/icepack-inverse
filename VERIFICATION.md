# Release verification

## Submitted manuscript and paper checks - 30 September 2026

- `manuscript/manuscript.tex` and `manuscript/bibliography.bib` are the
  submitted versions; `manuscript/manuscript.pdf` was rebuilt from them (53
  pages). Figures are unchanged from 29 September.
- New `production_workflow/paper_checks/` holds the read-only checks behind
  four statements in the text. Each reads the unpacked Zenodo artifacts only;
  nothing is retrained or re-simulated. Results of the runs on 30 September
  are listed below each script's name.
  - `regional_validation_c.py`: REG_INTER CFG04-06 median validation C RMSE /
    R_C^2 0.317/0.764, 0.021/0.999, 0.020/0.999, as in the appendix.
  - `check_blend_bound.py`: largest change in footprint C RMSE 0.0038 (SQ01
    CFG04); 0 affected rows in all 60 central squares; PIG unchanged at four
    decimals. Supports "at most 0.004".
  - `check_speed_classes.py`: in the complete SQ06 footprint, flow of
    100-1000 m/a carries 90.8% of the CFG04 squared velocity error (62.1% +
    28.7%) and flow below 100 m/a carries 3.6%; CFG02 has the smaller C error
    in the faster classes and CFG04 in the slowest. The C RMSEs printed here
    (0.435 and 0.437) use the controls applied in the simulations; the
    manuscript's 0.435 and 0.440 use the median predictions, as in the C tables.
  - `training_density_sensitivity.py`: identical to the 28 September run; the
    k = 20 baseline reproduces the published percentiles (largest difference
    1e-14); median within-square Spearman correlations stay between -0.131
    and 0.092 in every variant (PIG 0.014-0.089).
- Hash-pinned sources restored. The 28 September comment and docstring
  rewrite had changed 33 Python files whose committed SHA-256 is recorded in
  archived manifests (run manifests of the 660 training runs, L-curve and
  dataset provenance, diagnostic manifests); `verify_lcurve_selection_bundle.py`
  enforces one of them (`lcurve_selection.py`), so any script that loads the
  definitive inversion stopped with "Verifier is not using the frozen selector
  source". These 33 files are now byte-identical to the previous release.
  31 of the rewrites were comment-only; the other two only renamed plot
  labels ("geophysical" to "subglacial") in diagnostics not used in the paper,
  whose archived outputs keep the old labels. The only remaining tracked file
  whose previous hash appears in an archived manifest is Figure 5, which was
  regenerated on purpose (29 September entry below).
- Tested in a clean copy in the Icepack container after these changes: 8/8
  package tests and 66/66 workflow tests pass; `scripts/reproduce_paper.py`
  passes (22 outputs present; Tables 1-3 equal the archived values), and 21
  of 22 figures render identically to `manuscript/figures`, the remaining
  PIG panel (figure6a) differing only by anti-aliasing (mean 0.14 of 255).

## Predictor-group name and paper title - 29 September 2026

- The paper now calls the second predictor group "subglacial-product" instead
  of "geophysical-product", and its title is "Do ice and subglacial
  observables determine basal friction? A point-wise test in the Amundsen Sea
  sector". Visible labels in seven scripts changed accordingly (for example
  "Selected subglacial" for CFG04 in Table 2 and the Figure 5 column title).
  Identifiers such as `CFG04_best_geophysical` are unchanged because they name
  files and keys in the archived artifacts; the README explains this.
- `reproduce_paper.py` compares Tables 1-3 with the archived tables without
  the `configuration_label` column, which holds the earlier names in the
  archive; every other column, including the configuration IDs, is compared.
- Figure 5 in `manuscript/figures/results/` was regenerated; only the CFG04
  column title differs from the previous file.
- Figure 1 (`generate_method_overview.py`, a drawing with no data) was
  reworded at the author's request: step 3 now says "Use reference C as the
  target outside the withheld region. Train separate MLPs for each test.";
  step 4 is "Simulate velocity with predicted C" ("Replace reference C with
  predicted C inside the withheld region. Run one Icepack simulation with this
  C field."); step 5 says "Compare modeled with observed velocity. Repeat with
  uniform C and with reference C." The regenerated
  `manuscript/figures/appendix/method_overview.pdf` replaces the old one.
- Study-region panels (`figure1a_observed_speed_and_regions`,
  `figure1b_holdout_geometry`, drawn by `figure1()` in
  `generate_revision_figures_and_tables.py`): both are now saved at the same
  height (6.2 x 5.6 in and 4.9 x 5.6 in, no tight cropping) so that, placed
  side by side at equal height (subfigure widths 0.545 and 0.43 of the text
  width), they print at the same scale; the legend font is 10.5 pt (was 9) and
  the dashed-line legend entry reads "Withheld region". Data and drawing are
  otherwise unchanged. Not yet re-run through `reproduce_paper.py`.
- Figure-style pass for the reviewer's figure comment (30 September): Figure 6
  legend reads "50 km evaluation square" / "130 km withheld region"
  (`generate_corrected_figure5.py`); Figure 13 y-label and reference line use
  "first rule" (`generate_appendix_transfer_counts_figure.py`); larger cell
  values in Figure 5 (`generate_revision_figures_and_tables.py`,
  `generate_corrected_figure4b.py`); L-curve point labels 10 pt, labelled
  ticks at 1-2-3-5 on both log axes, rightmost label placed left
  (`extend_lcurve_figure.py`); eligibility-map legend 11 pt
  (`generate_appendix_eligibility_map.py`); reference-C distribution panels
  drawn on an 11 x 5.4 in canvas with larger text (`assemble_c_target_panels.py`).
  Data are unchanged. Regenerated from the archived artifacts; not yet re-run
  through the full `reproduce_paper.py`.
- Approved by the author (30 September): Figure 5a support map uses cividis
  (viridis is kept for speed only); the ML-minus-uniform local error
  difference in Figures 6 (rows e, f) and 7c uses PuOr_r (red-blue is kept for
  the C difference). Figure 3c uses the same velocity-error colour limits as
  Figures 6 and 7 (1-2000 m/a, log) and the label "Vector velocity error";
  Figure 3b is clipped to the same map extent as panels a and c
  (`generate_revision_figures_and_tables.py`,
  `controlled_replacement/generate_corrected_inversion_panels.py`,
  `generate_corrected_figure5.py`, `generate_corrected_figure6_velocity_panels.py`).
- Tested in a clean copy of the repository in the container: 8/8 package
  tests, 66/66 workflow tests, and a full `reproduce_paper.py` run from the
  archived artifacts passed (21/21 figures present, 3 tables match; 6 min 41 s).
  The reproduced Figure 5 matches the new file.

## Code comments, two figure scripts, and test settings - 28 September 2026

- Comments and docstrings in 96 Python files now describe what each script
  does in the paper's terms (inversion-reference C, misfit and roughness,
  restricted replacement), without internal stage names or revision history.
  A syntax-tree comparison with docstrings removed confirms that the code of
  all 96 files is unchanged. Content corrections: the grounding ramp is
  described as continuous but not smooth, scaling the friction coefficient
  C0 * phi * exp(C); C is described as the dimensionless basal-friction
  control; the `lcar` and `δ` parameters are described correctly; observation
  pixels are described as having equal projected area.
- `extend_lcurve_figure.py` draws the r_C = 0.005 run as a regular point and
  r_C = 0.05 as an unconverged point. The r_C = 0.005 run converged (final
  gradient norm 1.7e-4, below 1e-3) with the same scientific source code as the
  selection-window runs; it was stopped by a fixed iteration count rather than
  the block-wise rule. Recomputed with the six converged points, the maximum
  curvature is still at r_C = 0.014142, with the next-highest 7.1% lower.
  `reproduce_paper.py` passes the r_C = 0.005 manifest from the archived
  selection bundle and also copies a PDF version of the figure.
- `generate_appendix_eligibility_map.py` colours the 450 m pixel centres inside
  the model mesh using the saved dataset rows, instead of the 5 km design grid,
  whose domain flag differed from the mesh near the Dotson ice front and around
  islands. 94.6% of the 1,533,530 in-mesh pixels are eligible.
- `pytest.ini` sets the import paths and turns off output capture. In a clean
  clone, `python -m pytest tests` (8 passed) and
  `python -m pytest production_workflow/tests` (66 passed) need no further
  settings; the README gives the commands.
- `manuscript/VERIFICATION_REPORT.md` no longer records a local folder path.
- End to end: in the clean clone, `reproduce_paper.py` run on the deposited
  archives (with the current archive 07) finished, all 21 expected figure
  files were present, and the three regenerated tables match the archived
  values.

## Transfer-predictability addition — 19 September 2026

- New analysis for the appendix section "Predicting where transfer succeeds"
  (manuscript Fig. 13): `production_workflow/analyze_transfer_predictability.py`,
  `production_workflow/generate_appendix_transfer_predictability_figure.py`,
  and `production_workflow/controlled_replacement/export_map_fields_all_configs.py`.
  Accepted outputs are in the new archive 07,
  `07_transfer_predictability.tar.gz` (295,283,616 bytes, SHA256
  `142ea126842d21787b3f53552b3840e56f3e0c1c22974af1c186ae8651f08dc0`),
  registered in `configs/artifacts.json`. Archives 01–06 are unchanged.
- Export check: per-row fields for all six configurations were exported in the
  Icepack container by interpolating the saved controlled-replacement
  velocities. Re-exporting CFG02 and CFG04 reproduced the archive-06 arrays
  exactly (maximum absolute difference 0; row-ID and boolean arrays identical).
  The domain outline and three geophysics rasters needed by the export were
  checked against the SHA256 values in `amundsen_production_config.json`
  before use.
- Reproducibility check: the formal analysis run reproduced the preliminary
  run's square AUCs exactly (fixed seed), and the packaged figure script
  reproduced the workspace artwork byte for byte.
- `scripts/reproduce_paper.py` now regenerates both appendix figures added
  since the controlled-replacement release: the eligible-region map (Fig. 9,
  previously missing from the pipeline) and Fig. 13. The eligible-region
  script now takes its design grid from the base figure module, so it honours
  `JOG_ARTIFACT_ROOT`; regenerating it reproduces the published PNG byte for
  byte.
- `tests/test_public_package.py` requires the four new or changed entry points.
  All 8 package tests pass.
- 18 September figure styling now in the generators: the label-size,
  bounding-box and L-curve changes made to Figures 3, 5a, 7 and the L-curve on
  18 September are now in the package scripts
  (`generate_revision_figures_and_tables.py`,
  `generate_pig_cfg02_spatial_diagnostic.py`, and in `controlled_replacement/`
  `extend_lcurve_figure.py`, `generate_corrected_figure4b.py`,
  `generate_corrected_figure6_velocity_panels.py`,
  `generate_corrected_inversion_panels.py`). `reproduce_paper.py` now rebuilds
  the L-curve from `lcurve_points.csv` and the saved unconverged-candidate
  manifest in the archived selection bundle instead of copying a prebuilt PNG.
  The record key for that candidate is now
  `validated_unconverged_point_not_plotted` (the point is checked but not drawn).
- Clean-room check: archives 01, 04, 05, 06 and 07 as deposited were unpacked
  into an empty directory in the Icepack container and `reproduce_paper.py`
  was run end to end. It finished and the presence check passed (21 of 21
  files). Rendered-pixel comparison with `manuscript/figures`: 16 files
  identical, including the L-curve, Figures 3a-c, 4a, 4b, 5 and the appendix
  figures. Figure 1a/1b and the PIG panels (6a-c) differ only by anti-aliasing
  and, for 1a, a 2-pixel canvas width; the manuscript copies were rendered with
  Matplotlib 3.11.1 and 3.10.9, the container has 3.7.2. Side-by-side renders
  show the same content.
- Two unreferenced legacy PDFs (`figure2_end_to_end_workflow.pdf`,
  `figure5b_pig_holdout.pdf`) were removed from `manuscript/figures/appendix`;
  the manuscript does not include them. No archive changed.
- Later the same day: Figure 3a's colorbar label changed from "m a^-1" to
  "Observed speed (m a^-1)" to match the other figures
  (`generate_corrected_inversion_panels.py`); the panel was regenerated from
  the clean-room unpack and replaces the manuscript copy. The archive-06
  `figures/` folder still holds the earlier label; the script is
  authoritative. Manuscript text: one Results paragraph on training
  representation put in the present tense (with a missing word restored), and
  "Predicting where transfer succeeds" made a subsection. Figure and table
  numbering unchanged; 52 pages.

## External audit corrections - 26 September 2026

- Discussion ("The contrast between random and spatial holdouts"): the
  inference is now stated as comparable accuracy expected under a point-wise
  relationship, acknowledges that a fitted model can transfer poorly for other
  reasons, and gives the evidence against the main alternatives (errors do not
  decrease with denser representation: Table 8 medians 0.299, 0.297, 0.301);
  shrinkage toward a uniform control is described as less use of the
  predictors (a constant is itself a point-wise function).
- C diagnostics are defined as comparing the median MLP prediction with the
  reference C. Checked read-only in the container: the applied control differs
  from that prediction only in cells crossing the replacement edge (0 central
  square rows, 2.5% of PIG rows, 7.9% and 10.5% of the SQ05 and SQ06 footprint
  rows); using the applied control changes those C RMSE values by at most
  0.003 (SQ06 CFG04 0.4399 -> 0.4370) and leaves the PIG Spearman correlation at
  0.674.
- Classifier appendix: the more-common-outcome rule is labelled a hindsight
  reference (it uses the held-out region's outcome rate) and a rule available
  in advance is added, computed from `counts_folds.csv`: predicting the outcome
  more common in the training squares gives a median of 70.0%, and the
  classifiers exceed it in 98 of 294 tests (combined 14/50 and 19/48; speed
  only 17/50 and 22/48). The main-text sentence now says "rarely better than".
- Appendix E states that the L2 calibration retained the buffers, PIG and the
  corridors. Appendix C no longer calls the twelve-predictor selection
  stricter than every configuration: the criteria are not nested (10
  configuration-square pairs have joint support below 95%).
- Response letter M3 updated to match. Manuscript 54 pages; citations, labels,
  figures, environments and headings unchanged.

## Fourth audit - 26 September 2026

- `environments/paper_requirements.txt` now pins `meshio==4.4.6`. The Fig. 3b
  generator reads the model mesh with meshio; the tested container environment
  (Python 3.10.12, the versions listed in that file) already had it, but the
  documented environment did not, so `scripts/reproduce_paper.py` would have
  stopped at Fig. 3.
- Manuscript: the Methods state that only C is inverted and that the rate
  factor is prescribed from englacial temperature; the Discussion notes that C
  therefore also absorbs errors in ice softness, for example in shear margins;
  the Introduction adds that Kyrke-Smith and others (2017) found stronger
  correlation between profile-averaged values. 54 pages.
- Checked without change: the held-out marginal and joint support code, the
  area-weighted C diagnostics behind Table 7, the mesh file hash in git and
  archive 01, and the three prior-study summaries in the Introduction. The
  redrawn Figs. 10-11 come from a summary table byte-identical to the archived
  one.

## Second and third audits - 26 September 2026

- No result, table value or conclusion changed. Every number recomputed during
  the audits reproduced from the archives (Tables 2, 3, 5, 6 and 9, about 70
  square-level values, the Appendix A sensitivities, the support, PIG,
  training-convergence, L2, classifier and L-curve values). Table 6 was rebuilt
  from all 600 runs' raw validation predictions.
- Manuscript text corrected: the speed quarters in the classifier appendix are
  now computed on the rows used to train the PIG classifier (98.5, 86.4, 48.3
  and 52.1% by quarter; PIG 51.3 to 93.8%); the corridor comparison now agrees
  with the paired square results; "about four times" is limited to CFG05 and
  CFG06; the training-history text describes Fig. 12 as drawn (training MSE
  above validation MSE after the early epochs, from batch normalization); the
  inversion misfit uses the original pixel values; the joint-support separation
  is "at least 40 km along at least one grid axis"; the r_C values outside the
  L-curve range are described as run; predictors are stated to be
  finite-element fields on the model mesh (169-330 control points per central
  square), also in the Table 4 caption; the early-stopping threshold, the
  strain-rate floor (1e-5 a^-1), the 95% square-selection rule, the classifier
  settings and the power of the sign tests are stated; Eqn 4 carries a
  velocity scale U = 1 m a^-1 so both terms are dimensionless (numerically
  identical to the implementation); the median is taken at each control point.
  The response letter's sentence on predictor resolution was updated to match.
  54 pages; citations, labels, figure includes, environments and headings
  unchanged.
- `analyze_transfer_predictability_counts.py` now writes the square tests to
  `counts_folds.csv`, the PIG tests to `counts_pig.csv` (which the Fig. 13
  script reads) and a `counts_summary.json` with the `published_model` block;
  previously it wrote PIG rows into `counts_folds.csv` and pooled the two
  forests in its summary. Its `summarize()` reproduces the archived
  `counts_summary.json` exactly. Archive 07 is unchanged.
- `FULL_RECOMPUTATION.md` section 4 now documents the counts pipeline instead
  of the superseded AUC scripts.
- Figures redrawn with the package generators in the tested container
  environment (Matplotlib 3.7.2) from the deposited archives: Fig. 1 ("control
  point"), Fig. 2 (en dashes in the corridor labels; region legend in panel b),
  Fig. 3b (drawn on the model's quadratic mesh instead of as dots; colorbar
  "Inversion-reference C"), Fig. 7b/c (legend clear of the trunk; colorbar
  label and minus signs as in Fig. 6), Fig. 8 (axes labelled dimensionless),
  and Figs. 10-12 (titles removed from the artwork). All 8 package tests pass.

## Discussion: random versus spatial holdout as a test of the hypothesis - 26 September 2026

- One paragraph added to "Interpreting the point-wise relationship". It states
  that if C were a point-wise function of the predictors, validation on randomly
  withheld rows and transfer to a spatially separate region with good predictor
  support would agree, and shows that they do not: CFG02 median C RMSE 0.08 and
  R^2_C 0.99 on validation rows (Table 6) against 0.30 and negative R^2_C in eight
  of ten withheld squares (Table 7), with 94-99% of every square satisfying both
  support criteria (Fig. 5a). It adds that tuning regularization or early
  stopping against spatially withheld data is not needed under the hypothesis,
  and that a gain from smoothing toward a uniform control would move away from a
  point-wise relationship.
- Every number is already in the paper; checked against the archives: validation
  medians 0.079 and 0.986, withheld median 0.30, R^2_C negative in 8 of 10,
  CFG02 both-support 94.3-98.7% (gate2 support_categories.csv).
- No analysis, figure, table or archive changed. Citations, figure includes,
  environments and headings identical; four new cross-references, all resolving
  to the intended objects. 53 pages.

## Transfer-predictability reporting changed to counts - 25 September 2026

- The appendix section "Predicting where transfer succeeds" now reports how
  many held-out rows the classifier labels correctly, against the number
  obtained by predicting whichever outcome is more common in that region. It
  previously reported the area under the ROC curve. The experiment is
  unchanged: same leave-one-square-out folds, same two success criteria, same
  three input sets, same classifier settings and seed. Only the reported
  quantity differs, and the conclusion is unchanged.
- Reason: AUC is fragile here. In 90 of 360 folds more than 99% of the
  held-out rows carry one outcome, so the ranking score rests on a handful of
  points and ranged from 0.04 to 1.00 within that group; and the metric is
  hard to read for a non-specialist audience.
- What the counts show, published classifier, 294 scored folds: 1,967,017 of
  3,626,370 held-out rows labelled correctly (54.2%); median 51.4% against
  77.5% for the more-common-outcome rule, which the classifier exceeds in 32
  of 294 folds. PIG: 30.8-36.4% against 77.6% (improvement criterion) and
  64.2-90.8% against 92.0% (halving criterion), never exceeding the rule.
- Three checks were run before changing the text, all in archive 07 under
  `results/`:
  1. Capacity. A deliberately larger forest (400 trees, unlimited depth,
     minimum leaf 5) and a gradient-boosted model with early stopping on a
     validation split from the training squares give the same answer; across
     all 360 folds held-out AUC is 0.50-0.56 for every model, and a
     label-permutation control sits at 0.49.
  2. Decision cut-off. Choosing it on rows held out from the training squares,
     as a user could, beats the more-common-outcome rule in 29 of 294 folds;
     even the best possible cut-off for each withheld square, chosen in
     hindsight, does so in only 122 of 294.
  3. Goodness of fit. The same classifier labels 98.7% of its training rows
     and 98.3% of rows held out at random from within the training squares
     correctly, so the failure is one of spatial transfer, not of fitting.
- New entry points, all in `production_workflow/`:
  `analyze_transfer_predictability_counts.py`,
  `check_transfer_classifier_thresholds.py`,
  `check_transfer_classifier_fit.py` and
  `generate_appendix_transfer_counts_figure.py`. `scripts/reproduce_paper.py`
  now builds Fig. 13 with the last of these, and the package tests require all
  four. All 8 tests pass.
- Figure 13 replaced. Regenerating it from the packaged script reproduces the
  manuscript files exactly (rendered output identical; the PDF bytes differ
  only in the embedded timestamp).
- Archive 07 rebuilt: 295,660,673 bytes, SHA256
  `2f3586d3351b4c2d4fc6ce0c5fddb1fc33d269c93fa87c6d230e7ee564b2e94f`,
  recorded in `configs/artifacts.json` and the deposit's `SHA256SUMS.txt`. The
  per-row input fields are unchanged; the earlier AUC results and figures are
  kept in `auc_legacy/`, and every fold's AUC remains a column in
  `results/counts_folds.csv`. Archives 01-06 are untouched. The deposit was
  not yet published, so the file was replaced rather than versioned.
- Manuscript: the appendix subsection, its figure, and one sentence in
  "Interpreting the point-wise relationship". Nothing else changed - citations,
  labels, cross-references, figure includes, environments and headings are
  identical to the previous version. 52 pages.

## Author language pass — 23 September 2026

- The author ran the manuscript through Grammarly and returned a Word file.
  The language changes were merged into `manuscript.tex`: mainly
  passive-to-active rewrites of the authors' own procedural statements, plus
  small wording, punctuation and US-spelling fixes.
- Verified mechanically against the pre-pass file: citations (79), labels and
  cross-references (101), figure includes (20), environments (61) and all
  headings are identical. The only numeric difference is "Forty kilometres"
  becoming "40 km". Clauses the Word conversion had dropped were kept.
  Manuscript is now 51 pages.
- Four suggestions were rejected as inaccurate or overclaiming; they are
  listed in commit 134761f.
- No figure, archive or analysis changed.

## Deposit and manuscript finalization — 18-19 September 2026

- **Zenodo DOI assigned: `10.5281/zenodo.22839669`.** Recorded in
  `configs/artifacts.json` (replacing `TO_BE_ASSIGNED`) and in the
  manuscript's Code and data availability statement. The record holds the six
  archives listed in that manifest; all six were checksum-verified against
  `SHA256SUMS.txt` on the author's machine before upload.
- **Joint-support threshold construction is now shipped.**
  `production_workflow/compute_joint_support_thresholds.py` rebuilds each
  cutoff from the frozen 5 km grid alone and checks it against
  `frozen_design/five_region_partition_and_support.json`. All seven — the
  twelve-predictor selection screen and the six predictor configurations —
  reproduce to floating-point round-off, largest relative difference 5.3e-15,
  with matching PCA component counts. Previously
  `describe_heldout_distributions.py` only consumed `joint_q95_cutoff` and no
  shipped code produced it, so a referee could not verify the construction.
- **Separation metric identified and documented.** The frozen cutoffs
  reproduce only with `max(|dx|, |dy|) >= 40 km`, a square exclusion box
  matching the buffer geometry, not a Euclidean radius (Euclidean gives 0.5321
  for CFG02 against the frozen 0.5344). The appendix wording was corrected to
  match.
- **Appendix joint-support text corrected.** It described one criterion where
  the frozen design has two applications of the same construction: selection
  used all twelve predictors (8 components, cutoff 1.5733833091912042), while
  the support reported per configuration uses that configuration's predictors
  and its own cutoff. Science unaffected — selection was the stricter bar.
- **`configs/artifacts.json` archive 04 description rescoped** to
  original-campaign output, pointing to archive 06 for the
  controlled-replacement solves and corrected evaluations. Description string
  only; no code reads it, and the archive's `sha256` and `bytes` are
  unchanged. `tests/test_public_package.py` passes (3/3).
- **Figures.** Figures 3, 5a and 7 regenerated with larger axis labels and
  without `bbox_inches="tight"`, so Figure 3's three panels now share one
  canvas (331.2 x 320.4 pt each). The unconverged L-curve point was removed
  from the figure, its caption and the text.
- **Still open:** the release license remains to be decided, and the Journal
  of Glaciology article DOI has yet to be added to the Zenodo record as a
  related identifier.

## Second post-audit correction — 17 September 2026 (later pass)

A second, separate audit raised three items; all three were checked
independently rather than taken on trust.

- **Table 2's "Better than uniform" counts derived from the retired P_exp
  metric.** The code that actually feeds the current Table 2
  (`evaluate_controlled_campaign.py`) already computed this directly from
  RMSE, not P_exp. But a legacy, unreferenced function in
  `summarize_forward_evaluation.py` (not called by any other script — dead
  code) did derive the same statistic from `P_exp_percent > 0`. Fixed: it
  now derives `squares_better_than_uniform` directly from
  `vector_rmse_m_per_a < uniform_vector_rmse_m_per_a`. Verified
  behavior-preserving: recomputed both ways side by side and the six
  per-configuration counts are identical (8, 8, 6, 7, 9, 9).
- **Three numbers with no dedicated generator.** All three were
  independently reproduced exactly before any code change (Table 6's
  SQ01/CFG01 cell, the square-maps paragraph's three exceedance
  percentages, and Table 2's Uniform C row); see the entry below for how.
  Closed the gap by adding named generators:
  `production_workflow/generate_table6_validation_c_diagnostics.py`
  (aggregates the 600 relevant `validation_predictions.csv.gz` files;
  reproduced all 60 Table 6 cells exactly) and two new fields written by
  `production_workflow/controlled_replacement/summarize_corrected_map_fields.py`
  to a new `square_maps_table2_traceability.json`
  (`square_maps_exceedance_percentages`, `table2_uniform_c_row`; reproduced
  13.0%/22.8%/0.13% and 44.9/25.0-117.8 exactly).
- Rerunning `summarize_corrected_map_fields.py` to add those fields also
  regenerated `corrected_spatial_summary.csv`,
  `corrected_representation_velocity_categories.csv`,
  `corrected_pig_speed_classes.csv`, and `corrected_pig_details.json`. Every
  value in all four was diffed against the pre-rerun archived copies: the
  maximum absolute difference was 1.1e-13 (floating-point summation-order
  noise between numpy/pandas builds), not a real change.
- `06_controlled_replacement_results.tar.gz` was rebuilt again (491 entries,
  was 486; the 5 new entries are the traceability additions above) and
  `configs/artifacts.json` updated to the new hash: 305,119,872 bytes, SHA256
  `552be2eedb2ef73ffc257bb89e28d37fe75b2c7dd6d0d266a07f9a7fc93f87fe`,
  superseding `9c1413c7...` (305,116,604 bytes).
- All 8 regression tests still pass.
- No manuscript text or numerical value changed in this pass (the L266
  appendix-figure addition from the same audit round is recorded separately
  in the manuscript's own `CHANGELOG.md`).

## Post-audit correction — 17 September 2026 (later pass)

An independent audit found two things wrong with the entry below, both since
fixed:

- `manuscript/` in this package was the pre-correction (16 September)
  manuscript, PDF, and 8 of the affected figures, byte-identical to the
  superseded renderings, even though the code and `configs/artifacts.json`
  were already current. It has been resynced exactly from
  `JOG_CONTROLLED_CORRECTION_FINAL_20260917/manuscript` (verified
  byte-identical after the fix).
- `checks.py`'s `relative_rmse` was not actually imported by
  `production_workflow/controlled_replacement/evaluate_controlled_campaign.py`,
  which defined its own identical copy instead; the passing unit test therefore
  did not guarantee anything about the function the real evaluation used.
  `evaluate_controlled_campaign.py` now imports `relative_rmse` from
  `checks.py` directly (a behavior-preserving substitution: the two
  definitions were character-for-character identical except for the
  docstring), so `evaluate_intercatchment.py` and
  `compare_replacement_footprints.py`, which both import `relative_rmse`
  from `evaluate_controlled_campaign`, now transitively use the same tested
  function too.
  `align_original_observations` and `verify_control_pair` remain standalone
  in `checks.py`, exercised directly only by the unit test — but the
  invariants they encode are independently, redundantly enforced at runtime:
  `build_controlled_controls.py`/`build_intercatchment_controls.py`'s
  `write_control()` asserts exact reference-`C` equality outside each job's
  own replacement mask before saving it, and separately asserts every job in
  an experiment shares one identical mask hash; row-ID alignment against the
  original MEaSUREs components uses the older, separately-tested
  `build_observation_alignment` in `evaluate_forward_campaign.py`, not
  `checks.py`. The results are unaffected either way — this correction is
  about what protects them being accurately described, not about a defect in
  the protection itself.
- Independently recomputed the PIG spatial-concentration numbers the
  manuscript reports at the sentence beginning "Locations where the
  inversion-reference error is at least $100\,\mathrm{m\,a^{-1}}$..."
  (3.9% area / 47.1% of CFG02 squared error) directly from
  `production_workflow/controlled_replacement_20260917_a/map_fields/REG_PIG_CFG02_controlled_fields.npz`:
  both reproduce exactly (3.859% and 47.131%). They were not previously
  saved as named fields anywhere, which is why the audit flagged them as
  unsourced; `summarize_corrected_map_fields.py` now saves them explicitly
  as `high_inversion_error_area_fraction` and
  `high_inversion_error_fraction_of_ml_squared_error`. The manuscript number
  itself required no change.
- The archived `corrected_pig_details.json` was subsequently patched with
  those same two fields (every existing value byte-identical; verified by
  asserting the new area-fraction field matches the pre-existing
  `overall.fraction_inversion_error_ge_100` field exactly before writing),
  `06_controlled_replacement_results.tar.gz` was rebuilt (486 entries, same
  as before; only that one file's content changed), and
  `configs/artifacts.json` was updated to the rebuilt archive's hash:
  305,116,604 bytes, SHA256
  `9c1413c78266876aebb2d59c0f3306712c2b1c8e1b9d71afafeb80c2c3b18588`,
  superseding `97a5d613ad...` (305,405,825 bytes).

## Controlled-replacement addendum — 17 September 2026

- `production_workflow/controlled_replacement/` (construction, execution,
  evaluation, and integrity-check code for the controlled replacement design)
  and its two regression-test modules were added to this package.
- `tests/test_controlled_replacement.py` (4 tests) and
  `tests/test_public_package.py` (3 tests) pass: stable-row-ID alignment
  against original-raster observation components, identical ML/uniform
  replacement masks with exact reference-`C` preservation outside the mask,
  the near-zero relative-RMSE denominator guard (`checks.relative_rmse`,
  1e-12 tolerance, returns NaN below it), correct manuscript-scenario
  selection, the public-package artifact-manifest schema, and the absence of
  the private-provenance note from the public tree. (At the time this bullet
  was written, `checks.relative_rmse` was not yet the function the real
  evaluation used — see the correction above.)
- `06_controlled_replacement_results.tar.gz`
  (`JOG_REPRODUCIBILITY_RELEASE_20260917_CONTROLLED/`) was hashed and matches
  its `configs/artifacts.json` entry exactly: 305,405,825 bytes, SHA256
  `97a5d613ad5be936e52624019e8fdad4c92bf240ead29ceafae6726d792b3df5`.
- This addendum does not repeat the inversion, MLP training campaign, or
  L-curve selection; archives 01-05 from the 16 September deposit are
  unchanged and were not re-verified in this pass.
- The corrected manuscript (`JOG_CONTROLLED_CORRECTION_FINAL_20260917/`) was
  separately recompiled with Tectonic 0.17.0 (49 pages, no undefined
  references or citations) and its clean-extracted Overleaf ZIP was
  recompiled to the same page count and body text, confirming the packaged
  PDF is reproducible from the archived source.
- The Zenodo DOI and release license remain to be assigned before publication;
  the archive-06 addition has not been uploaded.

## 16 September 2026

The staged package was exported to a clean directory and tested with separately
extracted archives on 16 September 2026.

- Paper archive profile: all three SHA-256 hashes match the manifest.
- Package tests: 3 passed.
- Training tests: 7 passed.
- Workflow tests: 66 passed, with the Firedrake environment activated.
- All 17 manuscript figure assets were reproduced or, for the archived
  L-curve artwork, copied from the verified selection archive.
- The three regenerated summary tables match the archived tables exactly.
- Six figure assets render identically at the comparison resolution.
  The remaining comparisons were visually inspected: values, map patterns,
  contours and scales agree; fonts, spacing and raster rendering vary.
- The approved manuscript and its original artwork are retained unchanged.
- The Icepack patch applies to clean source at the pinned commit.
- Source parsing and tracked-file size/credential-marker checks passed.

The original figures were produced with several Matplotlib versions (3.7.2,
3.9.1.post1, 3.10.7 and 3.11.2). The tested paper environment uses 3.7.2;
pixel-identical reproduction of every original asset is therefore not claimed.
The reference artwork is included so readers can compare the outputs.

This release check did not repeat the full inversion/training/flow campaign or
rebuild the complete Docker image. It tested the preserved workflow code and
reproduced figures and summary tables from its accepted archived outputs.

The Zenodo DOI and release license remain to be assigned before publication.
