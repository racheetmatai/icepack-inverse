# Release verification

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
