# Release verification

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
