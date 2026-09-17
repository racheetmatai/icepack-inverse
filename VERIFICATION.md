# Release verification

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
  the private-provenance note from the public tree.
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
