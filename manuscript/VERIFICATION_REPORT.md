# Verification report

## Controlled-replacement verification — 17 September 2026

- The restartable registries contain exactly 76 jobs across two campaigns:
  60 square median-C, 10 square uniform-C, one PIG CFG02, and one PIG
  uniform-C simulation (`production_workflow/controlled_replacement_20260917_a`,
  72 jobs), plus CFG04-CFG06 inter-catchment median-C and one shared
  inter-catchment uniform-C simulation
  (`production_workflow/controlled_intercatchment_20260917_b`, 4 jobs, 4/4
  verified complete).
- All 76 jobs completed with finite velocities; file hashes match their
  manifests, reference C is exactly retained outside every replacement mask,
  and the ML and uniform masks are identical within each experiment.
- The accepted evaluator was reproduced for all 61 previously reported
  median-control cases to an absolute tolerance of 1e-10 before corrections.
- Correcting the observational reference and restricting C replacement caused
  no improve/worse classification changes and no changes in the winning
  configuration within any central square or in the inter-catchment
  configuration ordering.
- The manuscript's affected numerical claims were traced and corrected against
  the corrected tables: SQ02 CFG02 central relative RMSE, SQ03 local-error
  improvement percentages, one class-5-only square RMSE range endpoint, the
  six BedMachine error-correlation summaries, and the inter-catchment velocity
  values. See
  `production_workflow/controlled_replacement_20260917_a/MANUSCRIPT_NUMERICAL_AUDIT.md`.
  Unaffected C-target diagnostics, training histories, predictor-support
  definitions, and training-representation distances were not changed.
- The new appendix subsection on replacement-region sensitivity and its
  boundary-condition explanation were checked against the executed code:
  continuous quadratic finite elements, replacement-mask selection at control
  points, exact reference-C equality outside the mask, no explicit taper, and
  the 40 km buffer separation from the footprint boundary.
- The manuscript was recompiled with Tectonic 0.17.0 to 49 pages (48 before
  the appendix subsection was added), with no undefined citations or
  cross-references and no new overfull boxes; only the previously known benign
  underfull-hbox/lineno-encoding/Fontconfig messages remain.
- The rebuilt Overleaf upload ZIP (`JOG_Overleaf_CONTROLLED_20260917.zip`, 23
  files, 17 figure assets) was extracted to a clean directory and recompiled;
  the clean build reproduced the same 49 pages and body text as the packaged
  PDF (only the embedded CreationDate timestamp differs between the two PDF
  binaries). See `controlled_release_record.json` in
  `JOG_CONTROLLED_CORRECTION_FINAL_20260917/`.
- No MLP was retrained and no inversion was rerun. The 15 September
  author-approved package remains unchanged.

## 9 September 2026 training-representation addition

- Current compiled length: 41 pages; earlier page counts below describe
  previous builds.
- The new appendix subsection and Table 7 cross-references resolve.
- Main-text and appendix numerical entries were checked against the existing
  C category and association tables; no analysis was rerun.
- Affected Results and appendix pages were rendered and inspected. The table,
  percentile equation and text fit within the page boundaries.
- The build reports an underfull bibliography line, a bundled lineno encoding
  warning and a Fontconfig configuration message; PDF generation succeeds.
- The addition introduces no references, figures or external dependencies.

## Source resolution

- The latest package has one unambiguous active root:
  `latex_revision/manuscript/igs2eannalsguide.tex`.
- Its complete input order was resolved and flattened into `manuscript.tex`.
- The original archive was no longer present in Downloads at final packaging;
  the previously created faithful extraction at
  `F:/Codex/JOG/tmp/latest_overleaf_20260907` was used.
- An untouched-source build was attempted first. It exposed four non-portable
  Figure 5/6 paths in `pass1_results.tex`; the canonical package fixes those
  references without changing the source pass file.
- All ten source pass files remain unchanged in the faithful extraction. The
  canonical file contains no active `\\input` dependency.

## Content and artifact checks

- The latest active manuscript was the textual source of truth.
- Introduction and Conclusion retain the latest active manuscript except for
  the author-approved scientific clarifications recorded in `CHANGELOG.md`.
- Only explicitly approved main-text changes and the approved appendix
  replacement were made.
- No retired analysis was reintroduced, and the historical velocity-filtering
  defect is not discussed.
- No inversion, training, or forward simulation was rerun. The target-density
  figures were regenerated from the existing dataset and fixed masks solely to
  retain their 5th--95th-percentile display windows and use the manuscript's
  final terminology. The workflow and training-history figures were regenerated
  from their existing sources only to replace labels. No plotted value, row set,
  or numerical 1st--99th-percentile result changed.
- The manuscript has 18 active `\\includegraphics` references and the package
  contains exactly 18 figure files; every reference resolves within the package.
- `bibliography.bib`, `igs.cls`, and `igs.bst` are local to the package.

## Build and visual checks

- Clean Tectonic build: passed.
- Clean extracted-package build emulating direct upload: passed.
- Bibliography audit: passed; 57 cited keys and 57 entries, with no missing,
  duplicate, or uncited entries. Titles, protected capitalization, authors,
  years, publication metadata, and DOI/ISBN identifiers were checked against
  Crossref, DataCite, and focused publisher or official dataset records.
- Citation-context audit: passed; each active citation was checked against the
  claim it accompanies. Three directly relevant records were added, misplaced
  support was corrected, and no unsupported reference was introduced.
- Cross-references: passed; no undefined or multiply defined references.
- Overfull and underfull boxes: none.
- The four title-based appendix references resolve to their intended sections;
  the compiled PDF contains no literal `Appendix .`.
- The Roberts et al. (2017) and Valavi et al. (2019) citations and bibliography
  records resolve without warnings.
- The removed MLP/Icepack paragraph and the superseded workflow phrases do not
  occur in the compiled PDF.
- PDF after bibliography audit: 38 letter-size pages, unencrypted, with no
  missing or clipped figures.
- All five rendered bibliography pages were inspected; long author lists,
  DOI strings, page breaks, and the transition to the appendix are legible.
- Every rendered page was inspected; figure/table legibility, float placement,
  captions, units, appendix order, page numbering, and line numbering passed.
- After the approved plain-language pass and notation correction, the manuscript
  was rebuilt cleanly and every rendered page was inspected again.
- No existing pass file was edited, moved, deleted, or included in this package.
# 10 September 2026 targeted correction verification

This subsection records the earlier targeted pass. The support-audit pass below
is current and supersedes its statements about unchanged captions.

Starting source: canonical manuscript, preserved temporarily as
build_targeted_20260910/before.tex. A restricted diff checks the approved edits.
All 18 figure files match the previous ZIP byte-for-byte; all figure/table
environments, captions, equations and section order are unchanged. Citations
and cross-references resolve; no overfull or undefined-reference warnings.
The rebuilt 41-page PDF was rendered and inspected for layout changes.
The ZIP passed CRC and canonical-file identity checks. A clean extracted copy
compiled successfully with identical auxiliary references and page assignments.
The adopted saved theta field is zero everywhere, confirming A=A0 from the
prescribed temperature without an inverted fluidity multiplier.
Known build warnings remain: bundled lineno encoding/fontconfig messages and
one underfull bibliography paragraph, with no visible clipping.

Evidence consulted: gate2 canonical-dataset production source snapshot
(production_amundsen.py, invert_c_theta.py, feature_units.py) and configuration;
work_package_2/AUTHOR_DECISION_LOG.md, 11 August configuration-selection entry;
original baseline_source manuscript and Parish bibliography entry. No new
literature review or numerical/scientific recomputation was performed.
Build intermediates remain outside the submission directory. Existing pass
manuscript files were not edited. The coverage matrix was updated as authorized.

## Current: seven-item support-audit pass, 10 September 2026

Starting text is preserved in
`production_workflow/support_alignment_correction_20260910/source_before/manuscript.tex`
relative to workspace root. The restricted diff and numerical/release checks
are in that same correction directory.

- All 726 repaired control results pass verification; all 2,112 support
  partitions are complete. Whole-population metrics are exactly identical.
- Every non-support array in all 66 median-map archives is exactly unchanged.
  Gate2 fractions agree, and all 21 independently corrected representation
  cases agree by stable row ID. Existing PIG figures/statistics remain valid.
- All manuscript figure assets are byte-identical. Numerical tables are
  unchanged; Table 1 only standardizes the alignment name. No new bibliography
  entry, equation, or experimental definition was added.
- Canonical source compiles; citations and cross-references resolve with
  no undefined/multiply defined references or overfull boxes.
- All 41 rendered pages were reviewed; affected pages were enlarged as needed.
  The sign-test paragraph was restored to normal spacing/line numbering.
- The upload ZIP is checked for CRC/file identity and compiled after clean
  extraction, with the same auxiliary references and page assignments.
- Only saved-field interpolation and diagnostic reduction were performed.
  No inversion, training, forward simulation, or literature review was run.
- Existing pass files, legacy training behavior, unaffected figures, and
  superseded hash-bound archives were not edited. Retired metrics were not
  restored to the manuscript.

Known harmless warnings: bundled lineno encoding/fontconfig messages and one
underfull bibliography paragraph remain. Final author review, release
identifiers, response-letter locations, and submission metadata are deferred.
