# Verification report

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
