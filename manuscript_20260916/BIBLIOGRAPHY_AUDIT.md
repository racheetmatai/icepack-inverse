# Bibliography audit

## Scope and result

- Audited all references cited by `manuscript.tex` for authors, title,
  capitalization, year, journal or repository, volume, issue, pages or article
  number, and DOI where one exists.
- Primary metadata sources were Crossref and DataCite, with focused checks
  against publisher or official dataset records when preprint and final
  records could be confused.
- Final state: 57 cited keys, 57 bibliography entries, no missing keys, no
  duplicate keys, and no uncited entries.
- The 63 legacy entries that were not cited by the active manuscript were
  removed from the submission bibliography. They remain recoverable from the
  earlier source package and repository history.

## Material metadata corrections

- Replaced the discussion-paper records for McArthur et al. and Recinos et al.
  with their final *The Cryosphere* records.
- Replaced the Lu and Kingslake discussion paper with the final 2024 article,
  including its revised title.
- Added the final DOI for Haris et al. (2024).
- Corrected the Martos et al. page range from `11--417` to `11417--11426`.
- Corrected `Kim, Hyoung Rae` to `Kim, Hyung Rae` in the ADMAP dataset record.
- Corrected `Schaller, Thomas` to `Schaller, Theresa` in the AntGG2021 dataset
  record.
- Replaced the incomplete Quantarctica citation with the official DataCite
  record and DOI.
- Corrected the final DOI for Seroussi et al. (2014) rather than citing its
  discussion-paper DOI.
- Added verified DOIs to journal articles where they were absent and protected
  proper nouns and product names from unwanted BibTeX case conversion.

## Citation-context audit

- Added Morlighem et al. (2020) beside the BedMachine dataset citation because
  it directly documents the product construction discussed in the text.
- Added Barnes et al. (2021) where the manuscript explains that inferred
  friction depends on model formulation.
- Added Wolovick et al. (2023) where the manuscript describes L-curve use for
  an ice-sheet inversion.
- Separated citations for bed form, sediment mechanics, ice rheology, and
  hydrology so each group now supports the adjacent claim.
- Separated the gravity-method, bed-geometry, sediment, magnetic, and thermal
  citations so they are not asked to support claims outside their scope.
- Removed Zhao et al. (2025) from the height-above-flotation-ramp sentence;
  Leguy et al. (2014) and Seroussi et al. (2014) are the directly relevant
  grounding-line references.

## Coverage check

The introduction retains the recent machine-learning studies explicitly raised
in the earlier reviews: Jouvet (2023) on deep-learning-emulated inversion,
Jouvet and Cordonnier (2023), Cheng et al. (2024), and Wang et al. (2025). The
methods now cite the software/model, observational datasets, grounding-line
treatment, spatial validation rationale, and L-curve methodology where those
sources materially support the text. Standard network components are fully
specified in the manuscript; additional generic citations for Adam, batch
normalization, or SiLU were not added because no claim about those methods is
made beyond identifying the implemented training settings.

## Verification

- BibTeX completed without warnings.
- All citations and cross-references resolved.
- The rendered reference list was inspected on every bibliography page.
- DOI strings, long author lists, page breaks, and the transition from the
  references to the appendix render legibly without clipping or overlap.
