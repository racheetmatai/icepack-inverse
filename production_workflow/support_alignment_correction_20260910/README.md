# Support alignment correction — 10 September 2026

## Authoritative replacement and scope

Use `../gate4_forward_evaluation_support_aligned_20260910` for all new gate4
support-stratified evaluation. The old `gate4_forward_evaluation_20260829_a`
and the support panels of `gate4_forward_reporting_20260829_a` and `_b` are
superseded for support labels/stratified metrics only. Old hash-bound bundles
remain unchanged as provenance; no files were silently amended inside them.

The support archive's `row_index` indexes the original eligible CSV order.
The evaluator sorted its frame by stable row ID, then incorrectly reused those
positions. The repair resolves archived positions to IDs before joining to the
sorted evaluation frame. **255 is the unassigned marker, not an affected-row
count.** There were 1,988,421 marker entries and 2,063,649 changed category
entries across 66 old map archives, counting repeated geographic locations
across configurations separately.

## Results and verification

- All 726 member/median controls were re-evaluated by interpolation of saved
  velocity fields only. No inversion, MLP training, or ice-flow solve was run.
- All 2,112 control/population groups have complete four-category partitions.
- Whole-population metrics are exactly identical (maximum numerical change 0).
  Consequently configuration rankings and the five paired sign tests are unchanged.
- All 66 maps are exactly identical in every array except `support_category`.
- Independent counts reproduce the authoritative gate2 support summaries.
- All 21 later training-representation cases have identical corrected labels;
  their C extension inherits these stable-ID-aligned labels. These completed
  analyses were preserved, not replaced.
- PIG CFG02: relative RMSE 0.9197369496, improved area 78.148758%, both-support
  area 91.936667%, and supported share of net squared-error reduction 85.586325%.
  The manuscript's approximately 8%, 78.1%, 91.9%, and 85.6% remain valid.
- No manuscript numerical result was based on the defective support strata.
  The main support table combines independent gate2 fractions with whole-case
  RMSE; the PIG/representation analyses already corrected the row-ID join.

`numerical_verification.json`, `support_partition_checks.csv`,
`before_after_support_metrics.csv`, and `later_diagnostic_alignment_checks.csv`
provide machine-readable before/after evidence. Empty strata are represented by
zero counts in partition checks; metric CSVs omit undefined empty-stratum errors.

## Downstream consumer trace

| Consumer | Defective input used? | Disposition |
|---|---|---|
| evaluate_forward_campaign.py per-control support metrics | Yes | Rebuilt all 726 JSONs and combined metrics in the replacement bundle. |
| ensemble_member_summary.csv / median_population_metrics.csv | Support strata yes; whole populations no | Rebuilt from corrected per-control metrics. |
| median_map_data/*.npz | Embedded support arrays yes | All 66 repaired; other arrays exactly unchanged. |
| summarize_forward_evaluation.py map_atlas | Yes, support panel | Old panels superseded by corrected_support_maps/*.png (66 panels in 12 images); added a rejection guard for invalid categories. Unaffected speed/error panels were not regenerated. |
| summarize_forward_evaluation.py primary/configuration/paired tables | No, filters support_stratum=all | Existing results remain valid; no figure regeneration required. |
| generate_revision_figures_and_tables.py | No: main support uses gate2 summaries; performance filters all; regional maps read error only | Existing manuscript figure/table results preserved. |
| generate_final_figure5.py / analyze_spatial_patterns_cfg02_cfg04.py | No: whole-population RMSE and independent gate2 support | Square analysis and spatial figures preserved. |
| audit_solver_bounds_influence.py and its verifier | No: filters all | Retry influence conclusions unchanged. |
| generate_pig_cfg02_spatial_diagnostic.py | Explicit stable-ID correction already implemented | Independently verified PIG support fraction and supported improvement; preserved. |
| analyze_training_representation.py and c_diagnostic/analyze_c_representation.py | Correct stable-ID categories, not embedded map labels | All 21 cases verified against repaired maps; preserved. |
| Other plotting/ensemble/C-diagnostic scripts | No reads of defective support arrays/stratified columns found | No affected output to regenerate. |

Source searches covered workspace Python consumers of `support_category`,
`support_stratum`, `point_support_categories`, and `median_map_data`, followed by
inspection of the relevant reads/filters. Historical source snapshots are not
active code. Main manuscript figure assets remain byte-identical.

## Cache and provenance policy

The evaluator cache now includes evaluator SHA256, stable-ID alignment version,
dataset/support/summary hashes, adoption ID, and baseline manifest hash. Median
cache entries additionally verify their map hash. Old forward-manifest identity
alone cannot accept a stale result. The independent verifier rejects old
alignment identities, invalid map labels, and incomplete category partitions.
Use a new output directory; `--supersedes` prevents writing into the old bundle
and checks whole-population agreement before saving every corrected result.
Legacy schema fields such as P_exp remain solely for archive compatibility;
they were not restored as active analyses, manuscript metrics, or figures.

## Reproduction

Scripts: `../evaluate_forward_campaign.py`,
`../tests/test_support_alignment.py`, `../tools/verify_forward_evaluation.py`,
and `../audit_support_alignment_repair.py`. Original evaluator and manuscript
are retained in `source_before` solely for restricted diff/provenance.

In the existing Docker repository, run the evaluator with the existing production
config and `production_runs` adoption, dataset, support, forward, and baseline
bundles; set `--output production_runs/gate4_forward_evaluation_support_aligned_20260910`
and `--supersedes production_runs/gate4_forward_evaluation_20260829_a`.
Use the activated Firedrake environment (including its mpicc on PATH).
The optional SHAP import may be stubbed for this saved-field-only operation;
SHAP is not used. An initial attempt without mpicc failed before evaluation;
the corrected environment completed all controls. Then run the independent
verifier, copy the new bundle to the workspace, and run
`audit_support_alignment_repair.py --workspace F:/Codex/JOG`.

## Manuscript scope

Only the seven approved correction groups were edited. Velocity RMSE is now
described as equal-area observation-grid sampling; the uniform-C cell-area
mean remains distinct. Support uses marginal bounds from non-held-out eligible
rows and joint analogues from the 5 km grid within training geography. All five
Holm-adjusted values (1.0) are stated. Driving-stress and PIG claims are precise,
possible missing basal controls use existing citations, and current reviewer
responses are distinguished from historical planning. No general rewrite,
physical sensitivity experiment, new literature review, or legacy-code fix
was performed.
