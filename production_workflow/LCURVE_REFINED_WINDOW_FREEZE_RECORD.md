# Refined-window L-curve freeze record

**Decision date:** 19 August 2026  
**Selected regularization:** `reg_C = 0.01414213562`  
**Status at freeze:** selection decision only.

**19 August adoption addendum:** the selected confirmed endpoint subsequently
passed a separate definitive-state audit and was adopted without recomputation.
The controlling record is
`gate1_results/gate1_definitive_inversion_adoption_20260819_a.json`, manifest ID
`sha256-json-v1-406472fbb5141aa80e28f42a08bcb30f710726277877e3c2324deb6d25df8134`.
The historical selection-only distinction below remains accurate for this
freeze record; the external adoption record authorizes downstream use.

## Purpose and scope

This record defines the disclosure that must accompany the L-curve selection
bundle. The value above is supported by the accepted five-point curve in the
local interval `0.01 <= reg_C <= 0.02` and its selected-triple confirmation.
It must not be described as the output of a completed formal
`gate1_lcurve_v2_20260814_a` orchestrator study.

The author accepts a refined-window selection around the historically relevant
`reg_C=0.01` scale instead of completing or forcing acceptance of numerically
unsatisfactory higher-`reg_C` base points. This is an explicit override of the
formal driver's requirement that every base-grid point be valid before adaptive
selection (`lcurve_orchestrator.py`, lines 1323--1330). It does not relax the
pointwise finite-value, objective-identity, stability, or gradient-safety gates
for any point admitted to the five-point curve.

## Incomplete formal base sweep

The immutable formal study contract is
`sha256-json-v1-e129f8b1b3bf4a68d559254a642b410e47f668f4a5e11d7ffa9f55b62fabe74d`.
Its planned base grid was `{0.01, 0.02, 0.05, 0.1, 0.2}`. The outcomes must be
reported as follows:

| `reg_C` | disposition | required disclosure |
|---:|---|---|
| 0.01 | valid, 5 blocks | Used as the lower endpoint of the refined curve. |
| 0.02 | valid, 6 blocks | Used as the upper endpoint of the refined curve. |
| 0.05 | invalid, 6 blocks | Finite and objective-consistent, but it failed the common acceptance gate: final relative changes were `0.00509250` in `E` and `0.0157363` in `R`, and the reduced-gradient norm was `0.001379999 > 0.001`. Exclude from curvature; it may appear only as a visibly labeled context point. |
| 0.1 | failed after 5 completed blocks | PETSc/Firedrake failed with `DIVERGED_PCSETUP_FAILED`; terminal reduced-gradient norm before failure was `0.00663894`. Exclude from curvature. |
| 0.2 | never started | No result exists; do not infer or plot a value. |

The formal `run_state.json` is stale at a running `0.1` request even though the
point manifest records failure. No formal selection, study manifest, or
`definitive_inversion.json` was produced. The freeze bundle must preserve this
history without rewriting that immutable study.

The completed `reg_C=0.005` result belongs to the superseded v1 study and is a
historical solver diagnostic only. It is not an endpoint, extension, or
eligible point in this refined-window selection.

## Effective five-point curve

Use the original valid endpoints and the confirmed midpoint endpoints below.
The midpoint confirmation manifests replace their parent metrics for final
selection geometry.

| `reg_C` | effective `E` | effective unweighted `R` | effective manifest ID |
|---:|---:|---:|---|
| 0.01000000000 | 3104.730961477 | 0.0784802120685 | `sha256-json-v1-a42f883f5ff828ca9fbe4c20cd53e7c3f2766d3e84da094f616b76eeefec7d47` |
| 0.01189207115 | 2954.702361895 | 0.0963179485925 | `sha256-json-v1-121e4342676cfb120c062ef3c368ad8b0c0d5bcf05b089739d6d0a8d2187c7ae` |
| 0.01414213562 | 2833.519357417 | 0.116684062914 | `sha256-json-v1-94c28ac838c8e1eec63973b6511d5b42dea27b9dafbc6b1c6c08cce3fc810a1e` |
| 0.01681792831 | 2728.194953527 | 0.141878056070 | `sha256-json-v1-57e20d20c1b990cb1a1a96ada3721f681b1f32b3092186266be9c222bef034b0` |
| 0.02000000000 | 2637.970263855 | 0.172214148947 | `sha256-json-v1-5e58b4fcf4435ec6e065a8ca29a0f086b03d54e7c6f9ba20fc731fcc7b2e9314` |

All five endpoints have `status=valid`, finite positive geometry, consistent
objective decomposition, and a recognized acceptance basis. Across increasing
`reg_C`, `E` decreases monotonically and `R` increases monotonically.

## Deterministic selection and remaining ambiguity

Selection must be reproduced with the source-matched
`production_workflow/lcurve_selection.py` whose SHA-256 is
`5d496513e1da57b5bf000f037c831975b2a3f4eb119c1e370cb6957e23811cf0`:

1. retain only valid eligible points and sort by increasing `reg_C`;
2. independently min--max normalize `log10(E)` and `log10(R)` over the five
   effective points;
3. compute unsigned three-point Menger curvature at each interior point; and
4. select maximum curvature, breaking an exact tie toward smaller `reg_C`.

The post-confirmation result is:

| interior `reg_C` | curvature |
|---:|---:|
| 0.01189207115 | 0.134663352482 |
| **0.01414213562** | **0.171647133329** |
| 0.01681792831 | 0.164954940927 |

The leading-to-runner-up ratio is `1.040569820854 < 1.25`, so the local elbow
remains close or **curvature-ambiguous**. Do not call it uniquely resolved by
curvature alone. The protocol permits only one midpoint-refinement round; the
author-accepted refined-window round is treated as that round, and no further
midpoint insertion is used.

Before confirmation, the same code selected `0.01414213562` with curvature
`0.171134922748`; the runner-up at `0.01681792831` had curvature
`0.165234493101` (ratio `1.035709430493`). The before/after selection is thus
stable.

## Selected-triple confirmation

The immediate selected triple was
`{0.01189207115, 0.01414213562, 0.01681792831}`. Each point was continued from
its own hash-verified saved control for one confirmation block capped at 50 ROL
iterations.

| `reg_C` | relative `delta E` | relative `delta R` | terminal gradient | ROL last iteration | confirmation manifest ID |
|---:|---:|---:|---:|---:|---|
| 0.01189207115 | 1.42435e-6 | 5.16918e-6 | 7.72809e-5 | 2 | `sha256-json-v1-121e4342676cfb120c062ef3c368ad8b0c0d5bcf05b089739d6d0a8d2187c7ae` |
| 0.01414213562 | 1.40977e-5 | 3.35465e-5 | 6.66230e-5 | 7 | `sha256-json-v1-94c28ac838c8e1eec63973b6511d5b42dea27b9dafbc6b1c6c08cce3fc810a1e` |
| 0.01681792831 | 0 | 0 | 9.93878e-5 | 0 | `sha256-json-v1-57e20d20c1b990cb1a1a96ada3721f681b1f32b3092186266be9c222bef034b0` |

Every confirmation is valid, finite, objective-consistent, below the `0.005`
relative-change tolerances, and below the `0.001` gradient ceiling. Replacing
the three parent metrics by these confirmation metrics leaves both the selected
candidate and the selected triple unchanged. Under the frozen confirmation
rule, a second confirmation round is required only if a confirmation fails or
the selected/current triple changes. Neither occurred, so one round resolves
the stability question even though the curvature ratio remains below `1.25`.

## Two-process execution decision

The author approved at most two independent inversion processes concurrently
on this computer, provided they use distinct run/output directories and each
point's internal block sequence remains ordered. The independent midpoint runs
at `0.01414213562` and `0.01681792831` began at
`2026-08-18T01:45:54.702138Z` and `2026-08-18T01:45:58.240841Z`, respectively.
Both manifests record `run_kind=independent`, no parent, and
`block_01.start=independent_C_zero`. Their overlap is therefore an authorized
capacity choice, not a scientific defect, and is not a reason to rerun them.

## Minimum contents of the immutable freeze bundle

The bundle must contain or hash-reference all of the following and verify them
independently:

- this decision/disclosure record and the explicit author acceptance;
- the formal contract, stale state, valid `0.01`/`0.02` manifests, invalid
  `0.05` manifest, failed `0.1` manifest, and the absence of a `0.2` result;
- the superseded-v1 `0.005` disposition, explicitly excluded from geometry;
- all three midpoint parent manifests and all three confirmation manifests,
  including parent IDs, parent file hashes, output hashes, and state lineage;
- a complete inventory of zero-block/failed diagnostic attempts, explicitly
  excluded from selection;
- the common configuration hash
  `db5cae4d88c687c6bec6d8967119722732305b4dc24463153167510d429853e7`,
  scientific-source hashes, input/design hashes, mesh identity, environment,
  solver policy, and independent `C=0` start evidence;
- the effective five-point table, normalization bounds, complete curvature
  table, ambiguity record, monotonic diagnostics, selected-triple record, and
  a scripted appendix L-curve generated only from eligible effective points;
- canonical manifest-ID validation and independent SHA-256 verification of
  every declared output; and
- a final selection artifact that states both the deviation and
  `selected_reg_C=0.01414213562`, without fabricating a formal study completion
  or a historical `definitive_inversion.json`.

## Required language and downstream boundary

Permitted description: **author-accepted refined-window L-curve selection,
confirmed for endpoint stability**.

Do not describe the result as a completed full-base adaptive-v2 study or an
unambiguous curvature maximum. This freeze record alone is selection evidence;
the later external adoption audit establishes that its selected confirmed
endpoint also satisfies the definitive-inversion requirements, so no redundant
rerun is required.
