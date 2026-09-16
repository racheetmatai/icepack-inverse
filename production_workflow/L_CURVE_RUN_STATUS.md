# Production L-curve run status

Last recorded: 19 August 2026.

## Current state

Regularization selection is closed at `reg_C=0.01414213562`. The formal
adaptive-v2 study did not complete: `0.01` and `0.02` were valid, `0.05` was
invalid under the common gate, `0.1` failed numerically, and `0.2` was not run.
Its stale running state is preserved as attempt history and must not be resumed
or reported as a completed formal study.

- Run ID: `gate1_lcurve_v2_20260814_a`
- Docker path:
  `/home/firedrake/icepack/icepack-inverse/production_runs/gate1_lcurve_v2_20260814_a`
- Started: `2026-08-14T23:14:40.095419+00:00`
- Contract schema: `jog-production-lcurve-contract-v2`
- Contract ID:
  `sha256-json-v1-e129f8b1b3bf4a68d559254a642b410e47f668f4a5e11d7ffa9f55b62fabe74d`
- Final formal-study disposition: incomplete; no orchestrated study manifest or
  definitive inversion was produced.
- Do not report the historical `0.005` diagnostic as part of the accepted
  refined-window curve.

The author accepted a separately run and confirmed five-point local refinement
as an explicit deviation from the incomplete base study. The resulting
selection-only bundle is
`gate1_lcurve_selection_bundle_20260819_a`. It verified all 341 declared files
both in place and after a portable copy. Bundle manifest ID:
`sha256-json-v1-c40934dbb444fd7bf70130cd80ce09a9b63da2884be75a4d03d87195d4ddf6fc`.
The bundle itself remains selection-only. A separate verified adoption record
now designates its selected confirmed endpoint as the definitive inversion,
without recomputation. Adoption manifest ID:
`sha256-json-v1-406472fbb5141aa80e28f42a08bcb30f710726277877e3c2324deb6d25df8134`.

The exact-source prerequisite is
`gate1_forward_smoke_v2_20260814_a`, run from
`2026-08-14T23:03:50.129331+00:00` to
`2026-08-14T23:13:01.809126+00:00`. It passed 12/12 checks; all 28 declared
outputs and its identifier independently verified. Manifest ID:
`sha256-json-v1-00a5d9a67a6a8992c5d7980593e34fb6469ac998cd529c74678500651ae37af3`.
Its zero-control objective equals its misfit (`249483302.858409`), with zero
unweighted roughness and zero weighted penalty.

Before launch, the finalized v2 workflow passed all 58 focused tests in the
container, including real Firedrake saved-control continuation and
same-`reg_C` confirmation integration.

## Superseded v1 study -- preserved, do not resume

- Run ID: `gate1_lcurve_20260813_a`
- Docker path:
  `/home/firedrake/icepack/icepack-inverse/production_runs/gate1_lcurve_20260813_a`
- Started: `2026-08-13T22:43:35.267046+00:00`
- Stopped deliberately: `2026-08-14T22:03:37Z`
- Contract ID:
  `sha256-json-v1-9e2fa547fde3927292972a522df0daabc5de3e03b4c18456a8daf1fa1b2f3130`
- Disposition: preserved as an immutable historical/diagnostic bundle and
  superseded by the author-approved adaptive-v2 protocol.

The completed `reg_C=0.005` attempt is a historical solver diagnostic only. It
is excluded from the v2 candidate population, convergence decisions, curve
geometry, and regularization selection. It exhausted 300 iterations plus one
100-iteration continuation and ended with objective `5351.703849309328`,
reduced-gradient norm `1.6803520015248332e-4`, misfit
`3965.1524765031286`, and unweighted roughness `0.03466378432015464`.

The v1 `reg_C=0.01` point was interrupted when the study was deliberately
stopped. It is incomplete and is not reusable as a v2 point. Preserve its
partial attempt and logs without presenting them as a result.

Do not use `--resume` on `gate1_lcurve_20260813_a`. Do not edit, delete, or
overwrite its contract, attempts, logs, fields, or manifests.

## Approved adaptive-v2 protocol -- frozen 14 August 2026

### Base candidates and serial execution

- Run the base grid serially in this order:
  `reg_C={0.01,0.02,0.05,0.1,0.2}`.
- Every newly introduced base, extension, or midpoint point starts
  independently from exact logarithmic control `C=0`. A confirmation block is
  a same-`reg_C` continuation from that point's own saved control.
- The original protocol specified one inversion process at a time. On 19 August
  2026 the author confirmed that this PC can safely run two independent
  inversions concurrently. The accepted midpoint runs used at most two, had
  separate run IDs/directories and no shared mutable state, and are not invalid
  on that basis.
- Record misfit `E`, unweighted roughness `R`, weighted penalty, total
  objective, native ROL state, reduced-gradient norm, field checks, and full
  provenance after every block.

### Uniform blockwise stopping rule

- Optimize in blocks of 50 ROL iterations.
- Run at least four blocks (200 iterations) and at most six blocks
  (300 iterations) per ordinary point.
- A point may stop after the minimum only when both `E` and `R` have changed by
  no more than 0.5% in each of two consecutive block-to-block comparisons and
  its reduced-gradient norm is no greater than the `1e-3` safety ceiling.
- Nonfinite fields or metrics, an unsafe reduced gradient, or a native solver
  failure cannot be accepted as practical convergence. Preserve such attempts
  and report their status.
- Apply the rule identically and without reference to the eventual curve shape
  or downstream performance.

### Endpoint extension and midpoint refinement

- Compute selection only from finite points accepted by the common rule, using
  the preregistered normalized log-misfit/log-unweighted-roughness curvature
  calculation and the existing smaller-`reg_C` tie-break.
- If the provisional corner is boundary-adjacent on the low side, add exactly
  one low extension at `reg_C=0.005`. If it is boundary-adjacent on the high
  side, add exactly one high extension at `reg_C=0.5`. No further endpoint
  extension is allowed by this protocol.
- If the leading and second-highest eligible curvature values have a ratio
  below `1.25`, treat the corner as ambiguous and run one midpoint-refinement
  round. Add geometric midpoint candidates adjacent to the leading candidate,
  subject to deduplication against points already run. Do not run a second
  midpoint-refinement round.

### Selected-triple confirmation

- Once selection is provisionally resolved, identify the selected candidate
  and its immediate lower and upper evaluated neighbors: the selected triple.
- Give each member of that triple one additional 50-iteration confirmation
  block from its saved control, then recompute its metrics and selection.
- If the selected candidate changes or any confirmation delta fails the frozen
  stability tolerance, confirm the newly selected/current triple once more.
  Permit at most two confirmation rounds in total. Never continue only the
  favored point; if the second round remains unresolved, do not freeze a
  `reg_C` automatically.
- Freeze `reg_C` only after the final selected triple passes the common finite,
  stability, gradient-safety, provenance, and bundle-verification gates.

## Next action

Do not resume either superseded/incomplete study and do not rerun the selected
inversion. Use the independently verified adoption record
`gate1_definitive_inversion_adoption_20260819_a.json` to load the selected
confirmed state, requiring exact mesh-coordinate equality. Next export the
canonical dataset and its attrition/provenance manifest. Preserve the verified
selection-only bundle and its appendix curve unchanged.
