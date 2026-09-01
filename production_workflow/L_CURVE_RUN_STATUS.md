# Production L-curve run status

Last recorded: 14 August 2026.

## Current state

The approved adaptive-v2 study is running serially in Docker.

- Run ID: `gate1_lcurve_v2_20260814_a`
- Docker path:
  `/home/firedrake/icepack/icepack-inverse/production_runs/gate1_lcurve_v2_20260814_a`
- Started: `2026-08-14T23:14:40.095419+00:00`
- Contract schema: `jog-production-lcurve-contract-v2`
- Contract ID:
  `sha256-json-v1-e129f8b1b3bf4a68d559254a642b410e47f668f4a5e11d7ffa9f55b62fabe74d`
- Current phase at launch verification: base grid, first independent point
  `reg_C=0.01`, attempt 1 (`regc_0p01_312b95ee_attempt01`).
- Execution check: one controller and exactly one point child were active; no
  point-level parallelism was present.
- No v2 `reg_C` has been selected yet. Do not report the historical `0.005`
  diagnostic as part of this curve.

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
- Run one inversion process at a time; do not parallelize points on this
  machine.
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

Let `gate1_lcurve_v2_20260814_a` proceed serially under its immutable contract.
Do not edit any contract-locked source/configuration file, start points in
parallel, resume the superseded v1 study, or select a corner early. After the
adaptive study completes, independently verify the complete formal-point and
confirmation lineage before recording the selected `reg_C` and definitive
inversion.
