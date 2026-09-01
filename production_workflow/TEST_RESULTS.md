# Production workflow test results

Date: 14 August 2026

Environment: activated `/home/firedrake/firedrake` environment in Docker
container `xenodochial_cerf`.

Command:

```bash
python -m unittest discover -s production_workflow/tests -p 'test_*.py' -v
```

Result: **58 tests passed; 0 failures; 0 errors.**

Coverage includes configuration rejection and frozen boundaries, manifest-ID
canonicalization, driving-stress/directional-feature policy, Bouguer/snow
omission, explicit bed-class CRS handling, true raster centers, rotated-grid
rejection, nearest-valid solver-only velocity fill, paired VX/VY and SOURCE
eligibility, ERRX/ERRY diagnostic behavior, valid Firedrake cell 0, revised
vertex coordinates, and loud failure for empty observation selections.

The expanded inversion/L-curve coverage includes the exact adaptive-v2 solver
and five-point base configuration; ROL state priority and native-status
agreement; 50-iteration saved-control blocks; joint `E`/`R` stability and
gradient-safety gates; independent final objective reassembly; raw,
unweighted, and weighted regularization identities; finite `C`, fixed zero
`theta`, and recomputed velocity; frozen-mesh state round-trip; Menger
curvature and scaling/order invariance; the single low/high endpoint rule;
ambiguity-triggered geometric refinements; selected-triple confirmation;
whole-triple second-round recovery after a failed confirmation; the unresolved
two-round stop; immutable interrupted-run resume; complete formal/confirmation
lineage and output-tamper rejection; and refusal to start without an exact
passing forward smoke. The integration suite exercises real Firedrake
saved-control continuation and an independent point followed by one
same-`reg_C` confirmation block.

The live-input integration run is separately preserved under
`gate1_results/gate1_preflight_20260813_d` and passed 85/85 checks.

The current source-matched whole-sector forward gate is preserved under
`gate1_results/gate1_forward_smoke_v2_20260814_a`. It passed 12/12 checks; all
28 declared outputs and its manifest ID independently verified. Manifest ID:
`sha256-json-v1-00a5d9a67a6a8992c5d7980593e34fb6469ac998cd529c74678500651ae37af3`.
Adaptive-v2 production study `gate1_lcurve_v2_20260814_a` is now running
serially under contract
`sha256-json-v1-e129f8b1b3bf4a68d559254a642b410e47f668f4a5e11d7ffa9f55b62fabe74d`.
The superseded v1 study remains preserved and must not be resumed.
