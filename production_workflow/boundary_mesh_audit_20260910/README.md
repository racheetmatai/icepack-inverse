# Production mesh boundary audit — 10 September 2026

Result: no boundary-tagging or boundary-condition assignment defect found.
This closes the previously outstanding direct inspection of the Docker mesh.

## Evidence and checks

- Copied `amundsen.msh` and `amundsen.geo` from the running container
  `xenodochial_cerf`, repository `/home/firedrake/icepack/icepack-inverse`.
- Mesh SHA256 `b414ab0ab994cd377bc827c91c94cd8ed2368915905a8e5188f043a298b9a34a`
  exactly matches `../amundsen_production_config.json`.
- 9,069 vertices, 17,653 triangles, and 497 boundary edges. Every topological
  boundary edge has exactly one physical tag; no interior edge is tagged as
  a boundary. Triangle edge multiplicities are one or two.
- Tags 1 and 3 are prescribed-velocity sections of the outer boundary.
  Tags 5–11 are seven closed internal boundaries, also prescribed velocity.
- Tags 2 and 4 are the ocean-pressure terminus sections: respectively 71 and
  9 edges (402.34 and 55.42 km). Tag 2 is an extended coastal section, not a
  one-tag-per-glacier designation.
- Visually compared `boundary_tags.png` against
  `latex_revision/baseline_source/Results/amundsen/inverse/amundsen_mesh.png`:
  the same coastal sections carry the terminus condition, including the
  intervening prescribed-velocity section and internal boundaries.

## Actual Icepack behavior

Inspected `/home/firedrake/icepack/src/icepack/solvers/flow_solver.py`,
particularly lines 248–260, and `models/ice_stream.py`, lines 45–75 and
108–127. The PETSc solver prescribes the supplied velocity on Dirichlet tags;
its ice-front tags are the complement of Dirichlet and side-wall tags among
mesh boundary markers. With the frozen settings this complement is exactly
{2,4}. `stress_ids` in the project configuration documents and validates the
partition; it is not an additional FlowSolver keyword.

The default terminus contribution uses half the difference between the
depth-integrated ice and seawater pressure terms, with submerged depth
obtained from `min(s-h,0)` and squared. No custom terminus law is supplied
by the production Invert class. No side-wall class is enabled.

The preceding code/manifest audit traced inversion, ML forward simulations,
and uniform-C simulations through the same configured `simulation_C` solver
and prescribed `u_initial` boundary field. Nonlinear-iteration retries do not
change boundary conditions. Production boundary policy and the original
whole-sector evaluation map agree. This is not a claim that every prescribed
velocity value is numerically identical to legacy runs: the approved centered
velocity interpolation and missing-value handling were updated. Frozen input
checks record 11 Dirichlet velocity DOFs filled from nearest valid pixels.

## Reproduction and scope

Run `audit_mesh.py` with NumPy and Matplotlib installed; it reads the two mesh
copies and the parent production configuration, writes `audit.json`, asserts
all checks, and draws `boundary_tags.png`. The recorded checks are topology,
hash, and tag-partition checks; the geographic comparison was visual and the
solver behavior was verified by source inspection.

No mesh, solver, scientific result, manuscript, or submission ZIP was changed.
No inversion or forward simulation was run. Docker was started at the author's
request and left running.
