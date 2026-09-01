#!/usr/bin/env python3
"""Small local optimization used only to inspect the installed PyROL state API."""

import json

import firedrake
from icepack.statistics import MaximumProbabilityEstimator, StatisticsProblem


mesh = firedrake.UnitSquareMesh(2, 2)
Q = firedrake.FunctionSpace(mesh, "CG", 1)
control = firedrake.Function(Q)


def simulation(value):
    return value


def loss(value):
    return 0.5 * (value - 1.0) ** 2 * firedrake.dx(mesh)


def regularization(value):
    return 1.0e-12 * value**2 * firedrake.dx(mesh)


problem = StatisticsProblem(simulation, loss, regularization, control)
estimator = MaximumProbabilityEstimator(
    problem,
    max_iterations=0,
    gradient_tolerance=1.0e-12,
    step_tolerance=1.0e-12,
)
estimator.solve()
state = estimator._solver.getAlgorithmState()
result = {"state_type": str(type(state)), "state_repr": repr(state)}
for name in dir(state):
    if name.startswith("_"):
        continue
    try:
        value = getattr(state, name)
    except Exception as error:
        result[name] = f"ERROR: {error}"
        continue
    if isinstance(value, (str, int, float, bool, type(None))):
        result[name] = value
    else:
        result[name] = f"<{type(value).__name__}>"
solver = estimator._solver.solver
result["solver_type"] = str(type(solver))
result["solver_members"] = [
    name for name in dir(solver) if not name.startswith("_")
]
for name in (
    "iter", "nfval", "ngrad", "statusFlag", "flag", "searchSize",
    "getStatus", "getOutput", "getAlgorithmState",
):
    for prefix, value in (("state", state), ("solver", solver)):
        try:
            candidate = getattr(value, name)
            if callable(candidate) and name.startswith("get"):
                candidate = candidate()
            result[f"candidate:{prefix}.{name}"] = str(candidate)
        except Exception as error:
            result[f"candidate:{prefix}.{name}"] = f"ERROR: {error}"
print(json.dumps(result, indent=2, sort_keys=True))
