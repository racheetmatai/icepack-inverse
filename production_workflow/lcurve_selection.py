"""Deterministic selection utilities for the frozen revised L-curve protocol."""

from __future__ import annotations

import math

import numpy as np


ROL_ACCEPTANCE_BY_TERMINATION = {
    "rol_gradient": "converged_gradient",
    "rol_step": "converged_step",
}
PRACTICAL_ACCEPTANCE = "practical_er_stability"
VALID_ACCEPTANCE_BASES = {
    *ROL_ACCEPTANCE_BY_TERMINATION,
    PRACTICAL_ACCEPTANCE,
}


def is_valid_point(point: dict) -> bool:
    """Return whether a point is eligible for L-curve geometry."""
    try:
        values = [
            float(point["reg_c"]),
            float(point["misfit"]),
            float(point["unweighted_roughness"]),
            float(point["weighted_penalty"]),
            float(point["objective"]),
        ]
    except (KeyError, TypeError, ValueError):
        return False
    acceptance_basis = point.get("acceptance_basis")
    native_termination = point.get(
        "native_termination", point.get("termination")
    )
    rol_termination_is_consistent = (
        acceptance_basis not in ROL_ACCEPTANCE_BY_TERMINATION
        or native_termination == ROL_ACCEPTANCE_BY_TERMINATION[acceptance_basis]
    )
    return (
        point.get("status") == "valid"
        and acceptance_basis in VALID_ACCEPTANCE_BASES
        and rol_termination_is_consistent
        and all(math.isfinite(value) for value in values)
        and values[0] > 0.0
        and values[1] > 0.0
        and values[2] > 0.0
        and values[3] >= 0.0
        and values[4] > 0.0
    )


def valid_points(points) -> list[dict]:
    """Return valid points sorted by increasing ``reg_c`` with no duplicates."""
    selected = sorted(
        (dict(point) for point in points if is_valid_point(point)),
        key=lambda point: float(point["reg_c"]),
    )
    values = [float(point["reg_c"]) for point in selected]
    if len(values) != len(set(values)):
        raise ValueError("Valid L-curve points contain duplicate reg_C values.")
    return selected


def _normalized_log_coordinates(points: list[dict]) -> np.ndarray:
    if len(points) < 3:
        raise ValueError("At least three valid L-curve points are required.")
    coordinates = np.column_stack(
        (
            np.log10([float(point["misfit"]) for point in points]),
            np.log10(
                [float(point["unweighted_roughness"]) for point in points]
            ),
        )
    )
    spans = np.ptp(coordinates, axis=0)
    if not np.all(np.isfinite(coordinates)) or np.any(spans <= 0.0):
        raise ValueError(
            "Finite, non-degenerate log-misfit and log-roughness ranges are required."
        )
    return (coordinates - coordinates.min(axis=0)) / spans


def menger_curvature(first, middle, last) -> float:
    """Unsigned Menger curvature of three two-dimensional points."""
    first = np.asarray(first, dtype="float64")
    middle = np.asarray(middle, dtype="float64")
    last = np.asarray(last, dtype="float64")
    first_vector = middle - first
    second_vector = last - first
    twice_area = abs(
        float(
            first_vector[0] * second_vector[1]
            - first_vector[1] * second_vector[0]
        )
    )
    denominator = (
        float(np.linalg.norm(first - middle))
        * float(np.linalg.norm(middle - last))
        * float(np.linalg.norm(last - first))
    )
    if denominator == 0.0:
        return 0.0
    return 2.0 * twice_area / denominator


def curvature_table(points) -> list[dict]:
    """Compute normalized log-space curvature for each valid interior point."""
    selected = valid_points(points)
    coordinates = _normalized_log_coordinates(selected)
    output = []
    for index, (point, coordinate) in enumerate(zip(selected, coordinates)):
        curvature = None
        if 0 < index < len(selected) - 1:
            curvature = menger_curvature(
                coordinates[index - 1], coordinate, coordinates[index + 1]
            )
        output.append(
            {
                "reg_c": float(point["reg_c"]),
                "misfit": float(point["misfit"]),
                "unweighted_roughness": float(point["unweighted_roughness"]),
                "normalized_log_misfit": float(coordinate[0]),
                "normalized_log_roughness": float(coordinate[1]),
                "curvature": None if curvature is None else float(curvature),
            }
        )
    return output


def choose_maximum_curvature(table: list[dict]) -> dict:
    """Choose from a precomputed table; exact ties favor smaller ``reg_C``."""
    interior = [row for row in table if row.get("curvature") is not None]
    if not interior:
        raise ValueError("No valid interior L-curve point is available.")
    return min(
        interior,
        key=lambda row: (-float(row["curvature"]), float(row["reg_c"])),
    )


def select_corner(points) -> dict:
    """Select maximum discrete curvature; exact ties favor smaller ``reg_C``."""
    table = curvature_table(points)
    selected = choose_maximum_curvature(table)
    return {"selected": selected, "curvature_table": table}


def curvature_ambiguity(result: dict, ratio_threshold: float = 1.25) -> dict:
    """Assess whether the two largest interior curvatures are too similar.

    ``result`` is the mapping returned by :func:`select_corner`. Curvature
    candidates are ranked by decreasing curvature and then by increasing
    ``reg_c``, matching the deterministic corner tie-break. The corner is
    ambiguous exactly when ``top / second < ratio_threshold``.
    """
    if not math.isfinite(ratio_threshold) or ratio_threshold <= 1.0:
        raise ValueError("The curvature ratio threshold must be finite and > 1.")
    try:
        table = result["curvature_table"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "Curvature ambiguity requires a select_corner result."
        ) from error

    ranked = []
    for row in table:
        if row.get("curvature") is None:
            continue
        curvature = float(row["curvature"])
        reg_c = float(row["reg_c"])
        if not math.isfinite(curvature) or curvature < 0.0:
            raise ValueError("Interior curvatures must be finite and nonnegative.")
        ranked.append((curvature, reg_c, dict(row)))
    if len(ranked) < 2:
        raise ValueError(
            "At least two interior curvature candidates are required to assess "
            "ambiguity."
        )
    ranked.sort(key=lambda item: (-item[0], item[1]))
    top_curvature, _, top = ranked[0]
    second_curvature, _, second = ranked[1]
    if second_curvature == 0.0:
        ratio = 1.0 if top_curvature == 0.0 else math.inf
    else:
        ratio = top_curvature / second_curvature
    return {
        "top": top,
        "second": second,
        "ratio": float(ratio),
        "is_ambiguous": bool(ratio < ratio_threshold),
    }


def extension_side(points) -> str | None:
    """Return a side only when the corner is adjacent to that boundary."""
    selected_points = valid_points(points)
    result = select_corner(selected_points)
    selected_reg_c = float(result["selected"]["reg_c"])
    index = [float(point["reg_c"]) for point in selected_points].index(
        selected_reg_c
    )
    low_adjacent = index == 1
    high_adjacent = index == len(selected_points) - 2
    if low_adjacent and high_adjacent:
        raise ValueError(
            "A three-point curve is adjacent to both boundaries; an extension "
            "side is not uniquely determined."
        )
    if low_adjacent:
        return "low"
    if high_adjacent:
        return "high"
    return None


def geometric_refinements(points) -> dict:
    """Return the frozen midpoint refinements around the current corner."""
    selected_points = valid_points(points)
    result = select_corner(selected_points)
    selected_reg_c = float(result["selected"]["reg_c"])
    values = [float(point["reg_c"]) for point in selected_points]
    index = values.index(selected_reg_c)
    if not (0 < index < len(values) - 1):
        raise ValueError("The candidate corner requires valid neighbors on both sides.")
    low = math.sqrt(values[index - 1] * selected_reg_c)
    high = math.sqrt(selected_reg_c * values[index + 1])
    existing = set(values)
    refinements = [value for value in (low, high) if value not in existing]
    if len(refinements) != 2:
        raise ValueError("Both geometric-midpoint refinements must be new points.")
    return {
        "candidate_reg_c": selected_reg_c,
        "neighbor_reg_c": [values[index - 1], values[index + 1]],
        "refinement_reg_c": refinements,
        "curvature_table": result["curvature_table"],
    }


def corner_neighborhood(points) -> dict:
    """Return the selected corner and its two existing valid neighbors."""
    selected_points = valid_points(points)
    result = select_corner(selected_points)
    candidate = float(result["selected"]["reg_c"])
    values = [float(point["reg_c"]) for point in selected_points]
    index = values.index(candidate)
    if not (0 < index < len(values) - 1):
        raise ValueError("The candidate corner requires valid neighbors on both sides.")
    neighbors = [values[index - 1], values[index + 1]]
    return {
        "candidate_reg_c": candidate,
        "neighbor_reg_c": neighbors,
        "confirmation_reg_c": [neighbors[0], candidate, neighbors[1]],
    }
