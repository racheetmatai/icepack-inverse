import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lcurve_selection import (
    choose_maximum_curvature,
    corner_neighborhood,
    curvature_ambiguity,
    extension_side,
    geometric_refinements,
    is_valid_point,
    menger_curvature,
    select_corner,
    valid_points,
)


def point(
    reg_c,
    x,
    y,
    *,
    acceptance_basis="practical_er_stability",
    native_termination="iteration_limit",
):
    misfit = 10.0**x
    roughness = 10.0**y
    penalty = roughness / reg_c**2
    return {
        "status": "valid",
        "reg_c": reg_c,
        "misfit": misfit,
        "unweighted_roughness": roughness,
        "weighted_penalty": penalty,
        "objective": misfit + penalty,
        "acceptance_basis": acceptance_basis,
        "native_termination": native_termination,
    }


class LCurveSelectionTests(unittest.TestCase):
    def test_invalid_status_and_nonpositive_geometry_are_excluded(self):
        self.assertTrue(is_valid_point(point(0.1, 1.0, 1.0)))
        invalid_status = point(0.1, 1.0, 1.0)
        invalid_status["status"] = "invalid"
        self.assertFalse(is_valid_point(invalid_status))
        missing_basis = point(0.1, 1.0, 1.0)
        missing_basis.pop("acceptance_basis")
        self.assertFalse(is_valid_point(missing_basis))
        inconsistent_rol = point(
            0.1,
            1.0,
            1.0,
            acceptance_basis="rol_gradient",
            native_termination="iteration_limit",
        )
        self.assertFalse(is_valid_point(inconsistent_rol))
        accepted_by_rol = point(
            0.1,
            1.0,
            1.0,
            acceptance_basis="rol_gradient",
            native_termination="converged_gradient",
        )
        self.assertTrue(is_valid_point(accepted_by_rol))
        bad = point(0.1, 1.0, 1.0)
        bad["misfit"] = 0.0
        self.assertFalse(is_valid_point(bad))

    def test_valid_points_sort_and_reject_duplicate_reg_c(self):
        points = [point(1.0, 0.0, 1.0), point(0.1, 1.0, 0.0)]
        self.assertEqual([row["reg_c"] for row in valid_points(points)], [0.1, 1.0])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            valid_points(points + [point(0.1, 0.5, 0.5)])

    def test_known_right_angle_selects_middle(self):
        points = [
            point(0.01, 0.0, 2.0),
            point(0.1, 0.0, 1.0),
            point(1.0, 1.0, 1.0),
            point(10.0, 2.0, 1.0),
        ]
        selected = select_corner(points)["selected"]
        self.assertEqual(selected["reg_c"], 0.1)
        self.assertGreater(selected["curvature"], 0.0)

    def test_menger_curvature_known_geometry(self):
        self.assertAlmostEqual(
            menger_curvature((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)),
            math.sqrt(2.0),
        )
        self.assertEqual(
            menger_curvature((0.0, 0.0), (1.0, 0.0), (2.0, 0.0)),
            0.0,
        )

    def test_selection_is_invariant_to_input_order_and_axis_scaling(self):
        points = [
            point(0.01, 0.0, 2.0),
            point(0.1, 0.0, 1.0),
            point(1.0, 1.0, 1.0),
            point(10.0, 2.0, 1.0),
        ]
        baseline = select_corner(points)["selected"]["reg_c"]
        scaled = []
        for original in reversed(points):
            transformed = dict(original)
            transformed["misfit"] *= 1e6
            transformed["unweighted_roughness"] *= 1e-9
            transformed["weighted_penalty"] *= 1e-9
            transformed["objective"] = (
                transformed["misfit"] + transformed["weighted_penalty"]
            )
            scaled.append(transformed)
        self.assertEqual(select_corner(scaled)["selected"]["reg_c"], baseline)

    def test_exact_curvature_tie_favors_smaller_reg_c(self):
        table = [
            {"reg_c": 0.01, "curvature": None},
            {"reg_c": 0.1, "curvature": 2.0},
            {"reg_c": 1.0, "curvature": 2.0},
            {"reg_c": 10.0, "curvature": None},
        ]
        self.assertEqual(choose_maximum_curvature(table)["reg_c"], 0.1)

    def test_endpoint_proximity_requests_only_one_extension_side(self):
        low = [
            point(0.01, 0.0, 2.0),
            point(0.1, 0.0, 1.0),
            point(1.0, 1.0, 1.0),
            point(10.0, 2.0, 1.0),
        ]
        self.assertEqual(extension_side(low), "low")
        mirrored = [
            point(0.01, 0.0, 2.0),
            point(0.1, 0.5, 1.5),
            point(1.0, 1.0, 1.0),
            point(10.0, 2.0, 1.0),
        ]
        self.assertEqual(extension_side(mirrored), "high")

    def test_three_point_curve_cannot_choose_a_unique_extension_side(self):
        points = [
            point(0.01, 0.0, 2.0),
            point(0.1, 0.0, 1.0),
            point(1.0, 1.0, 1.0),
        ]
        with self.assertRaisesRegex(ValueError, "both boundaries"):
            extension_side(points)

    def test_curvature_ambiguity_uses_strict_top_to_second_ratio(self):
        def result(top, second):
            return {
                "selected": {"reg_c": 0.05, "curvature": top},
                "curvature_table": [
                    {"reg_c": 0.01, "curvature": None},
                    {"reg_c": 0.02, "curvature": second},
                    {"reg_c": 0.05, "curvature": top},
                    {"reg_c": 0.1, "curvature": 0.1},
                    {"reg_c": 0.2, "curvature": None},
                ],
            }

        ambiguous = curvature_ambiguity(result(1.249, 1.0), 1.25)
        self.assertTrue(ambiguous["is_ambiguous"])
        self.assertAlmostEqual(ambiguous["ratio"], 1.249)
        self.assertEqual(ambiguous["top"]["reg_c"], 0.05)
        self.assertEqual(ambiguous["second"]["reg_c"], 0.02)
        boundary = curvature_ambiguity(result(1.25, 1.0), 1.25)
        self.assertFalse(boundary["is_ambiguous"])

    def test_curvature_ambiguity_requires_two_interior_candidates(self):
        result = {
            "selected": {"reg_c": 0.02, "curvature": 1.0},
            "curvature_table": [
                {"reg_c": 0.01, "curvature": None},
                {"reg_c": 0.02, "curvature": 1.0},
                {"reg_c": 0.05, "curvature": None},
            ],
        }
        with self.assertRaisesRegex(ValueError, "two interior"):
            curvature_ambiguity(result)

    def test_refinements_are_geometric_midpoints(self):
        points = [
            point(0.01, 0.0, 2.0),
            point(0.1, 0.0, 1.0),
            point(1.0, 1.0, 1.0),
            point(10.0, 2.0, 1.0),
        ]
        result = geometric_refinements(points)
        self.assertEqual(result["candidate_reg_c"], 0.1)
        self.assertAlmostEqual(result["refinement_reg_c"][0], math.sqrt(0.001))
        self.assertAlmostEqual(result["refinement_reg_c"][1], math.sqrt(0.1))

    def test_confirmation_neighborhood_uses_existing_immediate_neighbors(self):
        points = [
            point(0.01, 0.0, 2.0),
            point(0.02, 0.0, 1.5),
            point(0.05, 0.0, 1.0),
            point(0.1, 0.5, 1.0),
            point(0.2, 1.0, 1.0),
        ]
        result = corner_neighborhood(points)
        self.assertEqual(result["candidate_reg_c"], 0.05)
        self.assertEqual(result["neighbor_reg_c"], [0.02, 0.1])
        self.assertEqual(result["confirmation_reg_c"], [0.02, 0.05, 0.1])

    def test_degenerate_axis_fails_loudly(self):
        points = [
            point(0.01, 1.0, 0.0),
            point(0.1, 1.0, 1.0),
            point(1.0, 1.0, 2.0),
        ]
        with self.assertRaisesRegex(ValueError, "non-degenerate"):
            select_corner(points)


if __name__ == "__main__":
    unittest.main()
