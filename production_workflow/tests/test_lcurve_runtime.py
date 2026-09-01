import ast
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lcurve_runtime import (
    assess_practical_stability,
    classify_algorithm_state,
    native_status_matches,
    parse_rol_attempts,
)


class State:
    def __init__(self, value=3.0, gnorm=1.0, snorm=1.0, cnorm=0.0):
        self.value = value
        self.gnorm = gnorm
        self.snorm = snorm
        self.cnorm = cnorm


def block(
    index,
    misfit,
    roughness,
    gradient_norm=8e-4,
    *,
    all_finite=True,
    objective_identities_passed=True,
):
    return {
        "block": index,
        "misfit": misfit,
        "unweighted_roughness": roughness,
        "gradient_norm": gradient_norm,
        "all_finite": all_finite,
        "objective_identities_passed": objective_identities_passed,
        # Practical E/R acceptance is intentionally independent of native ROL
        # termination, which remains a separately preserved diagnostic.
        "native_termination": "iteration_limit",
    }


def assess(blocks):
    return assess_practical_stability(
        blocks,
        min_blocks=4,
        stable_transitions=2,
        relative_misfit_tolerance=0.005,
        relative_roughness_tolerance=0.005,
        gradient_safety_tolerance=0.001,
    )


class LCurveRuntimeTests(unittest.TestCase):
    def test_practical_stability_requires_two_joint_er_transitions(self):
        result = assess(
            [
                block(1, 100.0, 50.0),
                block(2, 99.0, 51.0),
                block(3, 98.6, 50.8),
                block(4, 98.3, 50.6),
            ]
        )
        self.assertTrue(result["accepted"])
        self.assertEqual(result["acceptance_basis"], "practical_er_stability")
        self.assertEqual(result["evaluated_block_count"], 4)
        self.assertEqual(result["required_stable_transitions"], 2)
        self.assertTrue(result["gradient_safety_passed"])
        self.assertEqual(len(result["transitions"]), 3)
        self.assertTrue(all(row["joint_passed"] for row in result["transitions"][-2:]))

    def test_practical_stability_rejects_if_only_one_component_is_stable(self):
        stable_e_only = assess(
            [
                block(1, 100.0, 50.0),
                block(2, 99.0, 51.0),
                block(3, 98.6, 50.6),
                block(4, 98.3, 51.2),
            ]
        )
        self.assertFalse(stable_e_only["accepted"])
        self.assertIsNone(stable_e_only["acceptance_basis"])
        stable_r_only = assess(
            [
                block(1, 100.0, 50.0),
                block(2, 99.0, 51.0),
                block(3, 98.0, 50.8),
                block(4, 97.0, 50.6),
            ]
        )
        self.assertFalse(stable_r_only["accepted"])
        self.assertIsNone(stable_r_only["acceptance_basis"])

    def test_practical_stability_requires_minimum_blocks_gradient_and_identities(self):
        stable = [
            block(1, 100.0, 50.0),
            block(2, 99.0, 51.0),
            block(3, 98.6, 50.8),
            block(4, 98.3, 50.6),
        ]
        self.assertFalse(assess(stable[:3])["accepted"])
        unsafe_gradient = [dict(row) for row in stable]
        unsafe_gradient[-1]["gradient_norm"] = 1.001e-3
        self.assertFalse(assess(unsafe_gradient)["accepted"])
        broken_identity = [dict(row) for row in stable]
        broken_identity[-1]["objective_identities_passed"] = False
        self.assertFalse(assess(broken_identity)["accepted"])
        nonfinite = [dict(row) for row in stable]
        nonfinite[-1]["misfit"] = float("nan")
        nonfinite[-1]["all_finite"] = False
        self.assertFalse(assess(nonfinite)["accepted"])

    def test_practical_stability_tolerance_is_inclusive(self):
        result = assess(
            [
                block(1, 101.0, 49.0),
                block(2, 100.0, 50.0),
                block(3, 99.5, 50.25),
                block(4, 99.0025, 50.50125),
            ]
        )
        self.assertTrue(result["accepted"])
        for transition in result["transitions"][-2:]:
            self.assertLessEqual(transition["relative_misfit_change"], 0.005)
            self.assertLessEqual(transition["relative_roughness_change"], 0.005)

    def test_state_classification_matches_rol_priority(self):
        gradient = classify_algorithm_state(
            State(gnorm=1e-5, snorm=1e-6),
            gradient_tolerance=1e-4,
            step_tolerance=5e-3,
        )
        self.assertEqual(gradient["termination"], "converged_gradient")
        step = classify_algorithm_state(
            State(gnorm=1e-2, snorm=1e-3),
            gradient_tolerance=1e-4,
            step_tolerance=5e-3,
        )
        self.assertEqual(step["termination"], "converged_step")
        limit = classify_algorithm_state(
            State(gnorm=1e-2, snorm=1e-2),
            gradient_tolerance=1e-4,
            step_tolerance=5e-3,
        )
        self.assertEqual(limit["termination"], "iteration_limit")
        nonfinite = classify_algorithm_state(
            State(value=float("inf")),
            gradient_tolerance=1e-4,
            step_tolerance=5e-3,
        )
        self.assertEqual(nonfinite["termination"], "nonfinite_state")

    def test_native_log_parser_separates_continuation_attempts(self):
        log = """
JOG_ROL_ATTEMPT_BEGIN 1
  0     1.000000e+00   1.000000e+00
  300   5.000000e-01   1.000000e-02
Optimization Terminated with Status: Iteration Limit Exceeded
JOG_ROL_ATTEMPT_END 1
JOG_ROL_ATTEMPT_BEGIN 2
  0     5.000000e-01   1.000000e-02
  12    4.000000e-01   9.000000e-05
Optimization Terminated with Status: Converged
JOG_ROL_ATTEMPT_END 2
"""
        parsed = parse_rol_attempts(log)
        self.assertEqual(parsed[0]["rol_last_iteration"], 300)
        self.assertEqual(parsed[0]["rol_status_text"], "Iteration Limit Exceeded")
        self.assertEqual(parsed[1]["rol_last_iteration"], 12)
        self.assertEqual(parsed[1]["rol_status_text"], "Converged")
        self.assertTrue(all(row["end_marker_present"] for row in parsed))

    def test_native_status_must_agree_with_exposed_state(self):
        self.assertTrue(
            native_status_matches("converged_gradient", "Converged")
        )
        self.assertTrue(
            native_status_matches("converged_step", "Step Tolerance Met")
        )
        self.assertTrue(
            native_status_matches("iteration_limit", "Iteration Limit Exceeded")
        )
        self.assertFalse(
            native_status_matches("converged_gradient", "Iteration Limit Exceeded")
        )
        self.assertFalse(native_status_matches("converged_gradient", None))

    def test_invert_c_api_supports_explicit_continuation_control(self):
        source = (
            Path(__file__).resolve().parents[1] / "src" / "invert_c_theta.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)
        function = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "invert_C"
        )
        arguments = [argument.arg for argument in function.args.args]
        self.assertIn("initial_control", arguments)
        self.assertIn("return_estimator", arguments)


if __name__ == "__main__":
    unittest.main()
