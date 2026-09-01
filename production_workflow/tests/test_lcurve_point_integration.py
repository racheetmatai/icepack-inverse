from pathlib import Path
import json
import tempfile
import unittest

import firedrake
import numpy as np

from production_workflow.lcurve_runtime import run_lcurve_point, sha256_file
from src.invert_c_theta import Invert


class ToyPreflight:
    def __init__(self, config, repo_root, output_dir):
        self.config = config
        self.output_dir = Path(output_dir)

    def run(self, *, level, config_path):
        self.output_dir.mkdir(parents=True)
        return {"status": "pass", "manifest_id": "toy-input-preflight"}

    def build_invert(self, *, reg_c):
        mesh = firedrake.UnitSquareMesh(2, 2)
        Q = firedrake.FunctionSpace(mesh, "CG", 1)
        V = firedrake.VectorFunctionSpace(mesh, "CG", 1)
        x, y = firedrake.SpatialCoordinate(mesh)
        target = firedrake.as_vector((x + 2.0 * y, x + 2.0 * y))
        object_ = Invert.__new__(Invert)
        object_.mesh = mesh
        object_.Q = Q
        object_.V = V
        object_.degree = 1
        object_.N = 1
        object_.area = float(firedrake.assemble(1.0 * firedrake.dx(mesh)))
        object_.C = firedrake.Function(Q)
        setattr(object_, "\u03b8", firedrake.Function(Q))
        object_.simulation_C = lambda control: firedrake.interpolate(
            firedrake.as_vector((control, control)), V
        )
        object_.loss_functional_nosigma = lambda state: (
            0.5 * firedrake.inner(state - target, state - target) * firedrake.dx(mesh)
        )
        length = firedrake.Constant(7.5e3)
        object_.regularization_C_grad = lambda control: (
            0.5
            / object_.area
            * (length / reg_c) ** 2
            * firedrake.inner(firedrake.grad(control), firedrake.grad(control))
            * firedrake.dx(mesh)
        )
        self.config["expected_counts"]["cg2_scalar_dofs"] = Q.dim()
        return object_

    def source_identity(self):
        return {"toy": "source"}

    def environment(self):
        return {"toy": "environment"}


class LCurvePointIntegrationTests(unittest.TestCase):
    def test_one_point_captures_native_status_metrics_and_state(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config_path = root / "config.json"
            config = {
                "physics": {
                    "initial_log_friction_c": 0.0,
                    "initial_log_fluidity_theta": 0.0,
                },
                "inversion": {
                    "gradient_tolerance": 1e-7,
                    "step_tolerance": 1e-10,
                    "lcurve_block_iterations": 50,
                    "lcurve_min_blocks": 4,
                    "lcurve_max_blocks": 6,
                    "lcurve_stable_transitions": 2,
                    "lcurve_relative_misfit_tolerance": 0.005,
                    "lcurve_relative_roughness_tolerance": 0.005,
                    "lcurve_gradient_safety_tolerance": 0.001,
                },
                "expected_counts": {
                    "cg2_scalar_dofs": 0,
                    "selected_observations": 1,
                },
                "inputs": {"mesh": {"sha256": "toy-mesh-sha256"}},
            }
            config_path.write_text(
                json.dumps(config, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            output = root / "point"
            manifest = run_lcurve_point(
                config=config,
                config_path=config_path,
                repo_root=root,
                output_dir=output,
                reg_c=1e6,
                preflight_class=ToyPreflight,
            )
            self.assertEqual(
                manifest["schema"], "jog-production-lcurve-point-v2"
            )
            self.assertEqual(manifest["run_kind"], "independent")
            self.assertEqual(
                manifest["status"], "valid", manifest.get("exception")
            )
            self.assertTrue(manifest["solver_log_crosscheck_passed"])
            self.assertEqual(len(manifest["blocks"]), 4)
            self.assertEqual(
                [block["start"] for block in manifest["blocks"]],
                [
                    "independent_C_zero",
                    "same_reg_c_saved_control",
                    "same_reg_c_saved_control",
                    "same_reg_c_saved_control",
                ],
            )
            self.assertTrue(all(block["iteration_limit"] == 50 for block in manifest["blocks"]))
            self.assertTrue(
                all(
                    block["native_status_matches_state"]
                    for block in manifest["blocks"]
                )
            )
            for index, block_record in enumerate(manifest["blocks"], start=1):
                self.assertEqual(block_record["block"], index)
                state_path = output / block_record["C_path"]
                self.assertTrue(state_path.is_file())
                self.assertEqual(
                    sha256_file(state_path), block_record["C_sha256"]
                )
                self.assertTrue(block_record["metrics"]["all_finite"])
                self.assertTrue(
                    block_record["metrics"]["weighted_penalty_identity"]
                )
                self.assertTrue(
                    block_record["metrics"][
                        "rol_objective_matches_reassembled"
                    ]
                )
            self.assertTrue(manifest["stability"]["passed"])
            self.assertGreaterEqual(
                manifest["stability"]["consecutive_stable_transitions"], 2
            )
            self.assertIn(
                manifest["acceptance_basis"],
                {"rol_gradient", "rol_step", "practical_er_stability"},
            )
            self.assertIn(
                manifest["native_termination"],
                {"converged_gradient", "converged_step", "iteration_limit"},
            )
            self.assertTrue(manifest["metrics"]["weighted_penalty_identity"])
            self.assertTrue(
                manifest["metrics"]["rol_objective_matches_reassembled"]
            )
            self.assertGreater(manifest["metrics"]["raw_gradient_integral"], 0.0)
            self.assertTrue(manifest["fields"]["velocity_finite"])
            schema = json.loads(
                (output / "state_schema.json").read_text(encoding="utf-8")
            )
            self.assertEqual(schema["mesh_sha256"], "toy-mesh-sha256")
            self.assertTrue(np.isfinite(np.load(output / "state_C.npy")).all())
            self.assertTrue((output / "blocks.csv").is_file())
            self.assertFalse((output / "attempts.csv").exists())

            parent_path = output / "point_manifest.json"
            confirmation_output = root / "confirmation"
            confirmation = run_lcurve_point(
                config=config,
                config_path=config_path,
                repo_root=root,
                output_dir=confirmation_output,
                reg_c=1e6,
                preflight_class=ToyPreflight,
                parent_point_manifest_path=parent_path,
                confirmation_round=1,
            )
            self.assertEqual(confirmation["run_kind"], "confirmation")
            self.assertEqual(confirmation["status"], "valid")
            self.assertEqual(confirmation["confirmation_round"], 1)
            self.assertEqual(len(confirmation["blocks"]), 1)
            self.assertEqual(
                confirmation["blocks"][0]["start"],
                "verified_parent_same_reg_c_control",
            )
            self.assertEqual(
                confirmation["blocks"][0]["block"],
                len(manifest["blocks"]) + 1,
            )
            self.assertEqual(
                confirmation["cumulative_block_count"],
                len(manifest["blocks"]) + 1,
            )
            self.assertEqual(
                confirmation["parent_point_manifest"]["manifest_id"],
                manifest["manifest_id"],
            )
            self.assertEqual(
                confirmation["parent_point_manifest"]["manifest_sha256"],
                sha256_file(parent_path),
            )
            self.assertTrue(confirmation["confirmation"]["passed"])
            self.assertEqual(
                confirmation["confirmation"], confirmation["stability"]
            )
            self.assertLessEqual(
                confirmation["confirmation"]["relative_misfit_change"],
                0.005,
            )
            self.assertLessEqual(
                confirmation["confirmation"]["relative_roughness_change"],
                0.005,
            )


if __name__ == "__main__":
    unittest.main()
