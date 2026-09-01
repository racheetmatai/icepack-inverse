import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lcurve_orchestrator import reg_c_slug, run_lcurve_study
from lcurve_runtime import canonical_identifier, sha256_file


class FakePreflight:
    source_file = None

    def __init__(self, config, repo_root, output_dir):
        self.config = config

    def source_identity(self):
        path = Path(self.source_file).resolve()
        return {str(path): sha256_file(path)}

    def environment(self):
        return {
            "docker_image_hint": "test-image",
            "executable": sys.executable,
            "git_head": "test-head",
            "hostname": "test-host",
            "platform": "test-platform",
            "python": sys.version,
            "versions": {"test": "1"},
        }


def configuration():
    return {
        "physics": {"initial_log_friction_c": 0.0},
        "inversion": {
            "gradient_tolerance": 1e-4,
            "step_tolerance": 5e-3,
            "lcurve_block_iterations": 50,
            "lcurve_min_blocks": 4,
            "lcurve_max_blocks": 6,
            "lcurve_stable_transitions": 2,
            "lcurve_relative_misfit_tolerance": 0.005,
            "lcurve_relative_roughness_tolerance": 0.005,
            "lcurve_gradient_safety_tolerance": 0.001,
            "lcurve_base_reg_c": [0.01, 0.02, 0.05, 0.1, 0.2],
            "lcurve_low_extension_reg_c": 0.005,
            "lcurve_high_extension_reg_c": 0.5,
            "lcurve_curvature_ambiguity_ratio": 1.25,
            "lcurve_max_refinement_rounds": 1,
            "lcurve_max_confirmation_rounds": 2,
            "lcurve_max_formal_points": 8,
        },
        "inputs": {},
        "frozen_design": {},
    }


def write_valid_point(
    kwargs,
    *,
    config,
    config_path,
    root,
    corner,
    metric_profile=None,
):
    reg_c = float(kwargs["reg_c"])
    point_dir = kwargs["point_root"] / kwargs["run_id"]
    point_dir.mkdir()
    artifact = point_dir / "artifact.txt"
    artifact.write_text(f"reg_C={reg_c:.17g}\n", encoding="utf-8")
    if metric_profile is None:
        coordinate = math.log10(reg_c / corner)
        misfit = 10.0 ** max(coordinate, 0.0)
        roughness = 10.0 ** max(-coordinate, 0.0)
    else:
        misfit, roughness = metric_profile(reg_c)
    penalty = roughness / reg_c**2
    environment = FakePreflight(config, root, root).environment()
    source = FakePreflight(config, root, root).source_identity()
    preflight_dir = point_dir / "input_preflight"
    preflight_dir.mkdir()
    preflight = {
        "schema": "jog-production-preflight-manifest-v1",
        "status": "pass",
        "hard_failure_count": 0,
        "config_sha256": sha256_file(config_path),
        "source_sha256": source,
        "checks": [],
    }
    preflight["manifest_id"] = canonical_identifier(preflight)
    preflight_path = preflight_dir / "preflight_manifest.json"
    preflight_path.write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metrics = {
        "misfit": misfit,
        "raw_gradient_integral": roughness,
        "unweighted_roughness": roughness,
        "weighted_penalty": penalty,
        "objective": misfit + penalty,
        "all_finite": True,
        "weighted_penalty_identity": True,
        "rol_objective_matches_reassembled": True,
    }
    blocks = []
    output_sha256 = {
        "artifact.txt": sha256_file(artifact),
        "input_preflight/preflight_manifest.json": sha256_file(preflight_path),
    }
    for block_number in range(1, 5):
        state_path = point_dir / f"block_{block_number:02d}_C.npy"
        state_path.write_bytes(f"reg_C={reg_c};block={block_number}\n".encode())
        relative = None if block_number == 1 else 0.0
        blocks.append(
            {
                "block": block_number,
                "attempt": block_number,
                "iteration_limit": 50,
                "start": (
                    "independent_C_zero"
                    if block_number == 1
                    else "same_reg_c_saved_control"
                ),
                "native_termination": "iteration_limit",
                "objective": metrics["objective"],
                "gradient_norm": 8e-4,
                "step_norm": 1e-2,
                "constraint_norm": 0.0,
                "metrics": dict(metrics),
                "relative_misfit_change": relative,
                "relative_roughness_change": relative,
                "C_path": state_path.name,
                "C_sha256": sha256_file(state_path),
                "rol_last_iteration": 50,
                "rol_status_text": "Iteration Limit Exceeded",
                "end_marker_present": True,
                "native_status_matches_state": True,
            }
        )
        output_sha256[state_path.name] = sha256_file(state_path)
    manifest = {
        "schema": "jog-production-lcurve-point-v2",
        "run_kind": "independent",
        "status": "valid",
        "native_termination": "iteration_limit",
        "acceptance_basis": "practical_er_stability",
        "reg_c": reg_c,
        "config_sha256": sha256_file(config_path),
        "source_sha256": source,
        "environment": environment,
        "input_preflight_manifest_id": preflight["manifest_id"],
        "solver_log_crosscheck_passed": True,
        "blocks": blocks,
        "cumulative_block_count": len(blocks),
        "stability": {
            "passed": True,
            "acceptance_basis": "practical_er_stability",
            "minimum_blocks": 4,
            "observed_blocks": 4,
            "stable_transitions_required": 2,
            "consecutive_stable_transitions": 3,
            "terminal_gradient_norm": 8e-4,
            "gradient_safe": True,
            "finite_and_objective_consistent": True,
            "transition_evidence": [],
        },
        "metrics": metrics,
        "fields": {
            "C_finite": True,
            "theta_finite": True,
            "theta_minimum": 0.0,
            "theta_maximum": 0.0,
            "velocity_finite": True,
        },
        "output_sha256": output_sha256,
    }
    manifest["manifest_id"] = canonical_identifier(manifest)
    (point_dir / "point_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


def write_confirmation(
    kwargs,
    *,
    config,
    config_path,
    root,
    metric_transform=None,
    force_pass=True,
):
    reg_c = float(kwargs["reg_c"])
    round_number = int(kwargs["confirmation_round"])
    parent_path = Path(kwargs["parent_point_manifest_path"])
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    parent_metrics = parent["metrics"]
    if metric_transform is None:
        misfit = float(parent_metrics["misfit"])
        roughness = float(parent_metrics["unweighted_roughness"])
    else:
        misfit, roughness = metric_transform(
            reg_c, round_number, parent_metrics
        )
    penalty = roughness / reg_c**2
    metrics = {
        "misfit": misfit,
        "raw_gradient_integral": roughness,
        "unweighted_roughness": roughness,
        "weighted_penalty": penalty,
        "objective": misfit + penalty,
        "all_finite": True,
        "weighted_penalty_identity": True,
        "rol_objective_matches_reassembled": True,
    }
    delta_e = abs(misfit - parent_metrics["misfit"]) / abs(
        parent_metrics["misfit"]
    )
    delta_r = abs(roughness - parent_metrics["unweighted_roughness"]) / abs(
        parent_metrics["unweighted_roughness"]
    )
    passed = bool(force_pass and delta_e <= 0.005 and delta_r <= 0.005)
    point_dir = kwargs["point_root"] / kwargs["run_id"]
    point_dir.mkdir()
    state_path = point_dir / "confirmation_C.npy"
    state_path.write_bytes(
        f"reg_C={reg_c};confirmation_round={round_number}\n".encode()
    )
    artifact = point_dir / "artifact.txt"
    artifact.write_text(
        f"reg_C={reg_c:.17g};confirmation_round={round_number}\n",
        encoding="utf-8",
    )
    environment = FakePreflight(config, root, root).environment()
    source = FakePreflight(config, root, root).source_identity()
    preflight_dir = point_dir / "input_preflight"
    preflight_dir.mkdir()
    preflight = {
        "schema": "jog-production-preflight-manifest-v1",
        "status": "pass",
        "hard_failure_count": 0,
        "config_sha256": sha256_file(config_path),
        "source_sha256": source,
        "checks": [],
    }
    preflight["manifest_id"] = canonical_identifier(preflight)
    preflight_path = preflight_dir / "preflight_manifest.json"
    preflight_path.write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    parent_count = int(parent.get("cumulative_block_count", len(parent["blocks"])))
    block_number = parent_count + 1
    block_record = {
        "block": block_number,
        "attempt": 1,
        "iteration_limit": 50,
        "start": "verified_parent_same_reg_c_control",
        "native_termination": "iteration_limit",
        "objective": metrics["objective"],
        "gradient_norm": 8e-4,
        "step_norm": 1e-2,
        "constraint_norm": 0.0,
        "metrics": metrics,
        "relative_misfit_change": delta_e,
        "relative_roughness_change": delta_r,
        "C_path": state_path.name,
        "C_sha256": sha256_file(state_path),
        "rol_last_iteration": 50,
        "rol_status_text": "Iteration Limit Exceeded",
        "end_marker_present": True,
        "native_status_matches_state": True,
    }
    confirmation = {
        "passed": passed,
        "acceptance_basis": (
            "practical_er_stability" if passed else None
        ),
        "confirmation": True,
        "relative_misfit_change": delta_e,
        "relative_roughness_change": delta_r,
        "relative_misfit_tolerance": 0.005,
        "relative_roughness_tolerance": 0.005,
        "gradient_safety_tolerance": 0.001,
        "terminal_gradient_norm": 8e-4,
        "gradient_safe": True,
        "finite_and_objective_consistent": True,
    }
    manifest = {
        "schema": "jog-production-lcurve-point-v2",
        "run_kind": "confirmation",
        "status": "valid" if passed else "invalid",
        "native_termination": "iteration_limit",
        "acceptance_basis": (
            "practical_er_stability" if passed else None
        ),
        "reg_c": reg_c,
        "confirmation_round": round_number,
        "cumulative_block_count": block_number,
        "parent_point_manifest": {
            "path": str(parent_path),
            "manifest_id": parent["manifest_id"],
            "manifest_sha256": sha256_file(parent_path),
        },
        "confirmation": confirmation,
        "config_sha256": sha256_file(config_path),
        "source_sha256": source,
        "environment": environment,
        "input_preflight_manifest_id": preflight["manifest_id"],
        "solver_log_crosscheck_passed": True,
        "blocks": [block_record],
        "stability": confirmation,
        "metrics": metrics,
        "fields": {
            "C_finite": True,
            "theta_finite": True,
            "theta_minimum": 0.0,
            "theta_maximum": 0.0,
            "velocity_finite": True,
        },
        "output_sha256": {
            "artifact.txt": sha256_file(artifact),
            state_path.name: sha256_file(state_path),
            "input_preflight/preflight_manifest.json": sha256_file(
                preflight_path
            ),
        },
    }
    manifest["manifest_id"] = canonical_identifier(manifest)
    (point_dir / "point_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


class LCurveOrchestratorTests(unittest.TestCase):
    def setup_case(self, root):
        config = configuration()
        config_path = root / "config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        source_path = root / "source.py"
        source_path.write_text("SCIENTIFIC_SOURCE = True\n", encoding="utf-8")
        FakePreflight.source_file = source_path
        return config, config_path

    def run_case(
        self,
        root,
        *,
        corner,
        launcher=None,
        confirmation_launcher=None,
        metric_profile=None,
        resume=False,
    ):
        config, config_path = self.setup_case(root)
        if launcher is None:
            launcher = lambda **kwargs: write_valid_point(
                kwargs,
                config=config,
                config_path=config_path,
                root=root,
                corner=corner,
                metric_profile=metric_profile,
            )
        if confirmation_launcher is None:
            confirmation_launcher = lambda **kwargs: write_confirmation(
                kwargs,
                config=config,
                config_path=config_path,
                root=root,
            )
        return run_lcurve_study(
            config=config,
            config_path=config_path,
            repo_root=root,
            output_dir=root / "study",
            preflight_class=FakePreflight,
            resume=resume,
            point_launcher=launcher,
            confirmation_launcher=confirmation_launcher,
            create_plot=False,
            require_forward_smoke=False,
        )

    def test_reg_c_slug_is_stable_and_distinguishes_nearby_values(self):
        self.assertEqual(reg_c_slug(0.05), reg_c_slug(0.05))
        self.assertNotEqual(reg_c_slug(0.05), reg_c_slug(0.05000000000000001))
        with self.assertRaises(ValueError):
            reg_c_slug(0.0)

    def test_production_launch_requires_smoke_before_creating_study(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config, config_path = self.setup_case(root)
            with self.assertRaisesRegex(ValueError, "forward-smoke-dir"):
                run_lcurve_study(
                    config=config,
                    config_path=config_path,
                    repo_root=root,
                    output_dir=root / "study",
                    preflight_class=FakePreflight,
                    create_plot=False,
                )
            self.assertFalse((root / "study").exists())

    def test_five_base_points_and_one_confirmation_round_complete(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = self.run_case(root, corner=0.05)
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["extension_reg_c"], [])
            self.assertEqual(
                manifest["base_reg_c"], [0.01, 0.02, 0.05, 0.1, 0.2]
            )
            self.assertEqual(manifest["refinement_reg_c"], [])
            self.assertEqual(manifest["formal_point_count"], 5)
            self.assertEqual(manifest["point_count"], 5)
            self.assertEqual(manifest["confirmation_manifest_count"], 3)
            self.assertEqual(len(manifest["confirmation_rounds"]), 1)
            self.assertEqual(manifest["selected_reg_c"], 0.05)
            state = json.loads(
                (root / "study" / "run_state.json").read_text(encoding="utf-8")
            )
            independent = [
                request["reg_c"]
                for request in state["requests"]
                if request.get("run_kind", "independent") == "independent"
            ]
            self.assertEqual(independent, [0.01, 0.02, 0.05, 0.1, 0.2])
            confirmation = [
                (request["confirmation_round"], request["reg_c"])
                for request in state["requests"]
                if request.get("run_kind") == "confirmation"
            ]
            self.assertEqual(
                confirmation,
                [(1, 0.02), (1, 0.05), (1, 0.1)],
            )
            definitive = json.loads(
                (root / "study" / "definitive_inversion.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(definitive["reg_c"], 0.05)
            definitive_point = json.loads(
                (root / "study" / definitive["point_manifest_path"]).read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(definitive_point["run_kind"], "confirmation")
            self.assertEqual(definitive_point["confirmation_round"], 1)
            resumed = self.run_case(root, corner=0.05, resume=True)
            self.assertEqual(resumed["manifest_id"], manifest["manifest_id"])

    def test_boundary_adjacent_candidates_run_one_outward_extension(self):
        for corner, side, expected in (
            (0.02, "low", [0.005]),
            (0.1, "high", [0.5]),
        ):
            with self.subTest(corner=corner), tempfile.TemporaryDirectory() as folder:
                manifest = self.run_case(Path(folder), corner=corner)
                self.assertEqual(manifest["extension_side"], side)
                self.assertEqual(manifest["extension_reg_c"], expected)
                self.assertEqual(manifest["formal_point_count"], 6)
                self.assertEqual(manifest["point_count"], 6)
                self.assertEqual(manifest["confirmation_manifest_count"], 3)

    def test_ambiguous_curvature_runs_one_two_sided_midpoint_round(self):
        base_logs = {
            0.01: (1.0, 0.0),
            0.02: (0.85, 0.4),
            0.05: (0.65, 0.65),
            0.1: (0.35, 0.85),
            0.2: (0.0, 1.0),
        }

        def profile(reg_c):
            for value, coordinates in base_logs.items():
                if math.isclose(reg_c, value, rel_tol=0.0, abs_tol=1e-14):
                    return 10.0 ** coordinates[0], 10.0 ** coordinates[1]
            coordinate = math.log10(reg_c / 0.05)
            return (
                10.0 ** max(coordinate, 0.0),
                10.0 ** max(-coordinate, 0.0),
            )

        with tempfile.TemporaryDirectory() as folder:
            manifest = self.run_case(
                Path(folder), corner=0.05, metric_profile=profile
            )
            expected = [math.sqrt(0.02 * 0.05), math.sqrt(0.05 * 0.1)]
            self.assertEqual(len(manifest["refinement_reg_c"]), 2)
            for observed, target in zip(manifest["refinement_reg_c"], expected):
                self.assertAlmostEqual(observed, target)
            self.assertEqual(manifest["formal_point_count"], 7)
            self.assertEqual(manifest["point_count"], 7)
            self.assertLessEqual(manifest["formal_point_count"], 8)

    def test_failed_first_confirmation_round_retries_same_lineage_once(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config, config_path = self.setup_case(root)

            def launcher(**kwargs):
                return write_valid_point(
                    kwargs,
                    config=config,
                    config_path=config_path,
                    root=root,
                    corner=0.05,
                )

            calls = []

            def confirmation_launcher(**kwargs):
                calls.append((kwargs["confirmation_round"], kwargs["reg_c"]))
                return write_confirmation(
                    kwargs,
                    config=config,
                    config_path=config_path,
                    root=root,
                    force_pass=kwargs["confirmation_round"] == 2,
                )

            manifest = run_lcurve_study(
                config=config,
                config_path=config_path,
                repo_root=root,
                output_dir=root / "study",
                preflight_class=FakePreflight,
                point_launcher=launcher,
                confirmation_launcher=confirmation_launcher,
                create_plot=False,
                require_forward_smoke=False,
            )
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(
                calls,
                [
                    (1, 0.02),
                    (1, 0.05),
                    (1, 0.1),
                    (2, 0.02),
                    (2, 0.05),
                    (2, 0.1),
                ],
            )
            self.assertEqual(manifest["confirmation_manifest_count"], 6)
            self.assertEqual(
                [row["round"] for row in manifest["confirmation_rounds"]],
                [1, 2],
            )
            self.assertEqual(
                [row["passed"] for row in manifest["confirmation_rounds"]],
                [False, True],
            )

    def test_second_failed_confirmation_round_fails_stability_gate(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config, config_path = self.setup_case(root)

            def launcher(**kwargs):
                return write_valid_point(
                    kwargs,
                    config=config,
                    config_path=config_path,
                    root=root,
                    corner=0.05,
                )

            calls = []

            def confirmation_launcher(**kwargs):
                calls.append((kwargs["confirmation_round"], kwargs["reg_c"]))
                return write_confirmation(
                    kwargs,
                    config=config,
                    config_path=config_path,
                    root=root,
                    force_pass=False,
                )

            with self.assertRaisesRegex(RuntimeError, "confirmation.*stability"):
                run_lcurve_study(
                    config=config,
                    config_path=config_path,
                    repo_root=root,
                    output_dir=root / "study",
                    preflight_class=FakePreflight,
                    point_launcher=launcher,
                    confirmation_launcher=confirmation_launcher,
                    create_plot=False,
                    require_forward_smoke=False,
                )
            self.assertEqual(len(calls), 6)
            self.assertFalse((root / "study" / "study_manifest.json").exists())

    def test_interrupted_point_is_preserved_and_resume_uses_attempt_two(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config, config_path = self.setup_case(root)
            failed = {"done": False}

            def launcher(**kwargs):
                if not failed["done"]:
                    failed["done"] = True
                    (kwargs["point_root"] / kwargs["run_id"]).mkdir()
                    return 75
                return write_valid_point(
                    kwargs,
                    config=config,
                    config_path=config_path,
                    root=root,
                    corner=0.1,
                )

            def confirmation_launcher(**kwargs):
                return write_confirmation(
                    kwargs,
                    config=config,
                    config_path=config_path,
                    root=root,
                )

            with self.assertRaisesRegex(RuntimeError, "without a point manifest"):
                run_lcurve_study(
                    config=config,
                    config_path=config_path,
                    repo_root=root,
                    output_dir=root / "study",
                    preflight_class=FakePreflight,
                    point_launcher=launcher,
                    confirmation_launcher=confirmation_launcher,
                    create_plot=False,
                    require_forward_smoke=False,
                )
            manifest = run_lcurve_study(
                config=config,
                config_path=config_path,
                repo_root=root,
                output_dir=root / "study",
                preflight_class=FakePreflight,
                resume=True,
                point_launcher=launcher,
                confirmation_launcher=confirmation_launcher,
                create_plot=False,
                require_forward_smoke=False,
            )
            self.assertEqual(manifest["status"], "complete")
            state = json.loads(
                (root / "study" / "run_state.json").read_text(encoding="utf-8")
            )
            same_value = [
                request
                for request in state["requests"]
                if request["reg_c"] == configuration()["inversion"][
                    "lcurve_base_reg_c"
                ][0]
                and request.get("run_kind", "independent") == "independent"
            ]
            self.assertEqual([request["attempt"] for request in same_value], [1, 2])
            self.assertEqual(
                same_value[0]["status"], "interrupted_without_manifest"
            )
            self.assertEqual(same_value[1]["status"], "complete")
            self.assertTrue(
                (root / "study" / "points" / same_value[0]["run_id"]).is_dir()
            )

    def test_changed_contract_and_tampered_point_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.run_case(root, corner=0.1)
            artifact = next((root / "study" / "points").glob("*/artifact.txt"))
            artifact.write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                self.run_case(root, corner=0.1, resume=True)


if __name__ == "__main__":
    unittest.main()
