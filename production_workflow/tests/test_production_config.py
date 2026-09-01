import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


WORKFLOW = Path(__file__).resolve().parents[1]
CONFIG_PATH = WORKFLOW / "amundsen_production_config.json"
ENTRY_PATH = WORKFLOW / "production_amundsen.py"


def load_entrypoint():
    spec = importlib.util.spec_from_file_location("production_amundsen", ENTRY_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ProductionConfigurationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_entrypoint()
        cls.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def test_frozen_config_validates(self):
        self.module.validate_frozen_config(copy.deepcopy(self.config))

    def test_unknown_top_level_key_is_rejected(self):
        config = copy.deepcopy(self.config)
        config["accidental"] = True
        with tempfile.TemporaryDirectory() as folder:
            temporary = Path(folder) / "bad_config.json"
            temporary.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unknown"):
                self.module.load_config(temporary)

    def test_mesh_regeneration_is_rejected(self):
        config = copy.deepcopy(self.config)
        config["domain"]["read_mesh"] = False
        with self.assertRaisesRegex(ValueError, "frozen mesh"):
            self.module.validate_frozen_config(config)

    def test_boundary_policy_is_exact(self):
        self.assertEqual(
            self.config["domain"]["dirichlet_ids"],
            [1, 3, 5, 6, 7, 8, 9, 10, 11],
        )
        self.assertEqual(self.config["domain"]["stress_ids"], [2, 4])
        config = copy.deepcopy(self.config)
        config["domain"]["dirichlet_ids"] = [1, 2, 4, 6, 7, 8, 9, 10, 11]
        with self.assertRaisesRegex(ValueError, "Dirichlet"):
            self.module.validate_frozen_config(config)

    def test_predictor_order_is_exact_and_bouguer_excluded(self):
        self.assertEqual(self.config["predictors"], self.module.FROZEN_PREDICTORS)
        self.assertNotIn("boug_anomaly", self.config["predictors"])

    def test_lcurve_protocol_is_exact(self):
        inversion = self.config["inversion"]
        self.assertEqual(self.config["schema"], "jog-amundsen-production-config-v2")
        self.assertEqual(
            inversion["lcurve_base_reg_c"],
            [0.01, 0.02, 0.05, 0.1, 0.2],
        )
        self.assertEqual(inversion["lcurve_block_iterations"], 50)
        self.assertEqual(inversion["lcurve_min_blocks"], 4)
        self.assertEqual(inversion["lcurve_max_blocks"], 6)
        self.assertEqual(inversion["lcurve_stable_transitions"], 2)
        self.assertEqual(inversion["lcurve_relative_misfit_tolerance"], 0.005)
        self.assertEqual(
            inversion["lcurve_relative_roughness_tolerance"], 0.005
        )
        self.assertEqual(inversion["lcurve_gradient_safety_tolerance"], 0.001)
        self.assertEqual(inversion["lcurve_low_extension_reg_c"], 0.005)
        self.assertEqual(inversion["lcurve_high_extension_reg_c"], 0.5)
        self.assertEqual(inversion["lcurve_curvature_ambiguity_ratio"], 1.25)
        self.assertEqual(inversion["lcurve_max_refinement_rounds"], 1)
        self.assertEqual(inversion["lcurve_max_confirmation_rounds"], 2)
        self.assertEqual(inversion["lcurve_max_formal_points"], 8)
        for legacy_key in (
            "max_iterations",
            "continuation_iterations",
            "lcurve_initial_reg_c",
            "lcurve_low_extension",
            "lcurve_high_extension",
        ):
            self.assertNotIn(legacy_key, inversion)

        config = copy.deepcopy(self.config)
        config["inversion"]["gradient_tolerance"] = 1e-100
        with self.assertRaisesRegex(ValueError, "solver controls"):
            self.module.validate_frozen_config(config)
        config = copy.deepcopy(self.config)
        config["inversion"]["lcurve_base_reg_c"] = [0.05]
        with self.assertRaisesRegex(ValueError, "formal L-curve base grid"):
            self.module.validate_frozen_config(config)
        config = copy.deepcopy(self.config)
        config["inversion"]["lcurve_block_iterations"] = 100
        with self.assertRaisesRegex(ValueError, "solver controls"):
            self.module.validate_frozen_config(config)
        config = copy.deepcopy(self.config)
        config["inversion"]["lcurve_initial_reg_c"] = [0.05]
        with self.assertRaisesRegex(ValueError, "legacy L-curve keys"):
            self.module.validate_frozen_config(config)

    def test_centered_velocity_regression_counts_are_frozen(self):
        expected = self.config["expected_counts"]
        self.assertEqual(expected["selected_observations"], 1622598)
        self.assertEqual(expected["selected_with_valid_error_pair"], 1530236)
        self.assertEqual(expected["full_mesh_raw_velocity_missing_dofs"], 65)
        self.assertEqual(expected["initial_velocity_nearest_filled_dofs"], 65)
        self.assertEqual(
            expected["dirichlet_velocity_nearest_filled_dofs"], 11
        )
        self.assertEqual(expected["central_square_observations"]["SQ03"], 12432)
        self.assertEqual(expected["central_square_observations"]["SQ10"], 12321)

    def test_manifest_identifier_is_order_independent_and_self_excluding(self):
        first = {"b": 2, "a": 1}
        second = {"a": 1, "b": 2, "manifest_id": "ignored"}
        self.assertEqual(
            self.module.manifest_identifier(first),
            self.module.manifest_identifier(second),
        )


if __name__ == "__main__":
    unittest.main()
