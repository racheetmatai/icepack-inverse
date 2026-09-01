import importlib.util
from pathlib import Path
import unittest

import numpy as np


WORKFLOW = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FeaturePolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.units = load_module(
            "feature_units", WORKFLOW / "src" / "feature_units.py"
        )
        cls.rasters = load_module(
            "revised_raster_inputs",
            WORKFLOW / "src" / "revised_raster_inputs.py",
        )

    def test_alignment_matches_frozen_formula(self):
        bed_x = np.array([1.0, 0.0, 0.0, 1.0e100])
        bed_y = np.array([0.0, 0.0, 1.0, 1.0e100])
        surface_x = np.array([1.0, 1.0, 0.0, 1.0e100])
        surface_y = np.array([0.0, 0.0, -1.0, 1.0e100])
        observed = self.units.stabilized_gradient_alignment(
            bed_x, bed_y, surface_x, surface_y
        )
        bed_norm = np.hypot(np.hypot(bed_x, bed_y), 5.4e-3)
        surface_norm = np.hypot(np.hypot(surface_x, surface_y), 1.5e-4)
        expected = (
            (bed_x / bed_norm) * (surface_x / surface_norm)
            + (bed_y / bed_norm) * (surface_y / surface_norm)
        )
        expected = np.clip(expected, -1.0, 1.0)
        np.testing.assert_allclose(observed, expected)
        self.assertTrue(np.isfinite(observed).all())
        self.assertTrue(np.all((-1.0 <= observed) & (observed <= 1.0)))
        self.assertEqual(observed[1], 0.0)

    def test_revised_imports_omit_bouguer_and_snow(self):
        inputs = self.rasters.REVISED_GEOPHYSICS_INPUTS
        self.assertEqual(len(inputs), 8)
        self.assertIsNone(inputs[1])
        self.assertIsNone(inputs[5])

    def test_bed_class_exception_is_explicit_and_auxiliary(self):
        bed_class = self.rasters.REVISED_GEOPHYSICS_INPUTS[7]
        self.assertEqual(bed_class["method"], "nearest")
        self.assertEqual(bed_class["coordinate_mode"], "cell_center")
        self.assertEqual(bed_class["expected_crs"], 3031)
        self.assertEqual(bed_class["assumed_crs"], 3031)
        self.assertIn(
            "bedmachine_source",
            self.rasters.REVISED_CSV_EXPORT_POLICY["auxiliary"],
        )
        self.assertIn(
            "bedmachine_errbed",
            self.rasters.REVISED_CSV_EXPORT_POLICY["auxiliary"],
        )

    def test_direction_feature_is_allowed(self):
        self.assertIn(
            "cos_theta_bs", self.rasters.REVISED_ML_PREDICTOR_POLICY["allowed"]
        )


if __name__ == "__main__":
    unittest.main()
