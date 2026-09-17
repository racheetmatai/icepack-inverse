from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "production_workflow/controlled_replacement/checks.py"
SPEC = importlib.util.spec_from_file_location("controlled_checks", MODULE_PATH)
checks = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(checks)


class ControlledReplacementChecks(unittest.TestCase):
    def test_stable_row_alignment_uses_original_components(self) -> None:
        frame = pd.DataFrame({"row_id": ["b", "a"]})
        observations = pd.DataFrame({
            "row_id": ["a", "b"],
            "observed_vx_raw": [1.0, 2.0],
            "observed_vy_raw": [3.0, 4.0],
        })
        np.testing.assert_array_equal(
            checks.align_original_observations(frame, observations),
            np.array([[2.0, 4.0], [1.0, 3.0]]),
        )

    def test_control_pair_requires_same_mask_and_reference_outside(self) -> None:
        reference = np.array([1.0, 2.0, 3.0])
        mask = np.array([False, True, False])
        checks.verify_control_pair(reference, np.array([1.0, 8.0, 3.0]),
                                   np.array([1.0, 5.0, 3.0]), mask, mask.copy())
        with self.assertRaises(ValueError):
            checks.verify_control_pair(reference, np.array([1.0, 8.0, 3.0]),
                                       np.array([1.0, 5.0, 3.0]), mask, ~mask)

    def test_near_zero_denominator_is_undefined(self) -> None:
        self.assertTrue(np.isnan(checks.relative_rmse(1.0, 1.0e-14)))
        self.assertEqual(checks.relative_rmse(2.0, 4.0), 0.5)

    def test_manuscript_scenario_selection(self) -> None:
        table = pd.DataFrame({"scenario": ["whole_sector_original_measures", "controlled_original_measures"],
                              "value": [1, 2]})
        selected = checks.manuscript_rows(table)
        self.assertEqual(selected["value"].tolist(), [2])


if __name__ == "__main__":
    unittest.main()
