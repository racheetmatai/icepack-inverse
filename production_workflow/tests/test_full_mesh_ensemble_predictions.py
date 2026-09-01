from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

WORKFLOW = Path(__file__).resolve().parents[1]
if str(WORKFLOW) not in sys.path:
    sys.path.insert(0, str(WORKFLOW))

from full_mesh_ensemble_predictions import construct_hybrid_controls


class HybridControlTests(unittest.TestCase):
    def test_replacement_and_median_are_exact(self):
        reference = np.asarray([10.0, 20.0, 30.0, 40.0])
        eligible = np.asarray([False, True, True, False])
        predictions = np.column_stack((np.arange(10.0), np.arange(10.0) + 100.0))
        members, median = construct_hybrid_controls(reference, eligible, predictions)
        self.assertEqual(members.shape, (10, 4))
        np.testing.assert_array_equal(members[:, 0], 10.0)
        np.testing.assert_array_equal(members[:, 3], 40.0)
        np.testing.assert_array_equal(median[[0, 3]], reference[[0, 3]])
        self.assertEqual(median[1], 4.5)
        self.assertEqual(median[2], 104.5)

    def test_requires_exactly_ten_members(self):
        with self.assertRaises(ValueError):
            construct_hybrid_controls(
                np.ones(3), np.asarray([True, False, True]), np.ones((9, 2))
            )


if __name__ == "__main__":
    unittest.main()
