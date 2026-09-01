import unittest

import numpy as np
from affine import Affine

from src.data_preprocessing import (
    nearest_valid_velocity_samples,
    raster_window_dataarray,
)


class VelocityCoordinateTests(unittest.TestCase):
    def test_window_values_are_labeled_at_true_centers(self):
        values = np.arange(6, dtype="float64").reshape(2, 3)
        transform = Affine(450.0, 0.0, -2800225.0, 0.0, -450.0, 2800225.0)
        data = raster_window_dataarray(values, transform, name="VX")
        np.testing.assert_array_equal(data.to_numpy(), values)
        np.testing.assert_allclose(data.x.to_numpy(), [-2800000, -2799550, -2799100])
        np.testing.assert_allclose(data.y.to_numpy(), [2800000, 2799550])
        self.assertEqual(data.attrs["coordinate_mode"], "cell_center")

    def test_rotated_window_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Rotated/sheared"):
            raster_window_dataarray(
                np.zeros((2, 2)), Affine(450.0, 1.0, 0.0, 0.0, -450.0, 0.0)
            )

    def test_solver_fill_uses_nearest_valid_centered_pixel(self):
        transform = Affine(450.0, 0.0, 0.0, 0.0, -450.0, 900.0)
        vx = np.array([[10.0, np.nan], [np.nan, 40.0]])
        vy = np.array([[1.0, np.nan], [np.nan, 4.0]])
        valid = np.array([[True, False], [False, True]])
        values, distances = nearest_valid_velocity_samples(
            np.array([[700.0, 200.0]]), vx, vy, valid, transform
        )
        np.testing.assert_allclose(values, [[40.0, 4.0]])
        np.testing.assert_allclose(distances, [np.hypot(25.0, 25.0)])


if __name__ == "__main__":
    unittest.main()
