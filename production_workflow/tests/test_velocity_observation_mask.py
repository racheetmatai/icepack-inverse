"""Focused regressions for the production velocity-observation support."""

import unittest

import numpy as np
from rasterio.transform import Affine
from rasterio.windows import Window

from src import data_preprocessing as loader


class DummyMesh:
    """Return valid cell 0, then cell 4, then outside the mesh."""

    def locate_cell(self, point):
        return {0: 0, 1: 4}.get(int(point[0]))


class DummyRaster:
    def __init__(self, values):
        self.values = np.ma.asarray(values)
        self.width = self.values.shape[1]
        self.height = self.values.shape[0]
        self.transform = Affine(1, 0, 0, 0, -1, 1)
        self.crs = "EPSG:3031"
        self.closed = False

    def read(self, indexes=1, window=None, masked=False):
        self.last_masked = masked
        return self.values.copy()

    def window_transform(self, window):
        return self.transform

    def close(self):
        self.closed = True


class VelocityObservationMaskTests(unittest.TestCase):
    def test_loader_default_api_and_opt_in_diagnostics(self):
        rasters = {
            "VX": DummyRaster([[1.0, 2.0]]),
            "VY": DummyRaster([[3.0, 4.0]]),
            "ERRX": DummyRaster([[5.0, 6.0]]),
            "ERRY": DummyRaster([[7.0, 8.0]]),
            "SOURCE": DummyRaster([[1.0, 2.0]]),
        }
        original_fetch = loader.icepack.datasets.fetch_measures_antarctica
        original_open = loader.rasterio.open
        original_bounds = loader.get_min_max_coords
        try:
            loader.icepack.datasets.fetch_measures_antarctica = lambda: "velocity.nc"
            loader.rasterio.open = (
                lambda path, mode="r": rasters[path.rsplit(":", 1)[-1]]
            )
            loader.get_min_max_coords = lambda outline, delta: (0, 2, 0, 1)
            legacy = loader.get_windowed_velocity_file(None, object(), 0)
            extended = loader.get_windowed_velocity_file(
                None, object(), 0, return_validity=True
            )
        finally:
            loader.icepack.datasets.fetch_measures_antarctica = original_fetch
            loader.rasterio.open = original_open
            loader.get_min_max_coords = original_bounds
        self.assertEqual(len(legacy), 10)
        self.assertEqual(len(extended), 14)
        np.testing.assert_array_equal(extended[9], [[True, True]])
        self.assertEqual(extended[11]["valid_velocity_and_source"], 2)
        self.assertTrue(rasters["SOURCE"].closed)
        self.assertTrue(all(raster.last_masked for raster in rasters.values()))

    def test_velocity_and_source_not_error_define_population(self):
        vx = np.ma.array([[10.0, 20.0, 30.0, 40.0]], mask=[[0, 0, 1, 0]])
        vy = np.ma.array([[1.0, 2.0, 3.0, 4.0]])
        errx = np.ma.array([[0.0, 9999.0, 5.0, 6.0]], mask=[[1, 0, 0, 0]])
        erry = np.ma.array([[2.0, 3.0, 4.0, 5.0]])
        source = np.ma.array([[1.0, 2.0, 3.0, 0.0]], mask=[[0, 0, 0, 1]])
        valid, error_valid, summary = loader.build_velocity_observation_mask(
            vx, vy, errx, erry, source
        )
        np.testing.assert_array_equal(valid, [[True, True, False, False]])
        np.testing.assert_array_equal(error_valid, [[False, True, True, True]])
        self.assertEqual(summary["valid_velocity_and_source"], 2)
        self.assertEqual(summary["valid_observation_with_error_pair"], 1)

    def test_custom_velocity_without_source_uses_both_components(self):
        vx = np.ma.array([[0.0, 1.0, np.nan]])
        vy = np.ma.array([[0.0, 2.0, 3.0]], mask=[[0, 1, 0]])
        errors = np.ma.array([[np.nan, np.nan, np.nan]])
        valid, _, summary = loader.build_velocity_observation_mask(
            vx, vy, errors, errors, source=None
        )
        np.testing.assert_array_equal(valid, [[True, False, False]])
        self.assertFalse(summary["source_available"])

    def test_misaligned_components_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "must align"):
            loader.build_velocity_observation_mask(
                np.ones((1, 2)), np.ones((1, 3)),
                np.ones((1, 2)), np.ones((1, 2)),
            )

    def test_valid_cell_zero_is_retained(self):
        indices = loader.select_velocity_observation_indices(
            DummyMesh(), Window(0, 0, 3, 1), Affine.identity(),
            np.array([[True, True, True]]),
        )
        np.testing.assert_array_equal(indices, [[0, 0], [1, 0]])

    def test_revised_vertex_mesh_uses_centers(self):
        captured = {}
        original_vertex_mesh = loader.firedrake.VertexOnlyMesh
        original_function_space = loader.firedrake.FunctionSpace
        try:
            def vertex_mesh(mesh, points, **kwargs):
                captured["points"] = points
                return points

            loader.firedrake.VertexOnlyMesh = vertex_mesh
            loader.firedrake.FunctionSpace = lambda points, *args: points
            _, indices = loader.create_vertex_only_mesh_for_sparse_data(
                DummyMesh(), Window(0, 0, 2, 1), Affine.identity(),
                valid_velocity=np.array([[True, True]]),
                coordinate_mode="cell_center",
            )
        finally:
            loader.firedrake.VertexOnlyMesh = original_vertex_mesh
            loader.firedrake.FunctionSpace = original_function_space
        np.testing.assert_array_equal(indices, [[0, 0], [1, 0]])
        np.testing.assert_allclose(captured["points"], [[0.5, 0.5], [1.5, 0.5]])

    def test_empty_selection_fails_loudly(self):
        with self.assertRaisesRegex(ValueError, "No valid velocity observations"):
            loader.select_velocity_observation_indices(
                DummyMesh(), Window(0, 0, 3, 1), Affine.identity(),
                np.array([[False, False, False]]),
            )


if __name__ == "__main__":
    unittest.main()
