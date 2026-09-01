import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import rasterio
from rasterio.transform import from_origin


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "data_preprocessing.py"


def load_module():
    spec = importlib.util.spec_from_file_location("data_preprocessing", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ExplicitAssumedCRSTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module()

    def write_tiff(self, path, crs=None):
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=2,
            height=2,
            count=1,
            dtype="float32",
            transform=from_origin(-1000.0, 1000.0, 500.0, 500.0),
            crs=crs,
        ) as dataset:
            dataset.write(np.arange(4, dtype="float32").reshape(2, 2), 1)

    def test_missing_crs_is_rejected_without_explicit_assumption(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "missing_crs.tif"
            self.write_tiff(path)
            with self.assertRaisesRegex(ValueError, "no declared CRS"):
                self.module.read_raster_file(
                    {"path": str(path), "coordinate_mode": "cell_center"}
                )

    def test_missing_crs_can_be_explicitly_assumed(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "missing_crs.tif"
            self.write_tiff(path)
            raster = self.module.read_raster_file(
                {
                    "path": str(path),
                    "coordinate_mode": "cell_center",
                    "expected_crs": 3031,
                    "assumed_crs": 3031,
                }
            )
            self.assertTrue(raster.attrs["source_crs_missing"])
            self.assertEqual(raster.attrs["validated_crs"], "EPSG:3031")
            self.assertEqual(raster.x.values.tolist(), [-750.0, -250.0])
            self.assertEqual(raster.y.values.tolist(), [750.0, 250.0])

    def test_assumption_is_rejected_when_source_crs_exists(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "declared_crs.tif"
            self.write_tiff(path, crs="EPSG:3031")
            with self.assertRaisesRegex(ValueError, "only when"):
                self.module.read_raster_file(
                    {
                        "path": str(path),
                        "coordinate_mode": "cell_center",
                        "expected_crs": 3031,
                        "assumed_crs": 3031,
                    }
                )


if __name__ == "__main__":
    unittest.main()
