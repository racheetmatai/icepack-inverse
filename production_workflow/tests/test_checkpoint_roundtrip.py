import tempfile
from pathlib import Path
import json
import unittest

import firedrake
import numpy as np

from production_workflow.lcurve_runtime import _save_state


class Object:
    pass


class CheckpointRoundtripTests(unittest.TestCase):
    def test_lcurve_frozen_mesh_dof_state_roundtrips(self):
        mesh = firedrake.UnitSquareMesh(2, 2)
        Q = firedrake.FunctionSpace(mesh, "CG", 1)
        V = firedrake.VectorFunctionSpace(mesh, "CG", 1)
        x, y = firedrake.SpatialCoordinate(mesh)
        control = firedrake.interpolate(x + 2.0 * y, Q)
        control.rename("log_friction_C")
        theta = firedrake.Function(Q, name="log_fluidity_theta")
        velocity = firedrake.interpolate(firedrake.as_vector((x, y)), V)
        velocity.rename("velocity")
        object_ = Object()
        object_.mesh = mesh
        object_.Q = Q
        object_.V = V
        object_.degree = 1
        object_.C = control
        setattr(object_, "\u03b8", theta)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            result = _save_state(
                object_, velocity, root, mesh_sha256="synthetic-mesh-sha256"
            )
            schema = json.loads(
                (root / result["state_schema"]).read_text(encoding="utf-8")
            )
            self.assertEqual(schema["mesh_sha256"], "synthetic-mesh-sha256")
            restored_mesh = firedrake.UnitSquareMesh(2, 2)
            restored_Q = firedrake.FunctionSpace(restored_mesh, "CG", 1)
            restored_V = firedrake.VectorFunctionSpace(restored_mesh, "CG", 1)
            restored_coordinates = firedrake.interpolate(
                restored_mesh.coordinates, restored_V
            ).dat.data_ro[:, :2]
            np.testing.assert_array_equal(
                np.load(
                    root / result["array_files"]["coordinates"],
                    allow_pickle=False,
                ),
                restored_coordinates,
            )
            restored_control = firedrake.Function(restored_Q)
            restored_velocity = firedrake.Function(restored_V)
            restored_control.dat.data[:] = np.load(
                root / result["array_files"]["C"], allow_pickle=False
            )
            restored_velocity.dat.data[:] = np.load(
                root / result["array_files"]["velocity"], allow_pickle=False
            )
        np.testing.assert_allclose(
            restored_control.dat.data_ro, control.dat.data_ro
        )
        np.testing.assert_allclose(
            restored_velocity.dat.data_ro, velocity.dat.data_ro
        )


if __name__ == "__main__":
    unittest.main()
