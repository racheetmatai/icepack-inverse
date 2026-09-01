import unittest

import firedrake
import firedrake.adjoint
import numpy as np

from src.invert_c_theta import Invert


class InvertContinuationTests(unittest.TestCase):
    def test_nonzero_iteration_limited_control_can_continue_once(self):
        mesh = firedrake.UnitSquareMesh(2, 2)
        Q = firedrake.FunctionSpace(mesh, "CG", 1)
        object_ = Invert.__new__(Invert)
        object_.Q = Q
        object_.C = firedrake.Function(Q)
        object_.simulation_C = lambda control: control
        object_.loss_functional_nosigma = (
            lambda state: 0.5 * (state - 1.0) ** 2 * firedrake.dx(mesh)
        )
        object_.regularization_C_grad = (
            lambda control: 1.0e-12 * control**2 * firedrake.dx(mesh)
        )
        block_start = firedrake.Function(Q).assign(0.25)

        firedrake.adjoint.get_working_tape().clear_tape()
        first = object_.invert_C(
            gradient_tolerance=1e-10,
            step_tolerance=1e-10,
            max_iterations=0,
            loss_fcn_type="nosigma",
            regularization_grad_fcn=True,
            initial_control=block_start,
            return_estimator=True,
        )
        first_state = first._solver.getAlgorithmState()
        self.assertGreater(first_state.gnorm, 1e-10)
        np.testing.assert_allclose(object_.C.dat.data_ro, 0.25, atol=0.0)
        saved_control = object_.C.copy(deepcopy=True)
        np.testing.assert_allclose(
            saved_control.dat.data_ro, block_start.dat.data_ro, atol=0.0
        )

        firedrake.adjoint.get_working_tape().clear_tape()
        second = object_.invert_C(
            gradient_tolerance=1e-10,
            step_tolerance=1e-10,
            max_iterations=5,
            loss_fcn_type="nosigma",
            regularization_grad_fcn=True,
            initial_control=saved_control,
            return_estimator=True,
        )
        second_state = second._solver.getAlgorithmState()
        self.assertLessEqual(second_state.gnorm, 1e-10)
        np.testing.assert_allclose(object_.C.dat.data_ro, 1.0, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
