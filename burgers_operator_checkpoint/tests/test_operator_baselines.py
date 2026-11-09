"""Small unit tests that do not touch the locked test dataset."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from operator_baselines import (  # noqa: E402
    make_initial_conditions,
    relative_l2_per_function,
    scan_models,
    solve_burgers,
)


class BurgersSolverTests(unittest.TestCase):
    def test_zero_stays_zero(self):
        zero = np.zeros((3, 64))
        result, steps = solve_burgers(zero, viscosity=0.01, final_time=0.1)
        self.assertGreater(steps, 0)
        np.testing.assert_allclose(result, zero, atol=1e-15)

    def test_mass_and_energy(self):
        a = np.array([[1.0, -0.5, 0.2], [0.3, 0.1, 0.4]])
        b = np.array([[0.2, 0.7, -0.2], [-0.1, 0.4, 0.6]])
        initial = make_initial_conditions(a, b, n_grid=128, amplitude=0.25)
        final, _ = solve_burgers(initial, viscosity=0.01, final_time=0.25)
        np.testing.assert_allclose(
            initial.mean(axis=1), final.mean(axis=1), atol=1e-12
        )
        self.assertTrue(np.all(np.mean(final ** 2, axis=1) <=
                               np.mean(initial ** 2, axis=1) + 1e-12))

    def test_t_zero_is_identity(self):
        initial = np.array([[0.0, 1.0, 2.0, 3.0]])
        result, steps = solve_burgers(initial, viscosity=0.01, final_time=0.0)
        self.assertEqual(steps, 0)
        np.testing.assert_array_equal(result, initial)


class RegressionTests(unittest.TestCase):
    def test_relative_error(self):
        truth = np.array([[3.0, 4.0], [1.0, 0.0]])
        np.testing.assert_allclose(relative_l2_per_function(truth, truth), 0)
        np.testing.assert_allclose(
            relative_l2_per_function(np.zeros_like(truth), truth), 1
        )

    def test_pcr_exact_on_low_rank_linear_map(self):
        rng = np.random.default_rng(93)
        basis = rng.normal(size=(6, 64))
        train = rng.normal(size=(60, 6)) @ basis
        val = rng.normal(size=(20, 6)) @ basis
        best, _, _ = scan_models(train, 2.0 * train, val, 2.0 * val)
        self.assertLess(best["PCR"]["validation_error"], 1e-10)
        self.assertLess(best["ridge-PCR"]["validation_error"], 1e-5)


if __name__ == "__main__":
    unittest.main()
