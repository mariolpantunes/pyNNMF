# coding: utf-8
"""
Tests for NMF solver methods.
"""

__author__ = "Mário Antunes"
__version__ = "0.2.0"
__email__ = "mario.antunes@ua.pt"
__status__ = "Development"

import unittest

import numpy as np

import pynnmf


class TestNMFMethods(unittest.TestCase):
    def test_pynnmf_rwnmf_00(self):
        x = np.random.rand(5, 5)
        xr, _, _, _ = pynnmf.rwnmf(x, k=2)
        np.testing.assert_almost_equal(x, xr, decimal=0)

    def test_pynnmf_rwnmf_01(self):
        x = np.array([[1, np.nan, 3], [np.nan, 2, np.nan], [4, 5, 6]])
        xr, _, _, cost = pynnmf.rwnmf(x, k=3)
        self.assertAlmostEqual(0.0, cost, delta=4)

    def test_pynnmf_mu_00(self):
        x = np.random.rand(5, 5)
        xr, _, _, _ = pynnmf.nmf_mu(x, k=2)
        np.testing.assert_almost_equal(x, xr, decimal=0)

    def test_pynnmf_mu_01(self):
        x = np.array([[1, np.nan, 3], [np.nan, 2, np.nan], [4, 5, 6]])
        xr, _, _, cost = pynnmf.nmf_mu(x, k=3)
        self.assertAlmostEqual(0.0, cost, delta=0.2)

    def test_pynnmf_mu_kl_00(self):
        x = np.random.rand(5, 5)
        xr, _, _, cost = pynnmf.nmf_mu_kl(x, k=5)
        self.assertAlmostEqual(0.0, cost, delta=0.2)

    def test_pynnmf_mu_kl_01(self):
        x = np.array([[1, np.nan, 3], [np.nan, 2, np.nan], [4, 5, 6]])
        xr, _, _, cost = pynnmf.nmf_mu_kl(x, k=3)
        self.assertAlmostEqual(0.0, cost, delta=0.2)

    def test_pynnmf_mu_is_00(self):
        x = np.random.rand(5, 5)
        xr, _, _, cost = pynnmf.nmf_mu_is(x, k=5)
        self.assertAlmostEqual(0.0, pynnmf.cost_is(x, xr), delta=0.2)

    def test_pynnmf_mu_is_01(self):
        x = np.array([[1, np.nan, 3], [np.nan, 2, np.nan], [4, 5, 6]])
        xr, _, _, cost = pynnmf.nmf_mu_is(x, k=3)
        self.assertAlmostEqual(0.0, cost, delta=0.2)





    def test_cost_fb_00(self):
        x = np.array([[5, 0, 4], [0, 5, 0], [3, 4, 3]])
        xr = np.array([[4, 2, 3], [3, 4.5, 3], [3, 4, 3]])
        self.assertAlmostEqual(4.9244289, pynnmf.cost_fb(x, xr), places=4)

    def test_cost_fb_01(self):
        x = np.array([[5, np.nan, 4], [np.nan, 5, np.nan], [3, 4, 3]])
        xr = np.array([[4, 2, 3], [3, 4.5, 3], [3, 4, 3]])
        self.assertAlmostEqual(1.5, pynnmf.cost_fb(x, xr), places=4)

    def test_cost_kl_00(self):
        x = np.array([[5, 0, 4], [0, 5, 0], [3, 4, 3]])
        xr = np.array([[4, 2, 3], [3, 4.5, 3], [3, 4, 3]])
        self.assertAlmostEqual(8.2932486, pynnmf.cost_kl(x, xr), places=4)

    def test_cost_kl_01(self):
        x = np.array([[5, np.nan, 4], [np.nan, 5, np.nan], [3, 4, 3]])
        xr = np.array([[4, 2, 3], [3, 4, 3], [3, 4, 3]])
        self.assertAlmostEqual(0.3821638, pynnmf.cost_kl(x, xr), places=4)

    def test_cost_is_00(self):
        x = np.array([[5, 0, 4], [0, 5, 0], [3, 4, 3]])
        xr = np.array([[4, 2, 3], [3, 4.5, 3], [3, 4, 3]])
        self.assertAlmostEqual(0.0782583, pynnmf.cost_is(x, xr), places=4)

    def test_cost_is_01(self):
        x = np.array([[5, np.nan, 4], [np.nan, 5, np.nan], [3, 4, 3]])
        xr = np.array([[4, 2, 3], [3, 4.5, 3], [3, 4, 3]])
        self.assertAlmostEqual(0.0782583, pynnmf.cost_is(x, xr), places=4)

    def test_small_matrix_1x1(self):
        x = np.array([[5.0]])
        xr, _, _, _ = pynnmf.nmf_mu(x, k=1, seed=42)
        np.testing.assert_almost_equal(x, xr, decimal=3)

    def test_all_nan_matrix(self):
        x = np.array([[np.nan, np.nan], [np.nan, np.nan]])
        xr, _, _, _ = pynnmf.nmf_mu(x, k=2, seed=42)
        self.assertEqual(xr.shape, (2, 2))
        self.assertFalse(np.any(np.isnan(xr)))

    def test_large_matrix(self):
        x = np.random.rand(150, 150)
        xr, _, _, _ = pynnmf.nmf_mu(x.copy(), k=5, n=10, seed=42)
        self.assertEqual(xr.shape, (150, 150))

        xr, _, _, _ = pynnmf.nmf_mu(x.copy(), k=5, n=10, seed=42)
        self.assertEqual(xr.shape, (150, 150))

        xr, _, _, _ = pynnmf.nmf_mu(x.copy(), k=5, n=10, seed=42)
        self.assertEqual(xr.shape, (150, 150))

    def test_vector_matrix_1xn(self):
        x = np.random.rand(1, 10)
        xr, _, _, _ = pynnmf.nmf_mu(x, k=1, n=20, seed=42)
        self.assertEqual(xr.shape, (1, 10))

    def test_vector_matrix_nx1(self):
        x = np.random.rand(10, 1)
        xr, _, _, _ = pynnmf.nmf_mu(x, k=1, n=20, seed=42)
        self.assertEqual(xr.shape, (10, 1))

    def test_empty_matrix(self):
        x = np.empty((0, 0))
        with self.assertRaises(ValueError):
            _ = pynnmf.nmf_mu(x, k=2)

    def test_pynnmf_als_00(self):
        x = np.random.rand(5, 5)
        xr, _, _, _ = pynnmf.nmf_als(x, k=2, seed=42)
        np.testing.assert_almost_equal(x, xr, decimal=0)

    def test_pynnmf_als_01(self):
        x = np.array([[1, np.nan, 3], [np.nan, 2, np.nan], [4, 5, 6]])
        xr, _, _, cost = pynnmf.nmf_als(x, k=3, seed=42)
        self.assertAlmostEqual(0.0, cost, delta=0.2)

    def test_pynnmf_hals_00(self):
        x = np.random.rand(5, 5)
        xr, _, _, _ = pynnmf.nmf_hals(x, k=2, seed=42)
        np.testing.assert_almost_equal(x, xr, decimal=0)

    def test_pynnmf_hals_01(self):
        x = np.array([[1, np.nan, 3], [np.nan, 2, np.nan], [4, 5, 6]])
        xr, _, _, cost = pynnmf.nmf_hals(x, k=3, seed=42)
        self.assertAlmostEqual(0.0, cost, delta=0.2)

    def test_observed_zeros_not_imputed(self):
        x_observed_zero = np.array([[2.0, 0.0], [1.0, 3.0]])
        x_imputed_zero = np.array([[2.0, np.nan], [1.0, 3.0]])

        for solver_name, solver_fn in [
            ("mu_fro", lambda a: pynnmf.nmf_mu(a, k=1, cost="frobenius", seed=42)),
            ("mu_kl", lambda a: pynnmf.nmf_mu(a, k=1, cost="kl", seed=42)),
            ("mu_is", lambda a: pynnmf.nmf_mu(a, k=1, cost="is", seed=42)),
            ("als", lambda a: pynnmf.nmf_als(a, k=1, seed=42)),
            ("hals", lambda a: pynnmf.nmf_hals(a, k=1, seed=42)),
        ]:
            rec_obs, _, _, _ = solver_fn(x_observed_zero.copy())
            rec_imp, _, _, _ = solver_fn(x_imputed_zero.copy())
            diff = np.abs(rec_obs - rec_imp).max()
            self.assertGreater(diff, 1.0, f"Solver {solver_name} did not treat observed zeros differently from NaNs")


if __name__ == "__main__":
    _ = unittest.main()
