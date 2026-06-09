import math
import unittest

import numpy as np

import pynnmf.pynnmf as pynnmf


class TestSum(unittest.TestCase):
    def test_pynnmf_rwnmf_00(self):
        X = np.random.rand(5, 5)
        Xr, W, H, cost = pynnmf.rwnmf(X, k=2)
        np.testing.assert_almost_equal(X, Xr, decimal=0)

    def test_pynnmf_rwnmf_01(self):
        X = np.array([[1, np.nan, 3], [np.nan, 2, np.nan], [4, 5, 6]])
        Xr, W, H, cost = pynnmf.rwnmf(X, k=3)
        self.assertAlmostEqual(0.0, cost, delta=4)

    def test_pynnmf_mu_00(self):
        X = np.random.rand(5, 5)
        Xr, W, H, cost = pynnmf.nmf_mu(X, k=2)
        np.testing.assert_almost_equal(X, Xr, decimal=0)

    def test_pynnmf_mu_01(self):
        X = np.array([[1, np.nan, 3], [np.nan, 2, np.nan], [4, 5, 6]])
        Xr, W, H, cost = pynnmf.nmf_mu(X, k=3)
        self.assertAlmostEqual(0.0, cost, delta=0.2)

    def test_pynnmf_mu_kl_00(self):
        X = np.random.rand(5, 5)
        Xr, W, H, cost = pynnmf.nmf_mu_kl(X, k=5)
        self.assertAlmostEqual(0.0, cost, delta=0.2)

    def test_pynnmf_mu_kl_01(self):
        X = np.array([[1, np.nan, 3], [np.nan, 2, np.nan], [4, 5, 6]])
        Xr, W, H, cost = pynnmf.nmf_mu_kl(X, k=3)
        self.assertAlmostEqual(0.0, cost, delta=0.2)

    def test_pynnmf_mu_is_00(self):
        X = np.random.rand(5, 5)
        Xr, W, H, cost = pynnmf.nmf_mu_is(X, k=5)
        self.assertAlmostEqual(0.0, pynnmf.cost_is(X, Xr), delta=0.2)

    def test_pynnmf_mu_is_01(self):
        X = np.array([[1, np.nan, 3], [np.nan, 2, np.nan], [4, 5, 6]])
        Xr, W, H, cost = pynnmf.nmf_mu_is(X, k=3)
        self.assertAlmostEqual(0.0, cost, delta=0.2)

    def test_pynnmf_als_00(self):
        X = np.random.rand(5, 5)
        Xr, W, H, cost = pynnmf.nmf_als(X, k=2)
        np.testing.assert_almost_equal(X, Xr, decimal=0)

    def test_pynnmf_als_01(self):
        X = np.array([[1, np.nan, 3], [np.nan, 2, np.nan], [4, 5, 6]])
        Xr, W, H, cost = pynnmf.nmf_als(X, k=3)
        self.assertAlmostEqual(0.0, cost, delta=0.2)

    def test_pynnmf_hals_00(self):
        X = np.random.rand(5, 5)
        Xr, W, H, cost = pynnmf.nmf_hals(X, k=2)
        np.testing.assert_almost_equal(X, Xr, decimal=0)

    def test_pynnmf_hals_01(self):
        X = np.array([[1, np.nan, 3], [np.nan, 2, np.nan], [4, 5, 6]])
        Xr, W, H, cost = pynnmf.nmf_hals(X, k=3)
        self.assertAlmostEqual(0.0, cost, delta=0.2)

    def test_pynnmf_als_kl_00(self):
        X = np.random.rand(5, 5)
        Xr, W, H, cost = pynnmf.nmf_als_kl(X, k=5)
        self.assertAlmostEqual(0.0, cost, delta=0.2)

    def test_pynnmf_als_kl_01(self):
        X = np.array([[1, np.nan, 3], [np.nan, 2, np.nan], [4, 5, 6]])
        Xr, W, H, cost = pynnmf.nmf_als_kl(X, k=3)
        self.assertAlmostEqual(0.0, cost, delta=0.2)

    def test_pynnmf_als_is_00(self):
        X = np.random.rand(5, 5)
        Xr, W, H, cost = pynnmf.nmf_als_is(X, k=5)
        self.assertAlmostEqual(0.0, pynnmf.cost_is(X, Xr), delta=0.2)

    def test_pynnmf_als_is_01(self):
        X = np.array([[1, np.nan, 3], [np.nan, 2, np.nan], [4, 5, 6]])
        Xr, W, H, cost = pynnmf.nmf_als_is(X, k=3)
        self.assertAlmostEqual(0.0, cost, delta=0.2)

    def test_pynnmf_hals_kl_00(self):
        X = np.random.rand(5, 5)
        Xr, W, H, cost = pynnmf.nmf_hals_kl(X, k=5)
        self.assertAlmostEqual(0.0, cost, delta=0.2)

    def test_pynnmf_hals_kl_01(self):
        X = np.array([[1, np.nan, 3], [np.nan, 2, np.nan], [4, 5, 6]])
        Xr, W, H, cost = pynnmf.nmf_hals_kl(X, k=3)
        self.assertAlmostEqual(0.0, cost, delta=0.2)

    def test_pynnmf_hals_is_00(self):
        X = np.random.rand(5, 5)
        Xr, W, H, cost = pynnmf.nmf_hals_is(X, k=5)
        self.assertAlmostEqual(0.0, pynnmf.cost_is(X, Xr), delta=0.2)

    def test_pynnmf_hals_is_01(self):
        X = np.array([[1, np.nan, 3], [np.nan, 2, np.nan], [4, 5, 6]])
        Xr, W, H, cost = pynnmf.nmf_hals_is(X, k=3)
        self.assertAlmostEqual(0.0, cost, delta=0.2)

    def test_cost_fb_00(self):
        X = np.array([[5, 0, 4], [0, 5, 0], [3, 4, 3]])
        Xr = np.array([[4, 2, 3], [3, 4.5, 3], [3, 4, 3]])
        self.assertAlmostEqual(0.0, pynnmf.cost_fb(X, Xr), delta=2)

    def test_cost_fb_01(self):
        X = np.array([[5, math.nan, 4], [math.nan, 5, math.nan], [3, 4, 3]])
        Xr = np.array([[4, 2, 3], [3, 4.5, 3], [3, 4, 3]])
        self.assertAlmostEqual(0.0, pynnmf.cost_fb(X, Xr), delta=2)

    def test_cost_kl_00(self):
        X = np.array([[5, 0, 4], [0, 5, 0], [3, 4, 3]])
        Xr = np.array([[4, 2, 3], [3, 4.5, 3], [3, 4, 3]])
        self.assertAlmostEqual(0.0, pynnmf.cost_kl(X, Xr), delta=0.3)

    def test_cost_kl_01(self):
        X = np.array([[5, math.nan, 4], [math.nan, 5, math.nan], [3, 4, 3]])
        Xr = np.array([[4, 2, 3], [3, 4.5, 3], [3, 4, 3]])
        self.assertAlmostEqual(0.0, pynnmf.cost_kl(X, Xr), delta=0.3)

    def test_cost_is_00(self):
        X = np.array([[5, 0, 4], [0, 5, 0], [3, 4, 3]])
        Xr = np.array([[4, 2, 3], [3, 4.5, 3], [3, 4, 3]])
        self.assertAlmostEqual(0.0, pynnmf.cost_is(X, Xr), delta=0.3)

    def test_cost_is_01(self):
        X = np.array([[5, math.nan, 4], [math.nan, 5, math.nan], [3, 4, 3]])
        Xr = np.array([[4, 2, 3], [3, 4.5, 3], [3, 4, 3]])
        self.assertAlmostEqual(0.0, pynnmf.cost_is(X, Xr), delta=0.3)

    def test_cost_functions(self):
        import test.naive as naive

        # Test Case 1: Strictly positive matrix
        A1 = np.array([[1.0, 2.0], [3.0, 4.0]])
        B1 = np.array([[1.1, 1.9], [3.2, 3.8]])
        self.assertAlmostEqual(naive.cost_kl(A1, B1), pynnmf.cost_kl(A1, B1), places=7)
        self.assertAlmostEqual(naive.cost_is(A1, B1), pynnmf.cost_is(A1, B1), places=7)

        # Test Case 2: Matrix with NaNs but no zeros
        A2 = np.array([[1.0, np.nan], [3.0, 4.0]])
        B2 = np.array([[1.1, 1.9], [3.2, 3.8]])
        self.assertAlmostEqual(naive.cost_kl(A2, B2), pynnmf.cost_kl(A2, B2), places=7)
        self.assertAlmostEqual(naive.cost_is(A2, B2), pynnmf.cost_is(A2, B2), places=7)

    def test_reproducibility(self):
        X = np.random.rand(6, 6)
        Xr1, W1, H1, cost1 = pynnmf.nmf_mu_kl(X.copy(), k=3, seed=42)
        Xr2, W2, H2, cost2 = pynnmf.nmf_mu_kl(X.copy(), k=3, seed=42)
        np.testing.assert_array_almost_equal(W1, W2)
        np.testing.assert_array_almost_equal(H1, H2)
        self.assertAlmostEqual(cost1, cost2)

    def test_observed_zeros_kl_is(self):
        X = np.array([[1.0, 0.0, 3.0], [0.0, 2.0, 0.0], [4.0, 5.0, 6.0]])

        Xr, W, H, cost = pynnmf.nmf_mu_kl(X.copy(), k=2, seed=42)
        self.assertFalse(np.any(np.isnan(Xr)))
        self.assertFalse(np.any(np.isnan(W)))
        self.assertFalse(np.any(np.isnan(H)))

        Xr, W, H, cost = pynnmf.nmf_mu_is(X.copy(), k=2, seed=42)
        self.assertFalse(np.any(np.isnan(Xr)))

        Xr, W, H, cost = pynnmf.nmf_hals_kl(X.copy(), k=2, seed=42)
        self.assertFalse(np.any(np.isnan(Xr)))

        Xr, W, H, cost = pynnmf.nmf_hals_is(X.copy(), k=2, seed=42)
        self.assertFalse(np.any(np.isnan(Xr)))

    def test_small_matrix_1x1(self):
        X = np.array([[5.0]])
        Xr, W, H, cost = pynnmf.nmf_mu(X, k=1, seed=42)
        np.testing.assert_almost_equal(X, Xr, decimal=3)

    def test_all_nan_matrix(self):
        X = np.array([[np.nan, np.nan], [np.nan, np.nan]])
        Xr, W, H, cost = pynnmf.nmf_mu(X, k=2, seed=42)
        self.assertEqual(Xr.shape, (2, 2))
        self.assertFalse(np.any(np.isnan(Xr)))

    def test_large_matrix(self):
        X = np.random.rand(150, 150)
        Xr, W, H, cost = pynnmf.nmf_mu(X.copy(), k=5, n=10, seed=42)
        self.assertEqual(Xr.shape, (150, 150))

        Xr, W, H, cost = pynnmf.nmf_als(X.copy(), k=5, n=10, seed=42)
        self.assertEqual(Xr.shape, (150, 150))

        Xr, W, H, cost = pynnmf.nmf_hals(X.copy(), k=5, n=10, seed=42)
        self.assertEqual(Xr.shape, (150, 150))

    def test_vector_matrix_1xn(self):
        X = np.random.rand(1, 10)
        Xr, W, H, cost = pynnmf.nmf_mu(X, k=1, n=20, seed=42)
        self.assertEqual(Xr.shape, (1, 10))

    def test_vector_matrix_nx1(self):
        X = np.random.rand(10, 1)
        Xr, W, H, cost = pynnmf.nmf_mu(X, k=1, n=20, seed=42)
        self.assertEqual(Xr.shape, (10, 1))

    def test_empty_matrix(self):
        X = np.empty((0, 0))
        with self.assertRaises(ValueError):
            pynnmf.nmf_mu(X, k=2)


if __name__ == "__main__":
    unittest.main()
