import unittest
import numpy as np
import pynnmf.pynnmf as pynnmf

class TestSum(unittest.TestCase):
    def test_pynnmf_mu_00(self):
        X = np.random.rand(5,5)
        Xr, W, H, cost = pynnmf.nmf_mu(X, k=2)
        np.testing.assert_almost_equal(X, Xr, decimal=0)
    
    def test_pynnmf_mu_01(self):
        X = np.array([[1,np.nan,3], [np.nan,2,np.nan], [4,5,6]])
        Xr, W, H, cost = pynnmf.nmf_mu(X, k=3)
        self.assertAlmostEqual(0.0, pynnmf.cost_kl(X, Xr), delta=0.2)
    
    def test_pynnmf_mu_kl_00(self):
        X = np.random.rand(5,5)
        Xr, W, H, cost = pynnmf.nmf_mu_kl(X, k=5)
        self.assertAlmostEqual(0.0, pynnmf.cost_kl(X, Xr), delta=0.2)
    
    def test_pynnmf_mu_kl_01(self):
        X = np.array([[1,np.nan,3], [np.nan,2,np.nan], [4,5,6]])
        Xr, W, H, cost = pynnmf.nmf_mu_kl(X, k=3)
        self.assertAlmostEqual(0.0, pynnmf.cost_kl(X, Xr), delta=0.2)
    
    def test_pynnmf_mu_is_00(self):
        X = np.random.rand(5,5)
        Xr, W, H, cost = pynnmf.nmf_mu_is(X, k=5)
        self.assertAlmostEqual(0.0, pynnmf.cost_is(X, Xr), delta=0.2)
    
    def test_pynnmf_mu_is_01(self):
        X = np.array([[1,np.nan,3], [np.nan,2,np.nan], [4,5,6]])
        Xr, W, H, cost = pynnmf.nmf_mu_is(X, k=3)
        self.assertAlmostEqual(0.0, pynnmf.cost_is(X, Xr), delta=0.2)


if __name__ == '__main__':
    unittest.main()
