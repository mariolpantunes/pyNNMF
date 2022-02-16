import unittest
import numpy as np
import nmf.nmf as nmf


class TestSum(unittest.TestCase):
    def test_nmf_mu_00(self):
        X = np.random.rand(5,5)
        Xr, W, H, cost = nmf.nmf_mu(X, k=2)
        np.testing.assert_almost_equal(X, Xr, decimal=0)
    
    def test_nmf_mu_kl_00(self):
        X = np.random.rand(5,5)
        Xr, W, H = nmf.nmf_mu_kl(X, k=4)
        np.testing.assert_almost_equal(X, Xr, decimal=1)
    
    def test_nmf_mu_kl_01(self):
        X = np.array([[1,0,3], [0,2,0], [4,5,6]])
        
        Xr, W, H = nmf.nmf_mu_kl(X, k=3)
        print(X)
        print(Xr)

        np.testing.assert_almost_equal(X, Xr, decimal=1)


if __name__ == '__main__':
    unittest.main()
