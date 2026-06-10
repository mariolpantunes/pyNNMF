# coding: utf-8
"""
Convergence and Monotonicity tests for pyNNMF.
Ensures that the cost function is non-increasing across iterations.
"""

__author__ = "Mário Antunes"
__version__ = "0.2.0"
__email__ = "mario.antunes@ua.pt"
__status__ = "Development"

import unittest
import numpy as np
import numpy.typing as np_typing
from pynnmf import nmf_mu, nmf_mu_kl, nmf_mu_is, nmf_als, nmf_hals

class TestConvergence(unittest.TestCase):
    m: int = 0
    n: int = 0
    k: int = 0
    x: np_typing.NDArray[np.floating] = np.array([])
    x_missing: np_typing.NDArray[np.floating] = np.array([])

    def setUp(self):
        np.random.seed(42)
        self.m, self.n, self.k = 20, 20, 3
        # Generate a strictly positive low-rank matrix
        W_true = np.random.uniform(1, 10, (self.m, self.k))
        H_true = np.random.uniform(1, 10, (self.k, self.n))
        self.x = W_true @ H_true
        
        # Inject some missing values
        self.x_missing = self.x.copy()
        mask = np.random.rand(self.m, self.n) < 0.2
        self.x_missing[mask] = np.nan

    def check_monotonicity(self, solver_func, cost_func, **kwargs):
        costs = []
        def callback(iteration, cost):
            costs.append(cost)

        # The solver must support the 'callback' argument
        solver_func(self.x_missing, k=self.k, n=20, callback=callback, **kwargs)
        
        self.assertGreater(len(costs), 1, "Solver did not produce costs via callback")
        
        for i in range(len(costs) - 1):
            # We allow a small epsilon for floating point precision
            self.assertLessEqual(
                costs[i+1], costs[i] + 1e-7, 
                f"Cost increased at iteration {i+1}: {costs[i]} -> {costs[i+1]}"
            )

    def test_mu_frobenius_monotonicity(self):
        from pynnmf.core import cost_fb
        self.check_monotonicity(nmf_mu, cost_fb, alpha=1e-3)

    def test_mu_kl_monotonicity(self):
        from pynnmf.core import cost_kl
        self.check_monotonicity(nmf_mu_kl, cost_kl)

    def test_mu_is_monotonicity(self):
        from pynnmf.core import cost_is
        self.check_monotonicity(nmf_mu_is, cost_is)

    def test_als_monotonicity(self):
        from pynnmf.core import cost_fb
        self.check_monotonicity(nmf_als, cost_fb)

    def test_hals_monotonicity(self):
        from pynnmf.core import cost_fb
        self.check_monotonicity(nmf_hals, cost_fb)

if __name__ == "__main__":
    unittest.main()
