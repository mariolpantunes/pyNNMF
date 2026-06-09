# coding: utf-8
"""
Integration tests for pyNNMF initialization and solver convergence.
"""

__author__ = "Mário Antunes"
__version__ = "0.2.0"
__email__ = "mario.antunes@ua.pt"
__status__ = "Development"

import unittest

import numpy as np

import pynnmf


class TestIntegration(unittest.TestCase):
    def test_init_convergence_comparison(self):
        """
        Verify that advanced init methods (nndsvd, svd_impute)
        provide a better starting point than random init.
        """
        # 1. Setup: 20x20 low-rank matrix
        np.random.seed(42)
        W_true = np.random.uniform(1, 10, (20, 2))
        H_true = np.random.uniform(1, 10, (2, 20))
        X_true = W_true @ H_true

        # Remove some entries to simulate missing values (about 20%)
        X = X_true.copy()
        mask = np.random.rand(20, 20) < 0.2
        X[mask] = np.nan

        k = 2
        n_iter = 10

        # 2. Collect costs for multiple random seeds
        seeds = range(50)
        random_costs = []
        for s in seeds:
            _, _, _, c = pynnmf.nmf_mu(X, k=k, n=n_iter, init="random", seed=s)
            random_costs.append(c)
        
        random_costs = np.array(random_costs)
        avg_random = np.mean(random_costs)

        # 3. Get costs for deterministic advanced methods
        _, _, _, cost_nndsvd = pynnmf.nmf_mu(X, k=k, n=n_iter, init="nndsvd", seed=None)
        _, _, _, cost_impute = pynnmf.nmf_mu(X, k=k, n=n_iter, init="svd_impute", seed=None)

        # 4. Statistical Validation
        # Advanced methods should be better than the average random start
        self.assertLess(cost_nndsvd, avg_random * 1.2, 
                        f"NNDSVD cost {cost_nndsvd:.4f} should not be significantly worse than avg random {avg_random:.4f}")
        self.assertLess(cost_impute, avg_random * 1.2, 
                        f"SVD-Impute cost {cost_impute:.4f} should not be significantly worse than avg random {avg_random:.4f}")

        # They should beat random starts in a majority of cases
        win_rate_nndsvd = np.sum(random_costs > cost_nndsvd) / len(random_costs)
        win_rate_impute = np.sum(random_costs > cost_impute) / len(random_costs)

        self.assertGreater(win_rate_nndsvd, 0.5, 
                           f"NNDSVD should beat random start more than 50% of the time (current win rate: {win_rate_nndsvd:.2%})")
        self.assertGreater(win_rate_impute, 0.5, 
                           f"SVD-Impute should beat random start more than 50% of the time (current win rate: {win_rate_impute:.2%})")

    def test_missing_value_recovery(self):
        """
        Test if the library can recover the structure of a matrix with missing values.
        """
        W_true = np.random.rand(10, 2)
        H_true = np.random.rand(2, 10)
        X_true = W_true @ H_true

        X = X_true.copy()
        mask = np.random.rand(10, 10) < 0.2
        X[mask] = np.nan

        # Solve with high iterations to ensure convergence
        rec, _, _, _ = pynnmf.nmf_mu(X, k=2, n=500, init="svd_impute", seed=42)

        # Check that the reconstruction is a decent approximation of the ground truth
        rmse = np.sqrt(np.mean((X_true - rec) ** 2))
        self.assertLess(rmse, 0.5, f"Recovery RMSE {rmse:.4f} too high")


if __name__ == "__main__":
    unittest.main()
