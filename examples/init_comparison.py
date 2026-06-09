# coding: utf-8
"""
Example demonstrating the impact of different initialization methods on NMF convergence.
"""

__author__ = "Mário Antunes"
__version__ = "0.2.0"
__email__ = "mario.antunes@ua.pt"
__status__ = "Development"

import time
import numpy as np
from typing import Literal
from pynnmf import nmf_mu

def generate_low_rank_matrix(m: int = 100, n: int = 100, k: int = 5, noise: float = 0.1) -> np.ndarray:
    """Generate a synthetic low-rank matrix with Gaussian noise.

    Args:
        m (int): Number of rows. Defaults to 100.
        n (int): Number of columns. Defaults to 100.
        k (int): Rank of the underlying matrix. Defaults to 5.
        noise (float): Standard deviation of Gaussian noise. Defaults to 0.1.

    Returns:
        np.ndarray: A non-negative low-rank matrix of shape (m, n).
    """
    W_true = np.random.uniform(1, 10, (m, k))
    H_true = np.random.uniform(1, 10, (k, n))
    X = W_true @ H_true + np.random.normal(0, noise, (m, n))
    return np.maximum(X, 0.1)

def main() -> None:
    """Main execution function to compare NMF initialization methods.
    
    Generates a synthetic low-rank matrix with missing values and compares
    the convergence (cost) and accuracy (RMSE) of different initialization 
    strategies using the Multiplicative Update (MU) solver.
    """
    # Parameters
    m, n, k = 100, 100, 5
    missing_ratio = 0.3
    
    print(f"Generating {m}x{n} matrix with rank {k} and {missing_ratio*100}% missing values...")
    X_true = generate_low_rank_matrix(m, n, k)
    
    # Create missing values
    mask = np.random.rand(m, n) < missing_ratio
    X_missing = X_true.copy()
    X_missing[mask] = np.nan
    
    methods: list[Literal['random', 'nndsvd', 'svd_impute']] = ['random', 'nndsvd', 'svd_impute']
    results = {}

    for method in methods:
        print(f"Running NMF with init='{method}'...", end=" ", flush=True)
        start = time.time()
        
        # We use a small number of iterations to see the starting point advantage
        rec, W, H, cost = nmf_mu(X_missing, k=k, n=100, init=method, seed=42)
        
        elapsed = time.time() - start
        
        # Calculate actual RMSE against the ground truth X_true
        rmse = np.sqrt(np.mean((X_true - rec)**2))
        results[method] = (cost, rmse, elapsed)
        print(f"Done. (RMSE: {rmse:.4f}, Time: {elapsed:.2f}s)")

    print("\nSummary:")
    print(f"{'Method':<12} | {'Final Cost':<12} | {'Ground Truth RMSE':<18} | {'Time':<8}")
    print("-" * 55)
    for method, (cost, rmse, elapsed) in results.items():
        print(f"{method:<12} | {cost:<12.4f} | {rmse:<18.4f} | {elapsed:.2f}s")

if __name__ == "__main__":
    main()
