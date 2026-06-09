# coding: utf-8
"""
# pyNNMF

Non-Negative Matrix Factorization (NMF) with built-in support for missing value imputation.

$$X \\approx W \\cdot H$$

Unlike standard NMF libraries, `pyNNMF` is resilient to missing data (`NaN` values)
and handles them natively using optimized NumPy routines.

## Solvers

| Solver | Cost Function | Function |
|:---|:---|:---|
| Multiplicative Updates (MU) | Frobenius | `nmf_mu` |
| Multiplicative Updates (MU) | KL Divergence | `nmf_mu_kl` |
| Multiplicative Updates (MU) | Itakura-Saito | `nmf_mu_is` |
| Alternating Least Squares (ALS) | Frobenius | `nmf_als` |
| Alternating Least Squares (ALS) | KL Divergence | `nmf_als_kl` |
| Alternating Least Squares (ALS) | Itakura-Saito | `nmf_als_is` |
| Hierarchical ALS (HALS) | Frobenius | `nmf_hals` |
| Hierarchical ALS (HALS) | KL Divergence | `nmf_hals_kl` |
| Hierarchical ALS (HALS) | Itakura-Saito | `nmf_hals_is` |
| Robust Weighted NMF | Frobenius | `rwnmf` |

## Cost Functions

- `cost_fb`: Frobenius norm
- `cost_kl`: Kullback-Leibler divergence
- `cost_is`: Itakura-Saito divergence

## Quick Start

```python
import numpy as np
import pynnmf

X = np.array([[5, 3, 0, 1],
              [4, 0, 0, 1],
              [1, 1, 0, 5],
              [0, 1, 5, 4]])
X[X == 0] = np.nan  # mark zeros as missing

rec, W, H, cost = pynnmf.nmf_hals(X, k=2, seed=42)
```
"""

from .pynnmf import (
    cost_fb,
    cost_is,
    cost_kl,
    nmf_als,
    nmf_als_is,
    nmf_als_kl,
    nmf_hals,
    nmf_hals_is,
    nmf_hals_kl,
    nmf_mu,
    nmf_mu_is,
    nmf_mu_kl,
    rwnmf,
)

__all__ = [
    "rwnmf",
    "cost_fb",
    "nmf_mu",
    "cost_kl",
    "nmf_mu_kl",
    "cost_is",
    "nmf_mu_is",
    "nmf_als",
    "nmf_hals",
    "nmf_als_kl",
    "nmf_als_is",
    "nmf_hals_kl",
    "nmf_hals_is",
]
