# <img src="assets/logo.svg" alt="logo" width="128" height="128" align="middle"> pyNNMF - Non-Negative Matrix Factorization (NMF)

![PyPI - Version](https://img.shields.io/pypi/v/pyNNMF)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pyNNMF)
![GitHub License](https://img.shields.io/github/license/mariolpantunes/pyNNMF)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/mariolpantunes/pyNNMF/main.yml)
![GitHub last commit](https://img.shields.io/github/last-commit/mariolpantunes/pyNNMF)

`pyNNMF` is a Python library for computing Non-Negative Matrix Factorization (NMF) with built-in support for missing value imputation. Unlike standard NMF libraries (e.g., scikit-learn), `pyNNMF` is resilient to missing data (`NaN` values and unobserved entries) and handles them natively using optimized NumPy routines.

---

## Features

- **Native Python & NumPy**: Highly optimized vectorized linear algebra operations without compiled or external C/C++ dependencies.
- **Missing Value Resiliency**: Handles missing values (`NaN`) and observed/unobserved zeroes without failing or distorting optimization gradients.
- **Multiple Solvers**: Supports Multiplicative Updates (MU), Alternating Least Squares (ALS), and Hierarchical Alternating Least Squares (HALS).
- **Multiple Cost Functions**: Minimizes Frobenius Norm (Euclidean), Kullback-Leibler (KL) Divergence, and Itakura-Saito (IS) Divergence.

---

## Installation

To install `pyNNMF` locally or prepare it for development:

```bash
git clone https://github.com/mariolpantunes/pyNNMF.git
cd pyNNMF
pip install -e .
```

---

## Solver & Cost Function Selection Guide

To get the best speed and accuracy (matrix completion / imputation) out of `pyNNMF`, select the solver and cost function pair according to the data's noise distribution and missingness ratio:

### 1. By Noise Distribution

| Noise Type | Recommended Cost Function | Recommended Solver | Rationale |
| :--- | :--- | :--- | :--- |
| **Additive / Gaussian** | Frobenius Norm (`cost_fb`) | `nmf_als` (or `nmf_mu`) | Frobenius norm represents the true negative log-likelihood for Gaussian noise. ALS converges fast. |
| **Count / Poisson / Sparse** | KL Divergence (`cost_kl`) | `nmf_mu_kl` (or `nmf_als_kl`) | KL divergence corresponds to Poisson likelihood and enforces sparsity naturally. |
| **Scale-Invariant / Audio** | IS Divergence (`cost_is`) | `nmf_mu_is` (or `nmf_als_is`) | IS divergence measures relative rather than absolute errors, protecting small-magnitude values. |

### 2. By Missingness Ratio
- **Low-to-Moderate Missingness (< 30%)**: **HALS** solvers (`nmf_hals`, `nmf_hals_kl`, `nmf_hals_is`) are recommended. They update variables coordinate-wise and converge to the lowest training objective minima very fast.
- **High Missingness (> 30%) / Highly Noisy**: **MU** and **ALS** solvers (`nmf_mu`, `nmf_mu_kl`, `nmf_mu_is`, `nmf_als`) are recommended. Their slower, diagonally-scaled update trajectories act as an **implicit regularizer**, preventing overfitting on the small number of observed entries.

---

## Usage Examples

Demonstration scripts are available in the [examples](examples/) directory:

### 1. Basic Imputation Example
Demonstrates how to initialize a low-rank matrix, mask entries as missing (`NaN`), and reconstruct/impute them:
```bash
PYTHONPATH=src python examples/imputation_example.py
```

### 2. Solver Validation CLI
Benchmark execution times and validate prediction accuracy (out-of-sample RMSE/MAE) across different noise distributions and missingness ratios using your `exectimeit` library:
```bash
PYTHONPATH=src python examples/validate_solvers.py --size 100 --noise gaussian --ratio 0.15
```

### 3. Initialization Comparison
Evaluate the impact of different initialization strategies (`random`, `nndsvd`, `svd_impute`) on the convergence speed and final reconstruction error:
```bash
PYTHONPATH=src python examples/init_comparison.py
```

---

## Running Tests & Checks

The test suite can be run using the standard Python `unittest` module:

```bash
PYTHONPATH=src python -m unittest discover -s test
```

To run formatting and static type checking:

```bash
ruff check src/ test/ examples/
npx pyright src/ test/ examples/
```

---

## Authors

* **Mário Antunes** - [mariolpantunes](https://github.com/mariolpantunes)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
