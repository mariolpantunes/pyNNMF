#!/usr/bin/env python3
# coding: utf-8

__author__ = "Mário Antunes"
__version__ = "0.2.0"
__email__ = "mario.antunes@ua.pt"
__status__ = "Development"


import argparse
import logging

import exectimeit as et
import numpy as np

import pynnmf

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main(args: argparse.Namespace) -> None:
    """Benchmark pyNNMF solvers and validate prediction accuracy.

    Generates synthetic matrices with different noise distributions (Gaussian, 
    Poisson, Multiplicative), introduces missing values, and evaluates the 
    reconstruction performance (RMSE, MAE) and execution time of various solvers.
    """
    logger.info(f"Generating synthetic {args.size}x{args.size} matrix...")
    np.random.seed(args.seed)

    # 1. Generate clean low-rank matrix
    k_true = 5
    W_true = np.random.uniform(0.1, 5.0, size=(args.size, k_true))
    H_true = np.random.uniform(0.1, 5.0, size=(k_true, args.size))
    X_clean = W_true @ H_true

    # 2. Add requested noise type
    logger.info(f"Injecting noise of type: {args.noise}...")
    if args.noise == "gaussian":
        noise = np.random.normal(0, 0.5, size=(args.size, args.size))
        x_noisy = np.maximum(0.01, X_clean + noise)
    elif args.noise == "poisson":
        x_noisy = np.random.poisson(X_clean)
        x_noisy = np.maximum(0.01, x_noisy.astype(float))
    elif args.noise == "multiplicative":
        noise = np.random.exponential(1.0, size=(args.size, args.size))
        x_noisy = X_clean * noise
        x_noisy = np.maximum(0.01, x_noisy)
    else:
        x_noisy = X_clean.copy()

    # 3. Partition matrix: hide validation entries
    logger.info(f"Hiding {args.ratio * 100:.1f}% of entries as validation set...")
    mask_val = np.random.rand(args.size, args.size) < args.ratio
    X_train = x_noisy.copy()
    X_train[mask_val] = np.nan
    val_true = x_noisy[mask_val]

    # Define solvers
    solvers = {
        "rwnmf (Fro)": lambda: pynnmf.rwnmf(X_train.copy(), k=args.k, num_iter=args.n, seed=args.seed)[0],
        "nmf_mu (Fro)": lambda: pynnmf.nmf_mu(X_train.copy(), k=args.k, n=args.n, seed=args.seed)[0],
        "nmf_als (Fro)": lambda: pynnmf.nmf_als(X_train.copy(), k=args.k, n=args.n, seed=args.seed)[0],
        "nmf_hals (Fro)": lambda: pynnmf.nmf_hals(X_train.copy(), k=args.k, n=args.n, seed=args.seed)[0],
        "nmf_mu_kl (KL)": lambda: pynnmf.nmf_mu_kl(X_train.copy(), k=args.k, n=args.n, seed=args.seed)[0],
        "nmf_mu_is (IS)": lambda: pynnmf.nmf_mu_is(X_train.copy(), k=args.k, n=args.n, seed=args.seed)[0],
    }

    logger.info("\nEvaluating solvers performance and prediction accuracy...")
    logger.info(f"{'Solver':<18} | {'Time (s)':<12} | {'Time Var (s)':<12} | {'Val RMSE':<10} | {'Val MAE':<10}")
    logger.info("-" * 75)

    for name, solver_fn in solvers.items():
        try:
            # Benchmark execution time using exectimeit
            mean_time, var_time, rec = et.timeit(args.reps, solver_fn)

            # Evaluate out-of-sample metrics
            val_pred = rec[mask_val]
            rmse = np.sqrt(np.mean((val_true - val_pred) ** 2))
            mae = np.mean(np.abs(val_true - val_pred))

            logger.info(f"{name:<18} | {mean_time:<12.5f} | {var_time:<12.5e} | {rmse:<10.4f} | {mae:<10.4f}")
        except Exception as e:
            logger.error(f"Solver {name} failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark pyNNMF solvers and validate prediction accuracy")
    parser.add_argument("--size", type=int, default=150, help="Size of the NxN test matrix")
    parser.add_argument("-k", type=int, default=5, help="NMF factorization rank k")
    parser.add_argument("-n", type=int, default=50, help="Maximum iterations per solver")
    parser.add_argument(
        "--noise",
        type=str,
        choices=["gaussian", "poisson", "multiplicative"],
        default="gaussian",
        help="Noise distribution model",
    )
    parser.add_argument("--ratio", type=float, default=0.15, help="Validation missingness ratio (0.0 to 1.0)")
    parser.add_argument("--reps", type=int, default=3, help="Repetitions for exectimeit timing benchmarks")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    main(args)
