#!/usr/bin/env python3
# coding: utf-8

__author__ = "Mário Antunes"
__version__ = "0.2.0"
__email__ = "mario.antunes@ua.pt"
__status__ = "Development"


import argparse
import logging

import numpy as np

import pynnmf

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main(args: argparse.Namespace) -> None:
    """Simple NMF imputation example.
    
    Generates a small synthetic matrix, introduces missing values (NaNs),
    and uses the HALS solver to reconstruct the matrix and impute the
    missing entries.
    """
    # 1. Create a dummy non-negative matrix with some structural latent properties
    logger.info("Generating a 6x6 test matrix...")
    np.random.seed(args.seed)

    W_true = np.random.uniform(0.1, 5.0, size=(6, 3))
    H_true = np.random.uniform(0.1, 5.0, size=(3, 6))
    X_clean = W_true @ H_true

    # 2. Inject some missing values (NaNs)
    X = X_clean.copy()
    X[0, 1] = np.nan
    X[2, 4] = np.nan
    X[5, 3] = np.nan

    logger.info("\nOriginal Matrix with NaNs (missing entries):")
    logger.info(X)

    # 3. Factorize using HALS (Frobenius Norm, regularized, and missing value aware)
    logger.info(f"\nRunning NMF HALS solver with k={args.k} (iterations={args.n})...")
    rec, W, H, cost = pynnmf.nmf_hals(X, k=args.k, n=args.n, tol=1e-5, seed=args.seed)

    logger.info("\nFactor W:")
    logger.info(W)
    logger.info("\nFactor H:")
    logger.info(H)
    logger.info("\nReconstructed Matrix (Imputed):")
    logger.info(rec)
    logger.info(f"\nFinal reconstruction cost (Frobenius): {cost:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple NMF imputation example")
    parser.add_argument("-k", type=int, default=3, help="Number of latent components")
    parser.add_argument("-n", type=int, default=200, help="Number of iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    main(args)
