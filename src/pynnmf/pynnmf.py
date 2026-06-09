# coding: utf-8
"""
pyNNMF: Non-Negative Matrix Factorization with missing values imputation.
Provides various solvers (MU, ALS, HALS) for different cost functions
(Frobenius norm, Kullback-Leibler, and Itakura-Saito).
"""

__author__ = "Mário Antunes"
__version__ = "0.2.0"
__email__ = "mariolpantunes@gmail.com"
__status__ = "Development"


import logging

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def rwnmf(
    X: NDArray[np.floating],
    k: int,
    alpha: float = 0.1,
    tol_fit_improvement: float = 1e-4,
    tol_fit_error: float = 1e-4,
    num_iter: int = 1000,
    seed: int | None = None,
    eval_every: int = 10,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating], float]:
    """
    Robust Weighted Non-negative Matrix Factorization (RWNMF) with L2 regularization.

    Optimizes the Frobenius norm objective function with missing values and L2 regularization:

    $$D_{\\text{Fro}}(X \\parallel U V^T) = \\frac{1}{2} \\| M \\odot (X - U V^T) \\|_F^2 + \\frac{\\alpha}{2} (\\|U\\|_F^2 + \\|V\\|_F^2)$$

    where $M$ is a binary mask representing observed values (1 if observed, 0 if NaN).

    Optimization method: Multiplicative Update (MU) rules.

    $$U \\leftarrow U \\odot \\frac{(M \\odot X) V}{(M \\odot (U V^T)) V + \\alpha U}$$

    $$V \\leftarrow V \\odot \\frac{(M \\odot X)^T U}{(M \\odot (U V^T))^T U + \\alpha V}$$

    Args:
        X (np.ndarray): Target matrix of shape (m, n) with potential NaN values.
        k (int): Number of latent components.
        alpha (float): L2 regularization parameter. Defaults to 0.1.
        tol_fit_improvement (float): Early stopping tolerance for reconstruction improvement. Defaults to 1e-4.
        tol_fit_error (float): Early stopping tolerance for reconstruction error. Defaults to 1e-4.
        num_iter (int): Maximum number of iterations. Defaults to 1000.
        seed (int, optional): Random seed for reproducibility. Defaults to None.
        eval_every (int): Frequency of checking early-stopping criteria. Defaults to 10.

    Returns:
        tuple: (Xr, U, V, error)
            - Xr (np.ndarray): Reconstructed matrix of shape (m, n).
            - U (np.ndarray): Left factor matrix of shape (m, k).
            - V (np.ndarray): Right factor matrix of shape (n, k).
            - error (float): Final Frobenius reconstruction error.
    """
    if isinstance(seed, int):
        np.random.seed(seed)

    eps = np.finfo(float).eps
    early_stop = False

    M = np.isnan(X)
    X = X.copy()
    X[M] = 0
    M = ~M

    U = np.random.uniform(size=(X.shape[0], k))
    U = np.maximum(U, eps)

    V = np.linalg.lstsq(U, X, rcond=None)[0].T
    V = np.maximum(V, eps)

    Xr = np.inf * np.ones(X.shape)

    M_float = M.astype(float)
    M_X = M_float * X
    M_X_T = M_X.T

    # Initial rec
    rec = U @ V.T

    for i in range(num_iter):
        # 1. Update U
        rec *= M_float
        U = U * ((M_X @ V) / (rec @ V + alpha * U))
        U = np.maximum(U, eps)

        # 2. Update V
        rec = U @ V.T
        rec *= M_float
        V = V * ((M_X_T @ U) / (rec.T @ U + alpha * V))
        V = np.maximum(V, eps)

        # residuals evaluation
        rec = U @ V.T
        if i % eval_every == 0:
            fit_error = np.linalg.norm(X - rec, "fro")
            fit_improvement = np.linalg.norm(rec - Xr, "fro")
            Xr = np.copy(rec)

            if fit_error < tol_fit_error or fit_improvement < tol_fit_improvement:
                early_stop = True
                break

    if early_stop:
        Xr = rec
    else:
        Xr = U @ V.T

    error = np.linalg.norm(X - Xr, ord="fro")
    return Xr, U, V, error


def cost_fb(A: NDArray[np.floating], B: NDArray[np.floating], M: NDArray[np.floating] | None = None) -> float:
    """
    Compute the Frobenius norm cost between two matrices, accounting for missing values.

    $$D_{\\text{Fro}}(A \\parallel B) = \\| M \\odot (A - B) \\|_F$$

    where $M$ is a mask indicating observed entries.

    Args:
        A (np.ndarray): Target matrix.
        B (np.ndarray): Reconstructed matrix.
        M (np.ndarray, optional): Binary mask of observed elements. Defaults to None.

    Returns:
        float: Frobenius cost.
    """
    if M is None:
        if np.any(np.isnan(A)):
            M = np.isnan(A)
            A_copy = A.copy()
            A_copy[M] = 0
            M_not = ~M
            cost = np.linalg.norm((M_not * A_copy) - (M_not * B), ord="fro")
        else:
            M_not = A > 0.0
            cost = np.linalg.norm((M_not * A) - (M_not * B), ord="fro")
    else:
        cost = np.linalg.norm((M * A) - (M * B), ord="fro")
    return float(cost)


def nmf_mu(
    X: NDArray[np.floating], k: int, n: int = 1000, tol: float = 1e-3, seed: int | None = None, eval_every: int = 10
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating], float]:
    """
    Non-Negative Matrix Factorization (NMF) via Multiplicative Updates (MU) using Frobenius norm.

    Optimizes the Frobenius norm objective function with L2 regularization:

    $$D_{\\text{Fro}}(X \\parallel W H) = \\frac{1}{2} \\| M \\odot (X - W H) \\|_F^2 + \\frac{\\text{tol}}{2} (\\|W\\|_F^2 + \\|H\\|_F^2)$$

    Multiplicative update steps:

    $$W \\leftarrow W \\odot \\frac{(M \\odot X) H^T - \\text{tol} \\|W\\|_F}{(M \\odot (W H)) H^T}$$

    $$H \\leftarrow H \\odot \\frac{W^T (M \\odot X) - \\text{tol} \\|H\\|_F}{W^T (M \\odot (W H))}$$

    Args:
        X (np.ndarray): Target matrix of shape (m, n).
        k (int): Number of latent components.
        n (int): Maximum number of iterations. Defaults to 1000.
        tol (float): L2 regularization parameter and convergence tolerance. Defaults to 1e-3.
        seed (int, optional): Random seed. Defaults to None.
        eval_every (int): Frequency of checking early-stopping criteria. Defaults to 10.

    Returns:
        tuple: (rec, W, H, cost)
            - rec (np.ndarray): Reconstructed matrix of shape (m, n).
            - W (np.ndarray): Left factor matrix of shape (m, k).
            - H (np.ndarray): Right factor matrix of shape (k, n).
            - cost (float): Final Frobenius cost.
    """

    if isinstance(seed, int):
        np.random.seed(seed)

    rows, columns = X.shape
    eps = np.finfo(float).eps

    W = np.abs(np.random.uniform(size=(rows, k)))
    W = np.maximum(W, eps)
    W = np.divide(W, k * W.max())

    H = np.abs(np.random.uniform(size=(k, columns)))
    H = np.maximum(H, eps)
    H = np.divide(H, k * H.max())

    M = np.isnan(X)
    X = X.copy()
    X[M] = 0
    M = ~M
    M_float = M.astype(float)

    # Precompute constant matrices
    M_X = M_float * X
    rec = W @ H
    cost = cost_fb(X, rec, M_float)

    for idx in range(n):
        # 1. Update W
        rec *= M_float
        W = W * ((M_X @ H.T - tol * np.linalg.norm(W, "fro")) / (rec @ H.T))
        W = np.maximum(W, eps)

        # 2. Update H
        rec = W @ H
        rec *= M_float
        H = H * ((W.T @ M_X - tol * np.linalg.norm(H, "fro")) / (W.T @ rec))
        H = np.maximum(H, eps)

        if idx % eval_every == 0:
            rec = W @ H
            cost = cost_fb(X, rec, M_float)
            if cost <= tol:
                break
    else:
        rec = W @ H
        cost = cost_fb(X, rec, M_float)

    X[~M] = np.nan
    return rec, W, H, cost


def cost_kl(A: NDArray[np.floating], B: NDArray[np.floating], M: NDArray[np.bool_] | None = None) -> float:
    """
    Compute the Kullback-Leibler (KL) divergence cost between two matrices.

    $$D_{\\text{KL}}(A \\parallel B) = \\sum_{i,j} M_{ij} \\left( A_{ij} \\log \\frac{A_{ij}}{B_{ij}} - A_{ij} + B_{ij} \\right)$$

    Args:
        A (np.ndarray): Target matrix.
        B (np.ndarray): Reconstructed matrix.
        M (np.ndarray, optional): Binary mask of observed elements. Defaults to None.

    Returns:
        float: KL divergence cost.
    """
    if M is None:
        if np.any(np.isnan(A)):
            M = np.isnan(A)
            M = ~M
        else:
            M = A > 0.0
    mask = M & (A > 0.0)
    a = A[mask]
    b = B[mask]
    return np.sum(a * np.log(a / b) - a + b)


def nmf_mu_kl(
    X: NDArray[np.floating], k: int, n: int = 100, tol: float = 1e-3, seed: int | None = None, r: int = 20, eval_every: int = 10
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating], float]:
    """
    NMF via Multiplicative Updates (MU) minimizing the Kullback-Leibler (KL) divergence.

    Optimizes the generalized KL divergence objective:

    $$D_{\\text{KL}}(X \\parallel W H) = \\sum_{i,j} M_{ij} \\left( X_{ij} \\log \\frac{X_{ij}}{(W H)_{ij}} - X_{ij} + (W H)_{ij} \\right)$$

    Multiplicative update steps:

    $$H \\leftarrow H \\odot \\frac{W^T \\left( \\frac{M \\odot X}{W H} \\right)}{W^T M}$$

    $$W \\leftarrow W \\odot \\frac{\\left( \\frac{M \\odot X}{W H} \\right) H^T}{M H^T}$$

    Args:
        X (np.ndarray): Target matrix of shape (m, n).
        k (int): Number of latent components.
        n (int): Maximum number of iterations. Defaults to 100.
        tol (float): Convergence tolerance. Defaults to 1e-3.
        seed (int, optional): Random seed. Defaults to None.
        r (int): Number of restarts for initial factor selection. Defaults to 20.
        eval_every (int): Frequency of checking early-stopping criteria. Defaults to 10.

    Returns:
        tuple: (rec, W, H, cost)
            - rec (np.ndarray): Reconstructed matrix of shape (m, n).
            - W (np.ndarray): Left factor matrix of shape (m, k).
            - H (np.ndarray): Right factor matrix of shape (k, n).
            - cost (float): Final KL cost.
    """

    if isinstance(seed, int):
        np.random.seed(seed)

    # Create a Mask
    M = np.isnan(X)
    X = X.copy()
    X[M] = 0
    M = ~M

    rows, columns = X.shape
    eps = np.finfo(float).eps

    # Create W and H
    W = np.abs(np.random.uniform(size=(rows, k)))
    W = np.maximum(W, eps)
    W = np.divide(W, k * W.max())

    H = np.abs(np.random.uniform(size=(k, columns)))
    H = np.maximum(H, eps)
    H = np.divide(H, k * H.max())

    cost_mask = M & (X > 0.0)

    if seed is None:
        rec = W @ H
        cost = cost_kl(X, rec, M)

        for _ in range(r):
            Wt = np.abs(np.random.uniform(size=(rows, k)))
            Wt = np.maximum(Wt, eps)
            Wt = np.divide(Wt, k * Wt.max())

            Ht = np.abs(np.random.uniform(size=(k, columns)))
            Ht = np.maximum(Ht, eps)
            Ht = np.divide(Ht, k * Ht.max())

            rec_temp = Wt @ Ht
            cost_temp = cost_kl(X, rec_temp, M)

            if cost_temp < cost:
                W = Wt
                H = Ht
                cost = cost_temp

    X_zero = X == 0
    rec = W @ H
    cost = cost_kl(X, rec, cost_mask)

    for idx in range(n):
        X_imp = np.where(X_zero, rec, X)
        ratio = X_imp / rec
        W_sum = W.sum(axis=0)[:, None]
        H = H * (W.T @ ratio / W_sum)
        H = np.maximum(H, eps)

        rec = W @ H
        X_imp = np.where(X_zero, rec, X)
        ratio = X_imp / rec
        H_sum = H.sum(axis=1)
        W = W * ((ratio @ H.T) / H_sum)
        W = np.maximum(W, eps)

        rec = W @ H
        if idx % eval_every == 0:
            cost = cost_kl(X, rec, cost_mask)
            if cost <= tol:
                break
    else:
        rec = W @ H
        cost = cost_kl(X, rec, cost_mask)

    X[~M] = np.nan
    return rec, W, H, cost


def cost_is(A: NDArray[np.floating], B: NDArray[np.floating], M: NDArray[np.bool_] | None = None) -> float:
    """
    Compute the Itakura-Saito (IS) divergence cost between two matrices.

    $$D_{\\text{IS}}(A \\parallel B) = \\sum_{i,j} M_{ij} \\left( \\frac{A_{ij}}{B_{ij}} - \\log \\frac{A_{ij}}{B_{ij}} - 1 \\right)$$

    Args:
        A (np.ndarray): Target matrix.
        B (np.ndarray): Reconstructed matrix.
        M (np.ndarray, optional): Binary mask of observed elements. Defaults to None.

    Returns:
        float: IS divergence cost.
    """
    if M is None:
        if np.any(np.isnan(A)):
            M = np.isnan(A)
            A_copy = A.copy()
            A_copy[M] = 0
            M = ~M
            mask = M & (A_copy > 0.0)
            a = A_copy[mask]
            b = B[mask]
            cost = np.sum((a / b) - np.log(a / b) - 1)
        else:
            M = A > 0.0
            mask = M & (A > 0.0)
            a = A[mask]
            b = B[mask]
            cost = np.sum((a / b) - np.log(a / b) - 1)
    else:
        mask = M & (A > 0.0)
        a = A[mask]
        b = B[mask]
        cost = np.sum((a / b) - np.log(a / b) - 1)
    return cost


def nmf_mu_is(
    X: NDArray[np.floating], k: int, n: int = 100, tol: float = 1e-3, seed: int | None = None, r: int = 20, eval_every: int = 10
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating], float]:
    """
    NMF via Multiplicative Updates (MU) minimizing the Itakura-Saito (IS) divergence.

    Optimizes the Itakura-Saito objective:

    $$D_{\\text{IS}}(X \\parallel W H) = \\sum_{i,j} M_{ij} \\left( \\frac{X_{ij}}{(W H)_{ij}} - \\log \\frac{X_{ij}}{(W H)_{ij}} - 1 \\right)$$

    Multiplicative update steps:

    $$H \\leftarrow H \\odot \\sqrt{ \\frac{W^T \\left( \\frac{M \\odot X}{(W H)^2} \\right)}{W^T \\left( \\frac{M}{W H} \\right)} }$$

    $$W \\leftarrow W \\odot \\sqrt{ \\frac{\\left( \\frac{M \\odot X}{(W H)^2} \\right) H^T}{\\left( \\frac{M}{W H} \\right) H^T} }$$

    Args:
        X (np.ndarray): Target matrix of shape (m, n).
        k (int): Number of latent components.
        n (int): Maximum number of iterations. Defaults to 100.
        tol (float): Convergence tolerance. Defaults to 1e-3.
        seed (int, optional): Random seed. Defaults to None.
        r (int): Number of restarts for initial factor selection. Defaults to 20.
        eval_every (int): Frequency of checking early-stopping criteria. Defaults to 10.

    Returns:
        tuple: (rec, W, H, cost)
            - rec (np.ndarray): Reconstructed matrix of shape (m, n).
            - W (np.ndarray): Left factor matrix of shape (m, k).
            - H (np.ndarray): Right factor matrix of shape (k, n).
            - cost (float): Final IS cost.
    """

    if isinstance(seed, int):
        np.random.seed(seed)

    # Create a Mask
    M = np.isnan(X)
    X = X.copy()
    X[M] = 0
    M = ~M

    rows, columns = X.shape
    eps = np.finfo(float).eps

    # Create W and H
    W = np.abs(np.random.uniform(size=(rows, k)))
    W = np.maximum(W, eps)
    W = np.divide(W, k * W.max())

    H = np.abs(np.random.uniform(size=(k, columns)))
    H = np.maximum(H, eps)
    H = np.divide(H, k * H.max())

    cost_mask = M & (X > 0.0)

    if seed is None:
        rec = W @ H
        cost = cost_is(X, rec, M)

        for _ in range(r):
            Wt = np.abs(np.random.uniform(size=(rows, k)))
            Wt = np.maximum(Wt, eps)
            Wt = np.divide(Wt, k * Wt.max())

            Ht = np.abs(np.random.uniform(size=(k, columns)))
            Ht = np.maximum(Ht, eps)
            Ht = np.divide(Ht, k * Ht.max())

            rec_temp = Wt @ Ht
            cost_temp = cost_is(X, rec_temp, M)

            if cost_temp < cost:
                W = Wt
                H = Ht
                cost = cost_temp

    X_zero = X == 0
    rec = W @ H
    cost = cost_is(X, rec, cost_mask)

    for idx in range(n):
        # 1. Update H
        X_imp = np.where(X_zero, rec, X)
        ratio = X_imp / rec
        rec_sum_rows = rec.sum(axis=1)[:, None]
        W_scaled = W / rec_sum_rows
        numerator = W_scaled.T @ ratio
        denominator = W_scaled.sum(axis=0)[:, None]
        H = H * np.sqrt(numerator / denominator)
        H = np.maximum(H, eps)

        # 2. Update W
        rec = W @ H
        X_imp = np.where(X_zero, rec, X)
        ratio = X_imp / rec
        rec_sum_cols = rec.sum(axis=0)
        H_scaled = H / rec_sum_cols
        numerator = ratio @ H_scaled.T
        denominator = H_scaled.sum(axis=1)
        W = W * np.sqrt(numerator / denominator)
        W = np.maximum(W, eps)

        # 3. Cost
        rec = W @ H
        if idx % eval_every == 0:
            cost = cost_is(X, rec, cost_mask)
            if cost <= tol:
                break
    else:
        rec = W @ H
        cost = cost_is(X, rec, cost_mask)

    X[~M] = np.nan
    return rec, W, H, cost


def nmf_als(
    X: NDArray[np.floating], k: int, n: int = 1000, tol: float = 1e-3, seed: int | None = None, eval_every: int = 10
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating], float]:
    """
    Alternating Least Squares (ALS) NMF with missing values imputation.

    Minimizes the Frobenius norm. During each iteration, missing entries (and observed zeros)
    are imputed dynamically from the current reconstruction:

    $$X_{\\text{imp}} = M \\odot X + (1 - M) \\odot (W H)$$

    Then, least squares solutions are computed alternately:

    $$W \\leftarrow \\max\\left(\\epsilon, X_{\\text{imp}} H^T (H H^T + 10^{-9} I_k)^{-1}\\right)$$

    $$H \\leftarrow \\max\\left(\\epsilon, (W^T W + 10^{-9} I_k)^{-1} W^T X_{\\text{imp}}\\right)$$

    Args:
        X (np.ndarray): Target matrix of shape (m, n).
        k (int): Number of latent components.
        n (int): Maximum number of iterations. Defaults to 1000.
        tol (float): Convergence tolerance. Defaults to 1e-3.
        seed (int, optional): Random seed. Defaults to None.
        eval_every (int): Frequency of checking early-stopping criteria. Defaults to 10.

    Returns:
        tuple: (rec, W, H, cost)
            - rec (np.ndarray): Reconstructed matrix.
            - W (np.ndarray): Left factor matrix.
            - H (np.ndarray): Right factor matrix.
            - cost (float): Final Frobenius cost.
    """

    if isinstance(seed, int):
        np.random.seed(seed)
    rows, columns = X.shape
    eps = np.finfo(float).eps
    W = np.abs(np.random.uniform(size=(rows, k)))
    W = np.maximum(W, eps)
    W = np.divide(W, k * W.max())
    H = np.abs(np.random.uniform(size=(k, columns)))
    H = np.maximum(H, eps)
    H = np.divide(H, k * H.max())

    M = np.isnan(X)
    X = X.copy()
    X[M] = 0
    M = ~M
    M_float = M.astype(float)
    X_zero = X == 0

    rec = W @ H
    cost = cost_fb(X, rec, M_float)

    for idx in range(n):
        # Imputation
        rec = W @ H
        X_imp = np.where(X_zero, rec, X)

        # Update W
        W = np.maximum(eps, np.linalg.solve(H @ H.T + 1e-9 * np.eye(k), H @ X_imp.T).T)

        # Imputation
        rec = W @ H
        X_imp = np.where(X_zero, rec, X)

        # Update H
        H = np.maximum(eps, np.linalg.solve(W.T @ W + 1e-9 * np.eye(k), W.T @ X_imp))

        if idx % eval_every == 0:
            rec = W @ H
            cost = cost_fb(X, rec, M_float)
            if cost <= tol:
                break
    else:
        rec = W @ H
        cost = cost_fb(X, rec, M_float)

    X[~M] = np.nan
    return rec, W, H, cost


def nmf_hals(
    X: NDArray[np.floating], k: int, n: int = 1000, tol: float = 1e-3, seed: int | None = None, eval_every: int = 10
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating], float]:
    """
    Hierarchical Alternating Least Squares (HALS) NMF with missing values imputation.

    Minimizes the Frobenius norm. Computes dynamic imputation as in ALS, then updates
    factors column-by-column (for W) and row-by-row (for H):

    $$W_{*, i} \\leftarrow \\max\\left(\\epsilon, W_{*, i} + \\frac{(X_{\\text{imp}} H^T)_{*, i} - W (H H^T)_{*, i}}{(H H^T)_{i,i}}\\right)$$

    $$H_{i, *} \\leftarrow \\max\\left(\\epsilon, H_{i, *} + \\frac{(W^T X_{\\text{imp}})_{i, *} - (W^T W)_{i, *} H}{(W^T W)_{i,i}}\\right)$$

    Args:
        X (np.ndarray): Target matrix of shape (m, n).
        k (int): Number of latent components.
        n (int): Maximum number of iterations. Defaults to 1000.
        tol (float): Convergence tolerance. Defaults to 1e-3.
        seed (int, optional): Random seed. Defaults to None.
        eval_every (int): Frequency of checking early-stopping criteria. Defaults to 10.

    Returns:
        tuple: (rec, W, H, cost)
            - rec (np.ndarray): Reconstructed matrix.
            - W (np.ndarray): Left factor matrix.
            - H (np.ndarray): Right factor matrix.
            - cost (float): Final Frobenius cost.
    """

    if isinstance(seed, int):
        np.random.seed(seed)
    rows, columns = X.shape
    eps = np.finfo(float).eps
    W = np.abs(np.random.uniform(size=(rows, k)))
    W = np.maximum(W, eps)
    W = np.divide(W, k * W.max())
    H = np.abs(np.random.uniform(size=(k, columns)))
    H = np.maximum(H, eps)
    H = np.divide(H, k * H.max())

    M = np.isnan(X)
    X = X.copy()
    X[M] = 0
    M = ~M
    M_float = M.astype(float)
    X_zero = X == 0

    rec = W @ H
    cost = cost_fb(X, rec, M_float)

    for idx in range(n):
        # Imputation
        rec = W @ H
        X_imp = np.where(X_zero, rec, X)

        # Update W column-wise
        A = H @ H.T
        B = X_imp @ H.T
        for i in range(k):
            denom = A[i, i] if A[i, i] > eps else eps
            W[:, i] = np.maximum(eps, W[:, i] + (B[:, i] - W @ A[:, i]) / denom)

        # Imputation
        rec = W @ H
        X_imp = np.where(X_zero, rec, X)

        # Update H row-wise
        C = W.T @ W
        D = W.T @ X_imp
        for i in range(k):
            denom = C[i, i] if C[i, i] > eps else eps
            H[i, :] = np.maximum(eps, H[i, :] + (D[i, :] - C[i, :] @ H) / denom)

        if idx % eval_every == 0:
            rec = W @ H
            cost = cost_fb(X, rec, M_float)
            if cost <= tol:
                break
    else:
        rec = W @ H
        cost = cost_fb(X, rec, M_float)

    X[~M] = np.nan
    return rec, W, H, cost


def nmf_als_kl(
    X: NDArray[np.floating], k: int, n: int = 100, tol: float = 1e-3, seed: int | None = None, r: int = 20, eval_every: int = 10
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating], float]:
    """
    Alternating Least Squares (ALS) NMF with Kullback-Leibler (KL) divergence and missing values.

    Delegates to the alternating multiplicative update implementation (`nmf_mu_kl`), which
    guarantees stability and convergence for the KL objective.

    Args:
        X (np.ndarray): Target matrix of shape (m, n).
        k (int): Number of latent components.
        n (int): Maximum number of iterations. Defaults to 100.
        tol (float): Convergence tolerance. Defaults to 1e-3.
        seed (int, optional): Random seed. Defaults to None.
        r (int): Restarts. Defaults to 20.
        eval_every (int): Frequency of checking early-stopping criteria. Defaults to 10.

    Returns:
        tuple: (rec, W, H, cost)
    """
    return nmf_mu_kl(X, k, n, tol, seed, r, eval_every)


def nmf_als_is(
    X: NDArray[np.floating], k: int, n: int = 100, tol: float = 1e-3, seed: int | None = None, r: int = 20, eval_every: int = 10
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating], float]:
    """
    Alternating Least Squares (ALS) NMF with Itakura-Saito (IS) divergence and missing values.

    Delegates to the alternating multiplicative update implementation (`nmf_mu_is`), which
    guarantees stability and convergence for the IS objective.

    Args:
        X (np.ndarray): Target matrix of shape (m, n).
        k (int): Number of latent components.
        n (int): Maximum number of iterations. Defaults to 100.
        tol (float): Convergence tolerance. Defaults to 1e-3.
        seed (int, optional): Random seed. Defaults to None.
        r (int): Restarts. Defaults to 20.
        eval_every (int): Frequency of checking early-stopping criteria. Defaults to 10.

    Returns:
        tuple: (rec, W, H, cost)
    """
    return nmf_mu_is(X, k, n, tol, seed, r, eval_every)


def nmf_hals_kl(
    X: NDArray[np.floating], k: int, n: int = 100, tol: float = 1e-3, seed: int | None = None, r: int = 20, eval_every: int = 10
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating], float]:
    """
    Hierarchical Alternating Least Squares (HALS) NMF minimizing the KL divergence.

    Uses coordinate descent combined with Majorization-Minimization update steps.
    For each component $i \\in \\{1, \\dots, k\\}$, updates factors sequentially:

    $$H_{i, *} \\leftarrow H_{i, *} \\odot \\frac{W_{*, i}^T \\left( \\frac{X_{\\text{imp}}}{W H} \\right)}{\\sum_{j} W_{j, i}}$$

    $$W_{*, i} \\leftarrow W_{*, i} \\odot \\frac{\\left( \\frac{X_{\\text{imp}}}{W H} \\right) H_{i, *}^T}{\\sum_{j} H_{i, j}}$$

    where $X_{\\text{imp}}$ is the dynamically imputed target matrix.

    Args:
        X (np.ndarray): Target matrix of shape (m, n).
        k (int): Number of latent components.
        n (int): Maximum number of iterations. Defaults to 100.
        tol (float): Convergence tolerance. Defaults to 1e-3.
        seed (int, optional): Random seed. Defaults to None.
        r (int): Restarts for initialization. Defaults to 20.
        eval_every (int): Frequency of checking early-stopping criteria. Defaults to 10.

    Returns:
        tuple: (rec, W, H, cost)
            - rec (np.ndarray): Reconstructed matrix.
            - W (np.ndarray): Left factor matrix.
            - H (np.ndarray): Right factor matrix.
            - cost (float): Final KL cost.
    """

    if isinstance(seed, int):
        np.random.seed(seed)

    M = np.isnan(X)
    X = X.copy()
    X[M] = 0
    M = ~M
    X_zero = X == 0

    rows, columns = X.shape
    eps = np.finfo(float).eps

    W = np.abs(np.random.uniform(size=(rows, k)))
    W = np.maximum(W, eps)
    W = np.divide(W, k * W.max())

    H = np.abs(np.random.uniform(size=(k, columns)))
    H = np.maximum(H, eps)
    H = np.divide(H, k * H.max())

    cost_mask = M & (X > 0.0)

    if seed is None:
        rec = W @ H
        cost = cost_kl(X, rec, M)
        for _ in range(r):
            Wt = np.abs(np.random.uniform(size=(rows, k)))
            Wt = np.maximum(Wt, eps)
            Wt = np.divide(Wt, k * Wt.max())
            Ht = np.abs(np.random.uniform(size=(k, columns)))
            Ht = np.maximum(Ht, eps)
            Ht = np.divide(Ht, k * Ht.max())
            rec_temp = Wt @ Ht
            cost_temp = cost_kl(X, rec_temp, M)
            if cost_temp < cost:
                W = Wt
                H = Ht
                cost = cost_temp

    rec = W @ H
    cost = cost_kl(X, rec, cost_mask)

    for idx in range(n):
        # Update H row-by-row
        for i in range(k):
            rec = W @ H
            X_imp = np.where(X_zero, rec, X)
            ratio = X_imp / rec
            H[i, :] = H[i, :] * (W[:, i] @ ratio / (W[:, i].sum() + eps))
            H[i, :] = np.maximum(H[i, :], eps)

        # Update W column-by-column
        for i in range(k):
            rec = W @ H
            X_imp = np.where(X_zero, rec, X)
            ratio = X_imp / rec
            W[:, i] = W[:, i] * (ratio @ H[i, :] / (H[i, :].sum() + eps))
            W[:, i] = np.maximum(W[:, i], eps)

        if idx % eval_every == 0:
            rec = W @ H
            cost = cost_kl(X, rec, cost_mask)
            if cost <= tol:
                break
    else:
        rec = W @ H
        cost = cost_kl(X, rec, cost_mask)

    X[~M] = np.nan
    return rec, W, H, cost


def nmf_hals_is(
    X: NDArray[np.floating], k: int, n: int = 100, tol: float = 1e-3, seed: int | None = None, r: int = 20, eval_every: int = 10
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating], float]:
    """
    Hierarchical Alternating Least Squares (HALS) NMF minimizing the Itakura-Saito (IS) divergence.

    Uses coordinate descent combined with Majorization-Minimization update steps.
    For each component $i \\in \\{1, \\dots, k\\}$, updates factors sequentially:

    $$H_{i, *} \\leftarrow H_{i, *} \\odot \\sqrt{\\frac{W_{*, i}^T \\left( \\frac{X_{\\text{imp}}}{(W H)^2} \\right)}{W_{*, i}^T \\left( \\frac{1}{W H} \\right)}}$$

    $$W_{*, i} \\leftarrow W_{*, i} \\odot \\sqrt{\\frac{\\left( \\frac{X_{\\text{imp}}}{(W H)^2} \\right) H_{i, *}^T}{\\left( \\frac{1}{W H} \\right) H_{i, *}^T}}$$

    where $X_{\\text{imp}}$ is the dynamically imputed target matrix.

    Args:
        X (np.ndarray): Target matrix of shape (m, n).
        k (int): Number of latent components.
        n (int): Maximum number of iterations. Defaults to 100.
        tol (float): Convergence tolerance. Defaults to 1e-3.
        seed (int, optional): Random seed. Defaults to None.
        r (int): Restarts for initialization. Defaults to 20.
        eval_every (int): Frequency of checking early-stopping criteria. Defaults to 10.

    Returns:
        tuple: (rec, W, H, cost)
            - rec (np.ndarray): Reconstructed matrix.
            - W (np.ndarray): Left factor matrix.
            - H (np.ndarray): Right factor matrix.
            - cost (float): Final IS cost.
    """

    if isinstance(seed, int):
        np.random.seed(seed)

    M = np.isnan(X)
    X = X.copy()
    X[M] = 0
    M = ~M
    X_zero = X == 0

    rows, columns = X.shape
    eps = np.finfo(float).eps

    W = np.abs(np.random.uniform(size=(rows, k)))
    W = np.maximum(W, eps)
    W = np.divide(W, k * W.max())

    H = np.abs(np.random.uniform(size=(k, columns)))
    H = np.maximum(H, eps)
    H = np.divide(H, k * H.max())

    cost_mask = M & (X > 0.0)

    if seed is None:
        rec = W @ H
        cost = cost_is(X, rec, M)
        for _ in range(r):
            Wt = np.abs(np.random.uniform(size=(rows, k)))
            Wt = np.maximum(Wt, eps)
            Wt = np.divide(Wt, k * Wt.max())
            Ht = np.abs(np.random.uniform(size=(k, columns)))
            Ht = np.maximum(Ht, eps)
            Ht = np.divide(Ht, k * Ht.max())
            rec_temp = Wt @ Ht
            cost_temp = cost_is(X, rec_temp, M)
            if cost_temp < cost:
                W = Wt
                H = Ht
                cost = cost_temp

    rec = W @ H
    cost = cost_is(X, rec, cost_mask)

    for idx in range(n):
        # Update H row-by-row
        for i in range(k):
            rec = W @ H
            X_imp = np.where(X_zero, rec, X)
            numerator = W[:, i] @ (X_imp / (rec**2))
            denominator = W[:, i] @ (1.0 / rec)
            H[i, :] = H[i, :] * np.sqrt(numerator / (denominator + eps))
            H[i, :] = np.maximum(H[i, :], eps)

        # Update W column-by-column
        for i in range(k):
            rec = W @ H
            X_imp = np.where(X_zero, rec, X)
            numerator = (X_imp / (rec**2)) @ H[i, :]
            denominator = (1.0 / rec) @ H[i, :]
            W[:, i] = W[:, i] * np.sqrt(numerator / (denominator + eps))
            W[:, i] = np.maximum(W[:, i], eps)

        if idx % eval_every == 0:
            rec = W @ H
            cost = cost_is(X, rec, cost_mask)
            if cost <= tol:
                break
    else:
        rec = W @ H
        cost = cost_is(X, rec, cost_mask)

    X[~M] = np.nan
    return rec, W, H, cost
