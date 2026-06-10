# coding: utf-8
"""
pyNNMF: Non-Negative Matrix Factorization with missing values imputation.
Provides various solvers (MU, ALS, HALS) for different cost functions
(Frobenius norm, Kullback-Leibler, and Itakura-Saito).
"""

__author__ = "Mário Antunes"
__version__ = "0.2.0"
__email__ = "mario.antunes@ua.pt"
__status__ = "Development"


import logging
from typing import Literal, Optional, Callable, Tuple

import numpy as np
import numpy.typing as np_typing

from .initialization import initialize_nmf

logger = logging.getLogger(__name__)


def rwnmf(
    x: np_typing.NDArray[np.floating],
    k: int,
    alpha: float = 0.1,
    tol_fit_improvement: float = 1e-4,
    tol_fit_error: float = 1e-4,
    num_iter: int = 1000,
    seed: int | None = None,
    eval_every: int = 10,
    init: Literal["random", "nndsvd", "svd_impute"]
    | Callable[
        [np_typing.NDArray[np.floating], int], tuple[np_typing.NDArray[np.floating], np_typing.NDArray[np.floating]]
    ] = "random",
) -> tuple[np_typing.NDArray[np.floating], np_typing.NDArray[np.floating], np_typing.NDArray[np.floating], float]:
    """
    Robust Weighted Non-negative Matrix Factorization (RWNMF) with L2 regularization.

    Optimizes the Frobenius norm objective function with missing values and L2 regularization:

    $$D_{\\text{Fro}}(X \\parallel U V^T) = \\frac{1}{2} \\| M \\odot (X - U V^T) \\|_F^2 + \\frac{\\alpha}{2} (\\|U\\|_F^2 + \\|V\\|_F^2)$$

    where $M$ is a binary mask representing observed values (1 if observed, 0 if NaN).

    Optimization method: Multiplicative Update (MU) rules.

    $$U \\leftarrow U \\odot \\frac{(M \\odot X) V}{(M \\odot (U V^T)) V + \\alpha U}$$

    $$V \\leftarrow V \\odot \\frac{(M \\odot X)^T U}{(M \\odot (U V^T))^T U + \\alpha V}$$

    Args:
        x (np.ndarray): Target matrix of shape (m, n) with potential NaN values.
        k (int): Number of latent components.
        alpha (float): L2 regularization parameter. Defaults to 0.1.
        tol_fit_improvement (float): Early stopping tolerance for reconstruction improvement. Defaults to 1e-4.
        tol_fit_error (float): Early stopping tolerance for reconstruction error. Defaults to 1e-4.
        num_iter (int): Maximum number of iterations. Defaults to 1000.
        seed (int, optional): Random seed for reproducibility. Defaults to None.
        eval_every (int): Frequency of checking early-stopping criteria. Defaults to 10.
        init (str or callable): Initialization method ('random', 'nndsvd', 'svd_impute')
                                or a custom callable. Defaults to 'random'.

    Returns:
        tuple: (xr, u, v, error)
            - xr (np.ndarray): Reconstructed matrix of shape (m, n).
            - u (np.ndarray): Left factor matrix of shape (m, k).
            - v (np.ndarray): Right factor matrix of shape (n, k).
            - error (float): Final Frobenius reconstruction error.
    """
    eps = np.finfo(float).eps
    early_stop = False

    m_mask = np.isnan(x)
    x_copy = x.copy()
    x_copy[m_mask] = 0
    m_not = ~m_mask

    u, v = initialize_nmf(x, k, method=init, seed=seed)
    v: np_typing.NDArray[np.floating] = v.T

    xr: np_typing.NDArray[np.floating] = np.inf * np.ones(x.shape)

    m_float = m_not.astype(float)
    m_x = m_float * x_copy
    m_x_t = m_x.T

    # Initial rec
    rec: np_typing.NDArray[np.floating] = u @ v.T

    for i in range(num_iter):
        # 1. Update u
        rec = rec * m_float
        u: np_typing.NDArray[np.floating] = u * ((m_x @ v) / (rec @ v + alpha * u))
        u = np.maximum(u, eps)

        # 2. Update v
        rec = u @ v.T
        rec = rec * m_float
        v: np_typing.NDArray[np.floating] = v * ((m_x_t @ u) / (rec.T @ u + alpha * v))
        v = np.maximum(v, eps)

        # residuals evaluation
        rec = u @ v.T
        if i % eval_every == 0:
            fit_error: float = float(np.linalg.norm(x_copy - rec, "fro"))
            fit_improvement: float = float(np.linalg.norm(rec - xr, "fro"))
            xr = np.copy(rec)

            if fit_error < tol_fit_error or fit_improvement < tol_fit_improvement:
                early_stop = True
                break

    if early_stop:
        xr = rec
    else:
        xr = u @ v.T

    error = np.linalg.norm(x_copy - xr, ord="fro")
    return xr, u, v, float(error)


def cost_fb(
    a: np_typing.NDArray[np.floating], b: np_typing.NDArray[np.floating], m: np_typing.NDArray[np.floating] | None = None
) -> float:
    """
    Compute the Frobenius norm cost between two matrices, accounting for missing values.

    $$D_{\\text{Fro}}(A \\parallel B) = \\| M \\odot (A - B) \\|_F$$

    where $M$ is a mask indicating observed entries.

    Args:
        a (np.ndarray): Target matrix.
        b (np.ndarray): Reconstructed matrix.
        m (np.ndarray, optional): Binary mask of observed elements. Defaults to None.

    Returns:
        float: Frobenius cost.
    """
    if m is None:
        m_mask = np.isnan(a)
        a_copy = a.copy()
        a_copy[m_mask] = 0
        m_not = (~m_mask).astype(float)
        cost = np.linalg.norm((m_not * a_copy) - (m_not * b), ord="fro")
    else:
        cost = np.linalg.norm((m * a) - (m * b), ord="fro")
    return float(cost)


def cost_kl(
    a: np_typing.NDArray[np.floating], b: np_typing.NDArray[np.floating], m: np_typing.NDArray[np.bool_] | None = None
) -> float:
    """
    Compute the Kullback-Leibler (KL) divergence cost between two matrices.

    $$D_{\\text{KL}}(A \\parallel B) = \\sum_{i,j} M_{ij} \\left( A_{ij} \\log \\frac{A_{ij}}{B_{ij}} - A_{ij} + B_{ij} \\right)$$

    Args:
        a (np.ndarray): Target matrix.
        b (np.ndarray): Reconstructed matrix.
        m (np.ndarray, optional): Binary mask of observed elements. Defaults to None.

    Returns:
        float: KL divergence cost.
    """
    if m is None:
        m = ~np.isnan(a)
    mask = m & (a > 0.0)
    a_vals = a[mask]
    b_vals = b[mask]
    term_pos = a_vals * np.log(a_vals / b_vals) - a_vals
    term_b = b[m]
    return float(np.sum(term_pos) + np.sum(term_b))


def cost_is(
    a: np_typing.NDArray[np.floating], b: np_typing.NDArray[np.floating], m: np_typing.NDArray[np.bool_] | None = None
) -> float:
    """
    Compute the Itakura-Saito (IS) divergence cost between two matrices.

    $$D_{\\text{IS}}(A \\parallel B) = \\sum_{i,j} M_{ij} \\left( \\frac{A_{ij}}{B_{ij}} - \\log \\frac{A_{ij}}{B_{ij}} - 1 \\right)$$

    Args:
        a (np.ndarray): Target matrix.
        b (np.ndarray): Reconstructed matrix.
        m (np.ndarray, optional): Binary mask of observed elements. Defaults to None.

    Returns:
        float: IS divergence cost.
    """
    if m is None:
        m = ~np.isnan(a)
    mask = m & (a > 0.0)
    a_vals = a[mask]
    b_vals = b[mask]
    return float(np.sum((a_vals / b_vals) - np.log(a_vals / b_vals) - 1))




def nmf_mu(
    x: np_typing.NDArray[np.floating],
    k: int,
    cost: Literal["frobenius", "kl", "is"] = "frobenius",
    alpha: float = 1e-3,
    tol: float = 1e-3,
    n: int = 1000,
    seed: int | None = None,
    eval_every: int = 10,
    init: Literal["random", "nndsvd", "svd_impute"]
    | Callable[
        [np_typing.NDArray[np.floating], int], tuple[np_typing.NDArray[np.floating], np_typing.NDArray[np.floating]]
    ] = "random",
    callback: Optional[Callable[[int, float], None]] = None,
) -> Tuple[np.typing.NDArray[np.floating], np.typing.NDArray[np.floating], np.typing.NDArray[np.floating], float]:
    """
    Non-Negative Matrix Factorization (NMF) via Multiplicative Updates.

    This is the main NMF solver with configurable cost function.

    Args:
        x: Target matrix of shape (m, n)
        k: Number of latent components
        cost: Cost function: "frobenius", "kl", or "is"
        alpha: L2 regularization parameter (only used for 'frobenius' cost)
        tol: Convergence tolerance
        n: Maximum iterations
        seed: Random seed
        eval_every: Evaluation frequency
        init: Initialization method
        callback: Function called with (iteration, cost)

    Returns:
        Tuple of (rec, w, h, cost)

    Raises:
        ValueError: If cost is not "frobenius", "kl", or "is"
    """
    if cost == "frobenius":
        return _nmf_mu_frobenius(x, k, alpha, tol, n, seed, eval_every, init, callback)
    elif cost == "kl":
        return _nmf_mu_kl(x, k, alpha, tol, n, seed, eval_every, init, callback)
    elif cost == "is":
        return _nmf_mu_is(x, k, alpha, tol, n, seed, eval_every, init, callback)
    else:
        raise ValueError(f"cost must be 'frobenius', 'kl', or 'is', got '{cost}'")


def _nmf_mu_frobenius(
    x: np_typing.NDArray[np.floating],
    k: int,
    alpha: float,
    tol: float,
    n: int,
    seed: int | None,
    eval_every: int,
    init: Literal["random", "nndsvd", "svd_impute"]
    | Callable[
        [np_typing.NDArray[np.floating], int], tuple[np_typing.NDArray[np.floating], np_typing.NDArray[np.floating]]
    ],
    callback: Optional[Callable[[int, float], None]],
) -> Tuple[np.typing.NDArray[np.floating], np.typing.NDArray[np.floating], np.typing.NDArray[np.floating], float]:
    """
    Internal implementation for Frobenius norm Multiplicative Updates.

    Optimizes the Frobenius norm objective function with missing values and L2 regularization:

    $$D_{\\text{Fro}}(X \\parallel W H) = \\frac{1}{2} \\| M \\odot (X - W H) \\|_F^2 + \\frac{\\alpha}{2} (\\|W\\|_F^2 + \\|H\\|_F^2)$$

    where $M$ is a binary mask representing observed values (1 if observed, 0 if NaN).

    Update rules:

    $$W \\leftarrow \\max\\left(\\epsilon, W \\odot \\frac{(M \\odot X) H^T}{(M \\odot (W H)) H^T + \\alpha W}\\right)$$

    $$H \\leftarrow \\max\\left(\\epsilon, H \\odot \\frac{W^T (M \\odot X)}{W^T (M \\odot (W H)) + \\alpha H}\\right)$$
    """
    eps = 1e-9
    w, h = initialize_nmf(x, k, method=init, seed=seed)
    m_mask = np.isnan(x)
    x_copy = x.copy()
    x_copy[m_mask] = 0
    m_not = ~m_mask
    m_float = m_not.astype(float)
    m_x: np_typing.NDArray[np.floating] = m_float * x_copy
    rec: np_typing.NDArray[np.floating] = w @ h
    cost_val = cost_fb(x_copy, rec, m_float)
    for idx in range(n):
        rec = w @ h
        rec_masked: np_typing.NDArray[np.floating] = rec * m_float
        w = np.maximum(eps, w * ((m_x @ h.T) / (rec_masked @ h.T + alpha * w)))
        rec = w @ h
        rec_masked = rec * m_float
        h = np.maximum(eps, h * ((w.T @ m_x) / (w.T @ rec_masked + alpha * h)))
        rec = w @ h
        cost_val = cost_fb(x_copy, rec, m_float)
        if callback:
            callback(idx, cost_val)
        if idx % eval_every == 0:
            if cost_val <= tol:
                break
    else:
        rec = w @ h
        cost_val = cost_fb(x_copy, rec, m_float)
    x_copy[~m_not] = np.nan
    return rec, w, h, float(cost_val)


def _nmf_mu_kl(
    x: np_typing.NDArray[np.floating],
    k: int,
    alpha: float,
    tol: float,
    n: int,
    seed: int | None,
    eval_every: int,
    init: Literal["random", "nndsvd", "svd_impute"]
    | Callable[
        [np_typing.NDArray[np.floating], int], tuple[np_typing.NDArray[np.floating], np_typing.NDArray[np.floating]]
    ],
    callback: Optional[Callable[[int, float], None]],
) -> Tuple[np.typing.NDArray[np.floating], np.typing.NDArray[np.floating], np.typing.NDArray[np.floating], float]:
    """
    Internal implementation for Kullback-Leibler (KL) divergence Multiplicative Updates.

    Optimizes the KL divergence objective function with missing values imputed dynamically:

    $$D_{\\text{KL}}(X \\parallel W H) = \\sum_{i,j} M_{ij} \\left( X_{ij} \\log \\frac{X_{ij}}{(W H)_{ij}} - X_{ij} + (W H)_{ij} \\right)$$

    where NaNs in $X$ are dynamically imputed:

    $$X_{\\text{imp}} = M \\odot X + (1 - M) \\odot (W H)$$

    Update rules:

    $$H \\leftarrow \\max\\left(\\epsilon, H \\odot \\frac{W^T \\frac{X_{\\text{imp}}}{W H}}{W^T \\mathbf{1}}\\right)$$

    $$W \\leftarrow \\max\\left(\\epsilon, W \\odot \\frac{\\frac{X_{\\text{imp}}}{W H} H^T}{\\mathbf{1} H^T}\\right)$$
    """
    eps = 1e-9
    w, h = initialize_nmf(x, k, method=init, seed=seed)
    m_mask = np.isnan(x)
    x_copy = x.copy()
    x_copy[m_mask] = 0
    m_not = ~m_mask
    cost_mask = m_not
    x_zero: np_typing.NDArray[np.bool_] = m_mask
    rec: np_typing.NDArray[np.floating] = w @ h
    cost_val = cost_kl(x_copy, rec, cost_mask)
    for idx in range(n):
        rec = w @ h
        x_imp: np_typing.NDArray[np.floating] = np.where(x_zero, rec, x_copy)
        ratio: np_typing.NDArray[np.floating] = x_imp / rec
        w_sum: np_typing.NDArray[np.floating] = w.sum(axis=0)[:, None]
        h = np.maximum(eps, h * (w.T @ ratio / w_sum))
        rec = w @ h
        x_imp = np.where(x_zero, rec, x_copy)
        ratio = x_imp / rec
        h_sum: np_typing.NDArray[np.floating] = h.sum(axis=1)
        w = np.maximum(eps, w * ((ratio @ h.T) / h_sum))
        rec = w @ h
        cost_val = cost_kl(x_copy, rec, cost_mask)
        if callback:
            callback(idx, cost_val)
        if idx % eval_every == 0:
            if cost_val <= tol:
                break
    else:
        rec = w @ h
        cost_val = cost_kl(x_copy, rec, cost_mask)
    x_copy[~m_not] = np.nan
    return rec, w, h, float(cost_val)


def _nmf_mu_is(
    x: np_typing.NDArray[np.floating],
    k: int,
    alpha: float,
    tol: float,
    n: int,
    seed: int | None,
    eval_every: int,
    init: Literal["random", "nndsvd", "svd_impute"]
    | Callable[
        [np_typing.NDArray[np.floating], int], tuple[np_typing.NDArray[np.floating], np_typing.NDArray[np.floating]]
    ],
    callback: Optional[Callable[[int, float], None]],
) -> Tuple[np.typing.NDArray[np.floating], np.typing.NDArray[np.floating], np.typing.NDArray[np.floating], float]:
    """
    Internal implementation for Itakura-Saito (IS) divergence Multiplicative Updates.

    Optimizes the IS divergence objective function with missing values imputed dynamically:

    $$D_{\\text{IS}}(X \\parallel W H) = \\sum_{i,j} M_{ij} \\left( \\frac{X_{ij}}{(W H)_{ij}} - \\log \\frac{X_{ij}}{(W H)_{ij}} - 1 \\right)$$

    where NaNs in $X$ are dynamically imputed:

    $$X_{\\text{imp}} = M \\odot X + (1 - M) \\odot (W H)$$

    Update rules:

    $$H \\leftarrow \\max\\left(\\epsilon, H \\odot \\sqrt{\\frac{W^T \\left( \\frac{X_{\\text{imp}}}{(W H)^2} \\right)}{W^T \\left( \\frac{1}{W H} \\right)}}\\right)$$

    $$W \\leftarrow \\max\\left(\\epsilon, W \\odot \\sqrt{\\frac{\\left( \\frac{X_{\\text{imp}}}{(W H)^2} \\right) H^T}{\\left( \\frac{1}{W H} \\right) H^T}}\\right)$$
    """
    eps = 1e-9
    w, h = initialize_nmf(x, k, method=init, seed=seed)
    m_mask = np.isnan(x)
    x_copy = x.copy()
    x_copy[m_mask] = 0
    m_not = ~m_mask
    cost_mask = m_not & (x_copy > 0.0)
    x_zero: np_typing.NDArray[np.bool_] = m_mask
    rec: np_typing.NDArray[np.floating] = w @ h
    cost_val = cost_is(x_copy, rec, cost_mask)
    for idx in range(n):
        rec = w @ h
        x_imp: np_typing.NDArray[np.floating] = np.where(x_zero, rec, x_copy)
        ratio: np_typing.NDArray[np.floating] = x_imp / rec
        rec_sum_rows: np_typing.NDArray[np.floating] = rec.sum(axis=1)[:, None]
        w_scaled: np_typing.NDArray[np.floating] = w / rec_sum_rows
        numerator: np_typing.NDArray[np.floating] = w_scaled.T @ ratio
        denominator: np_typing.NDArray[np.floating] = w_scaled.sum(axis=0)[:, None]
        h = np.maximum(eps, h * np.sqrt(numerator / denominator))
        rec = w @ h
        x_imp = np.where(x_zero, rec, x_copy)
        ratio = x_imp / rec
        rec_sum_cols: np_typing.NDArray[np.floating] = rec.sum(axis=0)
        h_scaled: np_typing.NDArray[np.floating] = h / rec_sum_cols
        numerator = ratio @ h_scaled.T
        denominator = h_scaled.sum(axis=1)
        w = np.maximum(eps, w * np.sqrt(numerator / denominator))
        rec = w @ h
        cost_val = cost_is(x_copy, rec, cost_mask)
        if callback:
            callback(idx, cost_val)
        if idx % eval_every == 0:
            if cost_val <= tol:
                break
    else:
        rec = w @ h
        cost_val = cost_is(x_copy, rec, cost_mask)
    x_copy[~m_not] = np.nan
    return rec, w, h, float(cost_val)


def nmf_mu_kl(
    x: np_typing.NDArray[np.floating],
    k: int,
    n: int = 100,
    tol: float = 1e-3,
    seed: int | None = None,
    r: int = 20,
    eval_every: int = 10,
    init: Literal["random", "nndsvd", "svd_impute"]
    | Callable[
        [np_typing.NDArray[np.floating], int], tuple[np_typing.NDArray[np.floating], np_typing.NDArray[np.floating]]
    ] = "random",
    callback: Callable[[int, float], None] | None = None,
) -> Tuple[np.typing.NDArray[np.floating], np.typing.NDArray[np.floating], np.typing.NDArray[np.floating], float]:
    """Deprecated: Use nmf_mu(cost='kl', ...) instead."""
    return _nmf_mu_kl(x, k, alpha=1e-3, tol=tol, n=n, seed=seed, 
                     eval_every=eval_every, init=init, callback=callback)


def nmf_mu_is(
    x: np_typing.NDArray[np.floating],
    k: int,
    n: int = 100,
    tol: float = 1e-3,
    seed: int | None = None,
    r: int = 20,
    eval_every: int = 10,
    init: Literal["random", "nndsvd", "svd_impute"]
    | Callable[
        [np_typing.NDArray[np.floating], int], tuple[np_typing.NDArray[np.floating], np_typing.NDArray[np.floating]]
    ] = "random",
    callback: Callable[[int, float], None] | None = None,
) -> Tuple[np.typing.NDArray[np.floating], np.typing.NDArray[np.floating], np.typing.NDArray[np.floating], float]:
    """Deprecated: Use nmf_mu(cost='is', ...) instead."""
    return _nmf_mu_is(x, k, alpha=1e-3, tol=tol, n=n, seed=seed, 
                     eval_every=eval_every, init=init, callback=callback)
def nmf_als(
    x: np_typing.NDArray[np.floating],
    k: int,
    n: int = 1000,
    tol: float = 1e-3,
    seed: int | None = None,
    eval_every: int = 10,
    init: Literal["random", "nndsvd", "svd_impute"]
    | Callable[
        [np_typing.NDArray[np.floating], int], tuple[np_typing.NDArray[np.floating], np_typing.NDArray[np.floating]]
    ] = "random",
    callback: Callable[[int, float], None] | None = None,
) -> tuple[np_typing.NDArray[np.floating], np_typing.NDArray[np.floating], np_typing.NDArray[np.floating], float]:
    """
    Alternating Least Squares (ALS) NMF with missing values imputation.

    Minimizes the Frobenius norm. During each iteration, missing entries (NaNs)
    are imputed dynamically from the current reconstruction:

    $$X_{\\text{imp}} = M \\odot X + (1 - M) \\odot (W H)$$

    Then, least squares solutions are computed alternately:

    $$W \\leftarrow \\max\\left(\\epsilon, X_{\\text{imp}} H^T (H H^T + 10^{-9} I_k)^{-1}\\right)$$

    $$H \\leftarrow \\max\\left(\\epsilon, (W^T W + 10^{-9} I_k)^{-1} W^T X_{\\text{imp}}\\right)$$

    Args:
        x (np.ndarray): Target matrix of shape (m, n).
        k (int): Number of latent components.
        n (int): Maximum number of iterations. Defaults to 1000.
        tol (float): Convergence tolerance. Defaults to 1e-3.
        seed (int, optional): Random seed. Defaults to None.
        eval_every (int): Frequency of checking early-stopping criteria. Defaults to 10.
        init (str or callable): Initialization method ('random', 'nndsvd', 'svd_impute')
                                or a custom callable. Defaults to 'random'.
        callback (callable, optional): Function called at each iteration with (iteration, cost). Defaults to None.

    Returns:
        tuple: (rec, w, h, cost)
            - rec (np.ndarray): Reconstructed matrix.
            - w (np.ndarray): Left factor matrix.
            - h (np.ndarray): Right factor matrix.
            - cost (float): Final Frobenius cost.
    """
    eps = 1e-9
    w, h = initialize_nmf(x, k, method=init, seed=seed)

    m_mask = np.isnan(x)
    x_copy = x.copy()
    x_copy[m_mask] = 0
    m_not = ~m_mask
    m_float = m_not.astype(float)
    x_zero = m_mask

    rec: np_typing.NDArray[np.floating] = w @ h
    cost = cost_fb(x_copy, rec, m_float)

    for idx in range(n):
        # Imputation
        rec = w @ h
        x_imp: np_typing.NDArray[np.floating] = np.where(x_zero, rec, x_copy)

        # Update w
        w = np.maximum(eps, np.linalg.solve(h @ h.T + 1e-9 * np.eye(k), h @ x_imp.T).T)

        # Imputation
        rec = w @ h
        x_imp = np.where(x_zero, rec, x_copy)

        # Update h
        h = np.maximum(eps, np.linalg.solve(w.T @ w + 1e-9 * np.eye(k), w.T @ x_imp))

        # Evaluation and callback
        rec = w @ h
        cost = cost_fb(x_copy, rec, m_float)
        if callback:
            callback(idx, cost)

        if idx % eval_every == 0:
            if cost <= tol:
                break
    else:
        rec = w @ h
        cost = cost_fb(x_copy, rec, m_float)

    x_copy[~m_not] = np.nan
    return rec, w, h, float(cost)


def nmf_hals(
    x: np_typing.NDArray[np.floating],
    k: int,
    n: int = 1000,
    tol: float = 1e-3,
    seed: int | None = None,
    eval_every: int = 10,
    init: Literal["random", "nndsvd", "svd_impute"]
    | Callable[
        [np_typing.NDArray[np.floating], int], tuple[np_typing.NDArray[np.floating], np_typing.NDArray[np.floating]]
    ] = "random",
    callback: Callable[[int, float], None] | None = None,
) -> tuple[np_typing.NDArray[np.floating], np_typing.NDArray[np.floating], np_typing.NDArray[np.floating], float]:
    """
    Hierarchical Alternating Least Squares (HALS) NMF with missing values imputation.

    Minimizes the Frobenius norm. Computes dynamic imputation as in ALS (imputing only NaNs),
    then updates factors column-by-column (for w) and row-by-row (for h):

    $$W_{*, i} \\leftarrow \\max\\left(\\epsilon, W_{*, i} + \\frac{(X_{\\text{imp}} H^T)_{*, i} - W (H H^T)_{*, i}}{(H H^T)_{i,i}}\\right)$$

    $$H_{i, *} \\leftarrow \\max\\left(\\epsilon, H_{i, *} + \\frac{(W^T X_{\\text{imp}})_{i, *} - (W^T W)_{i, *} H}{(W^T W)_{i,i} }\\right)$$

    Args:
        x (np.ndarray): Target matrix of shape (m, n).
        k (int): Number of latent components.
        n (int): Maximum number of iterations. Defaults to 1000.
        tol (float): Convergence tolerance. Defaults to 1e-3.
        seed (int, optional): Random seed. Defaults to None.
        eval_every (int): Frequency of checking early-stopping criteria. Defaults to 10.
        init (str or callable): Initialization method ('random', 'nndsvd', 'svd_impute')
                                or a custom callable. Defaults to 'random'.
        callback (callable, optional): Function called at each iteration with (iteration, cost). Defaults to None.

    Returns:
        tuple: (rec, w, h, cost)
            - rec (np.ndarray): Reconstructed matrix.
            - w (np.ndarray): Left factor matrix.
            - h (np.ndarray): Right factor matrix.
            - cost (float): Final Frobenius cost.
    """
    eps = 1e-9
    w, h = initialize_nmf(x, k, method=init, seed=seed)

    m_mask = np.isnan(x)
    x_copy = x.copy()
    x_copy[m_mask] = 0
    m_not = ~m_mask
    m_float = m_not.astype(float)
    x_zero = m_mask

    rec = w @ h
    cost = cost_fb(x_copy, rec, m_float)

    for idx in range(n):
        # Imputation
        rec = w @ h
        x_imp = np.where(x_zero, rec, x_copy)

        # Update w column-wise
        a_mat = h @ h.T
        b_mat = x_imp @ h.T
        for i in range(k):
            denom = a_mat[i, i] if a_mat[i, i] > eps else eps
            w[:, i] = np.maximum(eps, w[:, i] + (b_mat[:, i] - w @ a_mat[:, i]) / denom)

        # Imputation
        rec = w @ h
        x_imp = np.where(x_zero, rec, x_copy)

        # Update h row-wise
        c_mat = w.T @ w
        d_mat = w.T @ x_imp
        for i in range(k):
            denom = c_mat[i, i] if c_mat[i, i] > eps else eps
            h[i, :] = np.maximum(eps, h[i, :] + (d_mat[i, :] - c_mat[i, :] @ h) / denom)

        # Evaluation and callback
        rec = w @ h
        cost = cost_fb(x_copy, rec, m_float)
        if callback:
            callback(idx, cost)

        if idx % eval_every == 0:
            if cost <= tol:
                break
    else:
        rec = w @ h
        cost = cost_fb(x_copy, rec, m_float)

    x_copy[~m_not] = np.nan
    return rec, w, h, float(cost)

