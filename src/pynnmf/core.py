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
import typing

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
    init: typing.Literal["random", "nndsvd", "svd_impute"]
    | typing.Callable[
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
        if np.any(np.isnan(a)):
            m_mask = np.isnan(a)
            a_copy = a.copy()
            a_copy[m_mask] = 0
            m_not = ~m_mask
            cost = np.linalg.norm((m_not * a_copy) - (m_not * b), ord="fro")
        else:
            m_not = a > 0.0
            cost = np.linalg.norm((m_not * a) - (m_not * b), ord="fro")
    else:
        cost = np.linalg.norm((m * a) - (m * b), ord="fro")
    return float(cost)


def nmf_mu(
    x: np_typing.NDArray[np.floating],
    k: int,
    n: int = 1000,
    alpha: float = 1e-3,
    tol: float = 1e-3,
    seed: int | None = None,
    eval_every: int = 10,
    init: typing.Literal["random", "nndsvd", "svd_impute"]
    | typing.Callable[
        [np_typing.NDArray[np.floating], int], tuple[np_typing.NDArray[np.floating], np_typing.NDArray[np.floating]]
    ] = "random",
    callback: typing.Callable[[int, float], None] | None = None,
) -> tuple[np_typing.NDArray[np.floating], np_typing.NDArray[np.floating], np_typing.NDArray[np.floating], float]:
    """
    Non-Negative Matrix Factorization (NMF) via Multiplicative Updates (MU) using Frobenius norm.

    Optimizes the Frobenius norm objective function with L2 regularization:

    $$D_{\\text{Fro}}(X \\parallel W H) = \\frac{1}{2} \\| M \\odot (X - W H) \\|_F^2 + \\frac{\\alpha}{2} (\\|W\\|_F^2 + \\|H\\|_F^2)$$

    Multiplicative update steps:

    $$W \\leftarrow W \\odot \frac{(M \\odot X) H^T}{(M \\odot (W H)) H^T + \\alpha W}$$

    $$H \\leftarrow H \\odot \frac{W^T (M \\odot X)}{W^T (M \\odot (W H)) + \\alpha H}$$

    Args:
        x (np.ndarray): Target matrix of shape (m, n).
        k (int): Number of latent components.
        n (int): Maximum number of iterations. Defaults to 1000.
        alpha (float): L2 regularization parameter. Defaults to 1e-3.
        tol (float): Convergence tolerance. Defaults to 1e-3.
        seed (int, optional): Random seed. Defaults to None.
        eval_every (int): Frequency of checking early-stopping criteria. Defaults to 10.
        init (str or callable): Initialization method ('random', 'nndsvd', 'svd_impute')
                                or a custom callable. Defaults to 'random'.

    Returns:
        tuple: (rec, w, h, cost)
            - rec (np.ndarray): Reconstructed matrix of shape (m, n).
            - w (np.ndarray): Left factor matrix of shape (m, k).
            - h (np.ndarray): Right factor matrix of shape (k, n).
            - cost (float): Final Frobenius cost.
    """
    eps = np.finfo(float).eps

    w, h = initialize_nmf(x, k, method=init, seed=seed)

    m_mask = np.isnan(x)
    x_copy = x.copy()
    x_copy[m_mask] = 0
    m_not = ~m_mask
    m_float = m_not.astype(float)

    # Precompute constant matrices
    m_x: np_typing.NDArray[np.floating] = m_float * x_copy
    rec: np_typing.NDArray[np.floating] = w @ h
    cost = cost_fb(x_copy, rec, m_float)

    for idx in range(n):
        # 1. Update w
        rec = w @ h
        rec_masked: np_typing.NDArray[np.floating] = rec * m_float
        w = w * ((m_x @ h.T) / (rec_masked @ h.T + alpha * w))
        w = np.maximum(w, eps)

        # 2. Update h
        rec = w @ h
        rec_masked: np_typing.NDArray[np.floating] = rec * m_float
        h = h * ((w.T @ m_x) / (w.T @ rec_masked + alpha * h))
        h = np.maximum(h, eps)

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
        if np.any(np.isnan(a)):
            m_mask = np.isnan(a)
            m = ~m_mask
        else:
            m = a > 0.0
    mask = m & (a > 0.0)
    a_vals = a[mask]
    b_vals = b[mask]
    return float(np.sum(a_vals * np.log(a_vals / b_vals) - a_vals + b_vals))


def nmf_mu_kl(
    x: np_typing.NDArray[np.floating],
    k: int,
    n: int = 100,
    tol: float = 1e-3,
    seed: int | None = None,
    r: int = 20,
    eval_every: int = 10,
    init: typing.Literal["random", "nndsvd", "svd_impute"]
    | typing.Callable[
        [np_typing.NDArray[np.floating], int], tuple[np_typing.NDArray[np.floating], np_typing.NDArray[np.floating]]
    ] = "random",
    callback: typing.Callable[[int, float], None] | None = None,
) -> tuple[np_typing.NDArray[np.floating], np_typing.NDArray[np.floating], np_typing.NDArray[np.floating], float]:
    """
    NMF via Multiplicative Updates (MU) minimizing the Kullback-Leibler (KL) divergence.

    Optimizes the generalized KL divergence objective:

    $$D_{\\text{KL}}(X \\parallel W H) = \\sum_{i,j} M_{ij} \\left( X_{ij} \\log \\frac{X_{ij}}{(W H)_{ij}} - X_{ij} + (W H)_{ij} \\right)$$

    Multiplicative update steps:

    $$H \\leftarrow H \\odot \\frac{W^T \\left( \\frac{M \\odot X}{W H} \\right)}{W^T M}$$

    $$W \\leftarrow W \\odot \\frac{\\left( \\frac{M \\odot X}{W H} \\right) H^T}{M H^T}$$

    Args:
        x (np.ndarray): Target matrix of shape (m, n).
        k (int): Number of latent components.
        n (int): Maximum number of iterations. Defaults to 100.
        tol (float): Convergence tolerance. Defaults to 1e-3.
        seed (int, optional): Random seed. Defaults to None.
        r (int): Number of restarts for initial factor selection. Defaults to 20.
        eval_every (int): Frequency of checking early-stopping criteria. Defaults to 10.
        init (str or callable): Initialization method ('random', 'nndsvd', 'svd_impute')
                                or a custom callable. Defaults to 'random'.
        callback (callable, optional): Function called at each iteration with (iteration, cost). Defaults to None.

    Returns:
        tuple: (rec, w, h, cost)
            - rec (np.ndarray): Reconstructed matrix of shape (m, n).
            - w (np.ndarray): Left factor matrix of shape (m, k).
            - h (np.ndarray): Right factor matrix of shape (k, n).
            - cost (float): Final KL cost.
    """
    eps = np.finfo(float).eps

    w, h = initialize_nmf(x, k, method=init, seed=seed)

    # Create a Mask
    m_mask = np.isnan(x)
    x_copy = x.copy()
    x_copy[m_mask] = 0
    m_not = ~m_mask

    cost_mask = m_not & (x_copy > 0.0)

    if seed is None and init == "random":
        rec = w @ h
        cost = cost_kl(x_copy, rec, m_not)

        for _ in range(r):
            wt, ht = initialize_nmf(x, k, method="random")
            rec_temp = wt @ ht
            cost_temp = cost_kl(x_copy, rec_temp, m_not)

            if cost_temp < cost:
                w = wt
                h = ht
                cost = cost_temp

    x_zero: np_typing.NDArray[np.bool_] = x_copy == 0
    rec: np_typing.NDArray[np.floating] = w @ h
    cost = cost_kl(x_copy, rec, cost_mask)

    for idx in range(n):
        # 1. Update h
        rec = w @ h
        x_imp: np_typing.NDArray[np.floating] = np.where(x_zero, rec, x_copy)
        ratio: np_typing.NDArray[np.floating] = x_imp / rec
        w_sum: np_typing.NDArray[np.floating] = w.sum(axis=0)[:, None]
        h = h * (w.T @ ratio / w_sum)
        h = np.maximum(h, eps)

        # 2. Update w
        rec = w @ h
        x_imp = np.where(x_zero, rec, x_copy)
        ratio = x_imp / rec
        h_sum: np_typing.NDArray[np.floating] = h.sum(axis=1)
        w = w * ((ratio @ h.T) / h_sum)
        w = np.maximum(w, eps)

        # Evaluation and callback
        rec = w @ h
        cost = cost_kl(x_copy, rec, cost_mask)
        if callback:
            callback(idx, cost)

        if idx % eval_every == 0:
            if cost <= tol:
                break
    else:
        rec = w @ h
        cost = cost_kl(x_copy, rec, cost_mask)

    x_copy[~m_not] = np.nan
    return rec, w, h, float(cost)


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
        if np.any(np.isnan(a)):
            m_mask = np.isnan(a)
            m = ~m_mask
        else:
            m = a > 0.0
    mask = m & (a > 0.0)
    a_vals = a[mask]
    b_vals = b[mask]
    return float(np.sum((a_vals / b_vals) - np.log(a_vals / b_vals) - 1))


def nmf_mu_is(
    x: np_typing.NDArray[np.floating],
    k: int,
    n: int = 100,
    tol: float = 1e-3,
    seed: int | None = None,
    r: int = 20,
    eval_every: int = 10,
    init: typing.Literal["random", "nndsvd", "svd_impute"]
    | typing.Callable[
        [np_typing.NDArray[np.floating], int], tuple[np_typing.NDArray[np.floating], np_typing.NDArray[np.floating]]
    ] = "random",
    callback: typing.Callable[[int, float], None] | None = None,
) -> tuple[np_typing.NDArray[np.floating], np_typing.NDArray[np.floating], np_typing.NDArray[np.floating], float]:
    """
    NMF via Multiplicative Updates (MU) minimizing the Itakura-Saito (IS) divergence.

    Optimizes the Itakura-Saito objective:

    $$D_{\\text{IS}}(X \\parallel W H) = \\sum_{i,j} M_{ij} \\left( \\frac{X_{ij}}{(W H)_{ij}} - \\log \\frac{X_{ij}}{(W H)_{ij}} - 1 \\right)$$

    Multiplicative update steps:

    $$H \\leftarrow H \\odot \\sqrt{ \\frac{W^T \\left( \\frac{M \\odot X}{(W H)^2} \\right)}{W^T \\left( \\frac{M}{W H} \\right)} }$$

    $$W \\leftarrow W \\odot \\sqrt{ \\frac{\\left( \\frac{M \\odot X}{(W H)^2} \\right) H^T}{\\left( \\frac{M}{W H} \\right) H^T} }$$

    Args:
        x (np.ndarray): Target matrix of shape (m, n).
        k (int): Number of latent components.
        n (int): Maximum number of iterations. Defaults to 100.
        tol (float): Convergence tolerance. Defaults to 1e-3.
        seed (int, optional): Random seed. Defaults to None.
        r (int): Number of restarts for initial factor selection. Defaults to 20.
        eval_every (int): Frequency of checking early-stopping criteria. Defaults to 10.
        init (str or callable): Initialization method ('random', 'nndsvd', 'svd_impute')
                                or a custom callable. Defaults to 'random'.
        callback (callable, optional): Function called at each iteration with (iteration, cost). Defaults to None.

    Returns:
        tuple: (rec, w, h, cost)
            - rec (np.ndarray): Reconstructed matrix of shape (m, n).
            - w (np.ndarray): Left factor matrix of shape (m, k).
            - h (np.ndarray): Right factor matrix of shape (k, n).
            - cost (float): Final IS cost.
    """
    eps = np.finfo(float).eps

    w, h = initialize_nmf(x, k, method=init, seed=seed)

    # Create a Mask
    m_mask = np.isnan(x)
    x_copy = x.copy()
    x_copy[m_mask] = 0
    m_not = ~m_mask

    cost_mask = m_not & (x_copy > 0.0)

    if seed is None and init == "random":
        rec = w @ h
        cost = cost_is(x_copy, rec, m_not)

        for _ in range(r):
            wt, ht = initialize_nmf(x, k, method="random")
            rec_temp = wt @ ht
            cost_temp = cost_is(x_copy, rec_temp, m_not)

            if cost_temp < cost:
                w = wt
                h = ht
                cost = cost_temp

    x_zero: np_typing.NDArray[np.bool_] = x_copy == 0
    rec: np_typing.NDArray[np.floating] = w @ h
    cost = cost_is(x_copy, rec, cost_mask)

    for idx in range(n):
        # 1. Update h
        rec = w @ h
        x_imp: np_typing.NDArray[np.floating] = np.where(x_zero, rec, x_copy)
        ratio: np_typing.NDArray[np.floating] = x_imp / rec
        rec_sum_rows: np_typing.NDArray[np.floating] = rec.sum(axis=1)[:, None]
        w_scaled: np_typing.NDArray[np.floating] = w / rec_sum_rows
        numerator: np_typing.NDArray[np.floating] = w_scaled.T @ ratio
        denominator: np_typing.NDArray[np.floating] = w_scaled.sum(axis=0)[:, None]
        h: np_typing.NDArray[np.floating] = h * np.sqrt(numerator / denominator)
        h = np.maximum(h, eps)

        # 2. Update w
        rec = w @ h
        x_imp = np.where(x_zero, rec, x_copy)
        ratio = x_imp / rec
        rec_sum_cols: np_typing.NDArray[np.floating] = rec.sum(axis=0)
        h_scaled: np_typing.NDArray[np.floating] = h / rec_sum_cols
        numerator = ratio @ h_scaled.T
        denominator = h_scaled.sum(axis=1)
        w: np_typing.NDArray[np.floating] = w * np.sqrt(numerator / denominator)
        w = np.maximum(w, eps)

        # 3. Evaluation and callback
        rec = w @ h
        cost = cost_is(x_copy, rec, cost_mask)
        if callback:
            callback(idx, cost)

        if idx % eval_every == 0:
            if cost <= tol:
                break
    else:
        rec = w @ h
        cost = cost_is(x_copy, rec, cost_mask)

    x_copy[~m_not] = np.nan
    return rec, w, h, float(cost)


def nmf_als(
    x: np_typing.NDArray[np.floating],
    k: int,
    n: int = 1000,
    tol: float = 1e-3,
    seed: int | None = None,
    eval_every: int = 10,
    init: typing.Literal["random", "nndsvd", "svd_impute"]
    | typing.Callable[
        [np_typing.NDArray[np.floating], int], tuple[np_typing.NDArray[np.floating], np_typing.NDArray[np.floating]]
    ] = "random",
    callback: typing.Callable[[int, float], None] | None = None,
) -> tuple[np_typing.NDArray[np.floating], np_typing.NDArray[np.floating], np_typing.NDArray[np.floating], float]:
    """
    Alternating Least Squares (ALS) NMF with missing values imputation.

    Minimizes the Frobenius norm. During each iteration, missing entries (and observed zeros)
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
    eps = np.finfo(float).eps
    w, h = initialize_nmf(x, k, method=init, seed=seed)

    m_mask = np.isnan(x)
    x_copy = x.copy()
    x_copy[m_mask] = 0
    m_not = ~m_mask
    m_float = m_not.astype(float)
    x_zero = x_copy == 0

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
    init: typing.Literal["random", "nndsvd", "svd_impute"]
    | typing.Callable[
        [np_typing.NDArray[np.floating], int], tuple[np_typing.NDArray[np.floating], np_typing.NDArray[np.floating]]
    ] = "random",
    callback: typing.Callable[[int, float], None] | None = None,
) -> tuple[np_typing.NDArray[np.floating], np_typing.NDArray[np.floating], np_typing.NDArray[np.floating], float]:
    """
    Hierarchical Alternating Least Squares (HALS) NMF with missing values imputation.

    Minimizes the Frobenius norm. Computes dynamic imputation as in ALS, then updates
    factors column-by-column (for w) and row-by-row (for h):

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
    eps = np.finfo(float).eps
    w, h = initialize_nmf(x, k, method=init, seed=seed)

    m_mask = np.isnan(x)
    x_copy = x.copy()
    x_copy[m_mask] = 0
    m_not = ~m_mask
    m_float = m_not.astype(float)
    x_zero = x_copy == 0

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

def nmf_hals_kl(
    x: np_typing.NDArray[np.floating],
    k: int,
    n: int = 100,
    tol: float = 1e-3,
    seed: int | None = None,
    eval_every: int = 10,
    init: typing.Literal["random", "nndsvd", "svd_impute"]
    | typing.Callable[
        [np_typing.NDArray[np.floating], int], tuple[np_typing.NDArray[np.floating], np_typing.NDArray[np.floating]]
    ] = "random",
    callback: typing.Callable[[int, float], None] | None = None,
) -> tuple[np_typing.NDArray[np.floating], np_typing.NDArray[np.floating], np_typing.NDArray[np.floating], float]:
    """
    Hierarchical Alternating Least Squares (HALS) NMF minimizing the KL divergence.

    For KL divergence, this implementation uses Multiplicative Updates (MU) for the
    entire factor matrix, which is the standard monotonic approach for missing values.

    Args:
        x (np.ndarray): Target matrix of shape (m, n).
        k (int): Number of latent components.
        n (int): Maximum number of iterations. Defaults to 100.
        tol (float): Convergence tolerance. Defaults to 1e-3.
        seed (int, optional): Random seed. Defaults to None.
        eval_every (int): Frequency of checking early-stopping criteria. Defaults to 10.
        init (str or callable): Initialization method ('random', 'nndsvd', 'svd_impute')
                                or a custom callable. Defaults to 'random'.
        callback (callable, optional): Function called at each iteration with (iteration, cost). Defaults to None.

    Returns:
        tuple: (rec, w, h, cost)
    """
    eps = np.finfo(float).eps
    w, h = initialize_nmf(x, k, method=init, seed=seed)

    m_mask = np.isnan(x)
    x_copy = x.copy()
    x_copy[m_mask] = 0
    m_not = ~m_mask
    m_float = m_not.astype(float)
    cost_mask = m_not & (x_copy > 0.0)

    rec = w @ h
    cost = cost_kl(x_copy, rec, cost_mask)

    for idx in range(n):
        # Update W
        rec = w @ h
        ratio: np_typing.NDArray[np.floating] = np.where(m_not, x_copy / np.maximum(rec, eps), 0.0)
        # Denominator: sum_j M_ij H_kj
        denom_w: np_typing.NDArray[np.floating] = m_float @ h.T
        w: np_typing.NDArray[np.floating] = w * ((ratio @ h.T) / (denom_w + eps))
        w = np.maximum(w, eps)

        # Update H
        rec = w @ h
        ratio = np.where(m_not, x_copy / np.maximum(rec, eps), 0.0)
        # Denominator: sum_i M_ij W_ik
        denom_h: np_typing.NDArray[np.floating] = w.T @ m_float
        h: np_typing.NDArray[np.floating] = h * ((w.T @ ratio) / (denom_h + eps))
        h = np.maximum(h, eps)

        # Evaluation and callback
        rec: np_typing.NDArray[np.floating] = w @ h
        cost = cost_kl(x_copy, rec, cost_mask)
        if callback:
            callback(idx, cost)

        if idx % eval_every == 0:
            if cost <= tol:
                break
    else:
        rec = w @ h
        cost = cost_kl(x_copy, rec, cost_mask)

    x_copy[~m_not] = np.nan
    return rec, w, h, float(cost)


def nmf_hals_is(
    x: np_typing.NDArray[np.floating],
    k: int,
    n: int = 100,
    tol: float = 1e-3,
    seed: int | None = None,
    eval_every: int = 10,
    init: typing.Literal["random", "nndsvd", "svd_impute"]
    | typing.Callable[
        [np_typing.NDArray[np.floating], int], tuple[np_typing.NDArray[np.floating], np_typing.NDArray[np.floating]]
    ] = "random",
    callback: typing.Callable[[int, float], None] | None = None,
) -> tuple[np_typing.NDArray[np.floating], np_typing.NDArray[np.floating], np_typing.NDArray[np.floating], float]:
    """
    Hierarchical Alternating Least Squares (HALS) NMF minimizing the IS divergence.

    Implements coordinate-descent style updates for IS divergence.

    Args:
        x (np.ndarray): Target matrix of shape (m, n).
        k (int): Number of latent components.
        n (int): Maximum number of iterations. Defaults to 100.
        tol (float): Convergence tolerance. Defaults to 1e-3.
        seed (int, optional): Random seed. Defaults to None.
        eval_every (int): Frequency of checking early-stopping criteria. Defaults to 10.
        init (str or callable): Initialization method ('random', 'nndsvd', 'svd_impute')
                                or a custom callable. Defaults to 'random'.
        callback (callable, optional): Function called at each iteration with (iteration, cost). Defaults to None.

    Returns:
        tuple: (rec, w, h, cost)
    """
    eps = np.finfo(float).eps
    w, h = initialize_nmf(x, k, method=init, seed=seed)

    m_mask = np.isnan(x)
    x_copy = x.copy()
    x_copy[m_mask] = 0
    m_not = ~m_mask
    cost_mask = m_not & (x_copy > 0.0)

    rec = w @ h
    cost = cost_is(x_copy, rec, cost_mask)

    for idx in range(n):
        # Update W column by column
        for j in range(k):
            rec_sq = np.maximum(rec**2, eps)
            rec_inv = 1.0 / np.maximum(rec, eps)
            
            num = (np.where(m_not, x_copy / rec_sq, 0.0)) @ h[j, :]
            den = (np.where(m_not, rec_inv, 0.0)) @ h[j, :]
            
            w_old = w[:, j].copy()
            w[:, j] = np.maximum(eps, w[:, j] * np.sqrt(num / (den + eps)))
            rec += np.outer(w[:, j] - w_old, h[j, :])

        # Update H row by row
        for i in range(k):
            rec_sq = np.maximum(rec**2, eps)
            rec_inv = 1.0 / np.maximum(rec, eps)
            
            num = (w.T @ np.where(m_not, x_copy / rec_sq, 0.0))[i, :]
            den = (w.T @ np.where(m_not, rec_inv, 0.0))[i, :]
            
            h_old = h[i, :].copy()
            h[i, :] = np.maximum(eps, h[i, :] * np.sqrt(num / (den + eps)))
            rec += w[:, i] @ (h[i, :] - h_old)

        # Evaluation and callback
        rec = w @ h
        cost = cost_is(x_copy, rec, cost_mask)
        if callback:
            callback(idx, cost)

        if idx % eval_every == 0:
            if cost <= tol:
                break
    else:
        rec = w @ h
        cost = cost_is(x_copy, rec, cost_mask)

    x_copy[~m_not] = np.nan
    return rec, w, h, float(cost)

