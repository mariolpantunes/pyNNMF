# coding: utf-8
"""
Initialization methods for Non-Negative Matrix Factorization (NMF).
Provides strategies for initializing factor matrices w and h,
including handling of missing values.
"""

__author__ = "Mário Antunes"
__version__ = "0.2.0"
__email__ = "mario.antunes@ua.pt"
__status__ = "Development"

import typing

import numpy as np
import numpy.typing as np_typing


def init_random(
    x: np_typing.NDArray[np.floating],
    k: int,
    seed: int | None = None,
) -> tuple[np_typing.NDArray[np.floating], np_typing.NDArray[np.floating]]:
    """
    Random initialization of w and h.

    Args:
        x (NDArray): Target matrix of shape (m, n).
        k (int): Number of latent components.
        seed (int, optional): Random seed. Defaults to None.

    Returns:
        tuple[NDArray, NDArray]: Initialized w (m, k) and h (k, n).
    """
    if seed is not None:
        np.random.seed(seed)

    shape = x.shape
    m: int = shape[0]
    n: int = shape[1]
    eps = np.finfo(float).eps

    w: np_typing.NDArray[np.floating] = np.random.uniform(size=(m, k))
    w = np.maximum(w, eps)
    w = w / (k * w.max())

    h: np_typing.NDArray[np.floating] = np.random.uniform(size=(k, n))
    h = np.maximum(h, eps)
    h = h / (k * h.max())

    return w, h


def _impute_mean(x: np.typing.NDArray[np.floating]) -> np.typing.NDArray[np.floating]:
    """Internal helper to fill NaNs with the global mean of observed values."""
    x_filled = x.copy()
    mask = np.isnan(x)
    if not np.any(mask):
        return x_filled

    mean_val = np.nanmean(x)
    if np.isnan(mean_val):  # All NaNs
        mean_val = 0.0

    x_filled[mask] = mean_val
    return x_filled


def init_nndsvd(
    x: np_typing.NDArray[np.floating],
    k: int,
    seed: int | None = None,
) -> tuple[np_typing.NDArray[np.floating], np_typing.NDArray[np.floating]]:
    """
    Non-negative Double Singular Value Decomposition (NNDSVD) initialization.
    Deterministic initialization that typically leads to faster convergence.
    Handles missing values via mean imputation before SVD.

    Args:
        x (NDArray): Target matrix of shape (m, n).
        k (int): Number of latent components.
        seed (int, optional): Random seed (only used if SVD is unstable). Defaults to None.

    Returns:
        tuple[NDArray, NDArray]: Initialized w (m, k) and h (k, n).
    """
    if seed is not None:
        np.random.seed(seed)

    x_filled = _impute_mean(x)
    u, s, vt = np.linalg.svd(x_filled, full_matrices=False)

    eps = np.finfo(float).eps
    shape = x.shape
    m: int = shape[0]
    n: int = shape[1]

    w: np_typing.NDArray[np.floating] = np.zeros((m, k))
    h: np_typing.NDArray[np.floating] = np.zeros((k, n))

    # First component is always non-negative if x is non-negative
    w[:, 0] = u[:, 0] * np.sqrt(s[0])
    h[0, :] = vt[0, :] * np.sqrt(s[0])

    for i in range(1, k):
        u_vec = u[:, i]
        v_vec = vt[i, :]

        # Split positive and negative parts
        u_pos = np.maximum(u_vec, 0)
        u_neg = np.maximum(-u_vec, 0)
        v_pos = np.maximum(v_vec, 0)
        v_neg = np.maximum(-v_vec, 0)

        u_pos_norm = np.linalg.norm(u_pos)
        u_neg_norm = np.linalg.norm(u_neg)
        v_pos_norm = np.linalg.norm(v_pos)
        v_neg_norm = np.linalg.norm(v_neg)

        m_pos = u_pos_norm * v_pos_norm
        m_neg = u_neg_norm * v_neg_norm

        if m_pos > m_neg:
            w[:, i] = u_pos * np.sqrt(s[i])
            h[i, :] = v_pos * np.sqrt(s[i])
        else:
            w[:, i] = u_neg * np.sqrt(s[i])
            h[i, :] = v_neg * np.sqrt(s[i])

    return np.maximum(w, eps), np.maximum(h, eps)


def init_svd_impute(
    x: np.typing.NDArray[np.floating],
    k: int,
    max_iter: int = 10,
    tol: float = 1e-4,
    seed: int | None = None,
) -> tuple[np.typing.NDArray[np.floating], np.typing.NDArray[np.floating]]:
    """
    Iterative SVD Imputation initialization.
    Iteratively fills missing values using a low-rank SVD approximation.
    Provides a high-quality 'warm start' for matrix completion.

    Args:
        x (NDArray): Target matrix of shape (m, n).
        k (int): Number of latent components.
        max_iter (int): Maximum iterations for imputation. Defaults to 10.
        tol (float): Convergence tolerance for imputation. Defaults to 1e-4.
        seed (int, optional): Random seed. Defaults to None.

    Returns:
        tuple[NDArray, NDArray]: Initialized w (m, k) and h (k, n).
    """
    if seed is not None:
        np.random.seed(seed)

    mask = np.isnan(x)
    if not np.any(mask):
        return init_nndsvd(x, k)

    x_filled = _impute_mean(x)

    for _ in range(max_iter):
        u, s, vt = np.linalg.svd(x_filled, full_matrices=False)

        # Low rank approximation
        s_k = np.diag(s[:k])
        x_rec = u[:, :k] @ s_k @ vt[:k, :]

        diff = np.linalg.norm(x_filled[mask] - x_rec[mask])
        x_filled[mask] = x_rec[mask]

        if diff < tol:
            break

    # Final decomposition of the imputed matrix to get w and h
    u, s, vt = np.linalg.svd(x_filled, full_matrices=False)
    w = u[:, :k] * np.sqrt(s[:k])
    h = np.sqrt(s[:k])[:, None] * vt[:k, :]

    # Ensure non-negativity
    eps = np.finfo(float).eps
    return np.maximum(w, eps), np.maximum(h, eps)


def initialize_nmf(
    x: np.typing.NDArray[np.floating],
    k: int,
    method: typing.Literal["random", "nndsvd", "svd_impute"]
    | typing.Callable[
        [np.typing.NDArray[np.floating], int], tuple[np.typing.NDArray[np.floating], np.typing.NDArray[np.floating]]
    ],
    seed: int | None = None,
    **kwargs: typing.Any,
) -> tuple[np.typing.NDArray[np.floating], np.typing.NDArray[np.floating]]:
    """
    Dispatcher for NMF initialization methods.

    Args:
        x (NDArray): Target matrix.
        k (int): Number of components.
        method (str or callable): Initialization method ('random', 'nndsvd', 'svd_impute')
                                 or a custom callable that takes (x, k) and returns (w, h).
        seed (int, optional): Random seed.
        **kwargs: Additional arguments for specific methods (e.g., max_iter for svd_impute).

    Returns:
        tuple[NDArray, NDArray]: Initialized w and h.
    """
    if callable(method):
        return method(x, k)

    # Cast method to Any to avoid "unreachable code" warning for the final else block
    m_any = typing.cast(typing.Any, method)
    if m_any == "random":
        return init_random(x, k, seed=seed)
    elif m_any == "nndsvd":
        return init_nndsvd(x, k, seed=seed)
    elif m_any == "svd_impute":
        max_iter = kwargs.get("max_iter", 10)
        tol = kwargs.get("tol", 1e-4)
        return init_svd_impute(x, k, max_iter=int(max_iter), tol=float(tol), seed=seed)
    else:
        raise ValueError(f"Unsupported initialization method: {method}")
