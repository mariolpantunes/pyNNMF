# coding: utf-8
"""
Naive implementations of NMF cost functions from the main branch,
used strictly as reference baselines in unit tests.
"""

__author__ = "Mário Antunes"
__version__ = "0.2.0"
__email__ = "mario.antunes@ua.pt"
__status__ = "Development"

import numpy as np
import numpy.typing as np_typing


def cost_fb(
    a: np_typing.NDArray[np.floating], b: np_typing.NDArray[np.floating], m: np_typing.NDArray[np.bool_] | None = None
) -> float:
    if m is None:
        m_mask = np.isnan(a)
        a_copy = a.copy()
        a_copy[m_mask] = 0
        m_not = ~m_mask
        cost = np.linalg.norm((m_not * a_copy) - (m_not * b), ord="fro")
    else:
        cost = np.linalg.norm((m * a) - (m * b), ord="fro")
    return float(cost)


def cost_kl(
    a: np_typing.NDArray[np.floating], b: np_typing.NDArray[np.floating], m: np_typing.NDArray[np.bool_] | None = None
) -> float:
    if m is None:
        m = ~np.isnan(a)
    mask = m & (a > 0.0)
    term_pos = a[mask] * np.log(a[mask] / b[mask]) - a[mask]
    term_b = b[m]
    return float(np.sum(term_pos) + np.sum(term_b))


def cost_is(
    a: np_typing.NDArray[np.floating], b: np_typing.NDArray[np.floating], m: np_typing.NDArray[np.bool_] | None = None
) -> float:
    if m is None:
        m = ~np.isnan(a)
    mask = m & (a > 0.0)
    return float(np.sum((a[mask] / b[mask]) - np.log(a[mask] / b[mask]) - 1))
