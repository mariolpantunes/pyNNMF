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


def cost_kl(
    a: np_typing.NDArray[np.floating], b: np_typing.NDArray[np.floating], m: np_typing.NDArray[np.bool_] | None = None
) -> float:
    if m is None:
        if np.any(np.isnan(a)):
            m_mask = np.isnan(a)
            m_not = ~m_mask
        else:
            m_not = a > 0.0
    else:
        m_not = m

    mask = m_not & (a > 0.0)
    return float(np.sum(a[mask] * np.log(a[mask] / b[mask]) - a[mask] + b[mask]))


def cost_is(
    a: np_typing.NDArray[np.floating], b: np_typing.NDArray[np.floating], m: np_typing.NDArray[np.bool_] | None = None
) -> float:
    if m is None:
        if np.any(np.isnan(a)):
            m_mask = np.isnan(a)
            a_copy = a.copy()
            a_copy[m_mask] = 0
            m_not = ~m_mask
            cost = np.sum((a_copy[m_not] / b[m_not]) - np.log(a_copy[m_not] / b[m_not]) - 1)
        else:
            m_not = a > 0.0
            cost = np.sum((a[m_not] / b[m_not]) - np.log(a[m_not] / b[m_not]) - 1)
    else:
        cost = np.sum((a[m] / b[m]) - np.log(a[m] / b[m]) - 1)
    return float(cost)
