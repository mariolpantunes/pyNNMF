# coding: utf-8
"""
Tests for NMF initialization methods.
"""

__author__ = "Mário Antunes"
__version__ = "0.2.0"
__email__ = "mario.antunes@ua.pt"
__status__ = "Development"

import typing
import unittest

import numpy as np
import numpy.typing as np_typing

import pynnmf.initialization as pynnmf_init


class TestInitMethods(unittest.TestCase):
    def test_random_init(self):
        x = np.random.rand(10, 8)
        k = 3
        w, h = pynnmf_init.init_random(x, k)
        self.assertEqual(w.shape, (10, 3))
        self.assertEqual(h.shape, (3, 8))
        self.assertTrue(np.all(w >= 0))
        self.assertTrue(np.all(h >= 0))

    def test_nndsvd_init(self):
        x = np.random.rand(10, 8)
        k = 3
        w, h = pynnmf_init.init_nndsvd(x, k)
        self.assertEqual(w.shape, (10, 3))
        self.assertEqual(h.shape, (3, 8))
        self.assertTrue(np.all(w >= 0))
        self.assertTrue(np.all(h >= 0))

    def test_svd_impute_init(self):
        x = np.random.rand(10, 8)
        x[0, 0] = np.nan
        k = 3
        w, h = pynnmf_init.init_svd_impute(x, k)
        self.assertEqual(w.shape, (10, 3))
        self.assertEqual(h.shape, (3, 8))
        self.assertTrue(np.all(w >= 0))
        self.assertTrue(np.all(h >= 0))
        self.assertFalse(np.any(np.isnan(w)))
        self.assertFalse(np.any(np.isnan(h)))

    def test_custom_init(self):
        def custom_init(
            x: np_typing.NDArray[np.floating], k: int
        ) -> tuple[np_typing.NDArray[np.floating], np_typing.NDArray[np.floating]]:
            return np.ones((x.shape[0], k)), np.ones((k, x.shape[1]))

        x = np.random.rand(10, 8)
        k = 3
        w, h = pynnmf_init.initialize_nmf(x, k, method=custom_init)
        self.assertTrue(np.all(w == 1))
        self.assertTrue(np.all(h == 1))

    def test_invalid_method(self):
        x = np.random.rand(10, 8)
        with self.assertRaises(ValueError):
            _ = pynnmf_init.initialize_nmf(x, 3, method=typing.cast(typing.Any, "unknown"))


if __name__ == "__main__":
    _ = unittest.main()
