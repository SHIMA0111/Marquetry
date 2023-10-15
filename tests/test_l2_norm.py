import unittest

import numpy as np

import marquetry
from marquetry import functions
from marquetry.utils import array_close, gradient_check


def get_params(batch_size, channels, height=None, weight=None, dtype: any = "f"):
    if height is not None:
        x = np.random.randn(batch_size, channels, height, weight).astype(dtype)
    else:
        x = np.random.randn(batch_size, channels).astype(dtype)

    return x


class TestFixedLayerNorm(unittest.TestCase):

    def test_type1(self):
        batch_size, channels = 8, 5
        x = get_params(batch_size, channels)
        with marquetry.test_mode():
            y = functions.l2_normalization(x)

        self.assertTrue(y.data.dtype == np.float32)

    def test_forward1(self):
        batch_size, channels = 2, 10
        x = get_params(batch_size, channels, dtype=np.float64)
        with marquetry.test_mode():
            y = functions.l2_normalization(x)

        expected_norm = np.linalg.norm(x, axis=1, keepdims=True)
        expected = x / expected_norm

        self.assertTrue(array_close(y.data, expected))

    def test_forward2(self):
        batch_size, channels, height, width = 20, 3, 10, 10
        x = get_params(batch_size, channels, height, width)

        with marquetry.test_mode():
            y = functions.l2_normalization(x, axis=(1, 2))

        expected_norm = np.linalg.norm(x, axis=(1, 2), keepdims=True)
        expected = x / expected_norm

        self.assertTrue(array_close(y.data, expected))

    def test_backward1(self):
        batch_size, channels = 2, 10
        x = get_params(batch_size, channels, dtype=np.float64)

        self.assertTrue(gradient_check(functions.l2_normalization, x))

    def test_backward2(self):
        batch_size, channels, height, width = 20, 3, 10, 10
        x = get_params(batch_size, channels, height, width, dtype=np.float64)

        f = lambda x: functions.l2_normalization(x, axis=(1, 2))

        self.assertTrue(gradient_check(f, x))
