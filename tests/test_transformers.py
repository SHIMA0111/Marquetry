import unittest

import numpy as np

from marquetry.transformers import AsType, Compose, Flatten, Normalize, ToFloat, ToInt


class TestAsType(unittest.TestCase):

    def test_default_is_float32(self):
        x = np.arange(4)

        y = AsType()(x)

        self.assertEqual(y.dtype, np.float32)

    def test_to_float_and_to_int(self):
        x = np.array([1.5, 2.5])

        self.assertEqual(ToFloat()(x).dtype, np.float32)
        self.assertTrue(np.issubdtype(ToInt()(np.array([1.0, 2.0])).dtype, np.integer))


class TestFlatten(unittest.TestCase):

    def test_flattens_multi_dimension(self):
        x = np.zeros((2, 3, 4))

        y = Flatten()(x)

        self.assertEqual(y.shape, (24,))


class TestNormalize(unittest.TestCase):

    def test_scalar_mean_std(self):
        x = np.array([2.0, 4.0, 6.0])

        y = Normalize(mean=2.0, std=2.0)(x)

        np.testing.assert_allclose(y, [0.0, 1.0, 2.0])

    def test_channel_wise_mean(self):
        x = np.ones((2, 3, 3), dtype=np.float64)

        y = Normalize(mean=[0.0, 1.0], std=1.0)(x)

        np.testing.assert_allclose(y[0], np.ones((3, 3)))
        np.testing.assert_allclose(y[1], np.zeros((3, 3)))


class TestCompose(unittest.TestCase):

    def test_applies_in_order(self):
        compose = Compose([lambda x: x + 1, lambda x: x * 10])

        self.assertEqual(compose(1), 20)

    def test_empty_compose_is_identity(self):
        compose = Compose([])

        self.assertEqual(compose(5), 5)
