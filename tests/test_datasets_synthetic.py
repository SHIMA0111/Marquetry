"""Tests for the synthetic datasets that require no network access."""
import unittest

import numpy as np

from marquetry.datasets import Spiral, TrigonometricCurve
from marquetry.transformers import Flatten


class TestSpiral(unittest.TestCase):

    def test_shapes(self):
        dataset = Spiral()

        self.assertEqual(len(dataset), 600)
        self.assertEqual(dataset.source_shape, (600, 2))
        self.assertEqual(dataset.target.shape, (600, 3))

    def test_targets_are_one_hot(self):
        dataset = Spiral()

        np.testing.assert_array_equal(dataset.target.sum(axis=1), np.ones(600))

    def test_custom_class_setup(self):
        dataset = Spiral(class_num=2, class_data_size=50)

        self.assertEqual(len(dataset), 100)
        self.assertEqual(dataset.target.shape, (100, 2))

    def test_reproducible_with_same_seed(self):
        first = Spiral(random_state=7)
        second = Spiral(random_state=7)

        np.testing.assert_array_equal(first.source, second.source)

    def test_getitem_applies_transform(self):
        dataset = Spiral(transform=lambda x: x * 0.0)

        source, target = dataset[0]

        np.testing.assert_array_equal(source, np.zeros(2))


class TestTrigonometricCurve(unittest.TestCase):

    def test_shapes(self):
        dataset = TrigonometricCurve()

        self.assertEqual(dataset.source.shape, (4999, 1))
        self.assertEqual(dataset.target.shape, (4999, 1))

    def test_target_is_next_step(self):
        dataset = TrigonometricCurve(train=False)

        # without noise (test mode = cos curve), target equals source shifted by one
        np.testing.assert_array_equal(dataset.source[1:], dataset.target[:-1])

    def test_getitem_returns_single_step(self):
        dataset = TrigonometricCurve()

        source, target = dataset[0]

        self.assertEqual(source.shape, (1,))
        self.assertEqual(target.shape, (1,))

    def test_transform_applied(self):
        dataset = TrigonometricCurve(transform=Flatten())

        source, _ = dataset[3]

        self.assertEqual(source.shape, (1,))
