"""Tests for the autodiff engine mechanics (Container / Function / configuration)."""
import unittest

import numpy as np

import marquetry
import marquetry.functions as funcs
from marquetry import Container
from marquetry.utils import array_close, array_equal


class TestGradAccumulation(unittest.TestCase):

    def test_same_container_used_twice(self):
        x = Container(np.array(3.0))

        y = x + x
        y.backward()

        self.assertEqual(float(x.grad.data), 2.0)

    def test_diamond_graph(self):
        # y = (x^2)^2 + (x^2)^2 = 2x^4 -> dy/dx = 8x^3
        x = Container(np.array(2.0))

        a = funcs.square(x)
        y = funcs.square(a) + funcs.square(a)
        y.backward()

        self.assertEqual(float(x.grad.data), 8.0 * 2.0 ** 3)


class TestNoBackpropMode(unittest.TestCase):

    def test_graph_is_not_created(self):
        x = Container(np.array([1.0]))

        with marquetry.no_backprop_mode():
            y = x + 1.0

        self.assertIsNone(y.creator)

    def test_backward_is_noop(self):
        x = Container(np.array([1.0]))

        with marquetry.no_backprop_mode():
            y = x + 1.0
        y.backward()

        self.assertIsNone(x.grad)


class TestRetainGrad(unittest.TestCase):

    def test_intermediate_grad_released_by_default(self):
        x = Container(np.array(1.0))

        t = x + 1.0
        y = t * 2.0
        y.backward()

        self.assertIsNone(t.grad)
        self.assertEqual(float(x.grad.data), 2.0)

    def test_intermediate_grad_kept_when_requested(self):
        x = Container(np.array(1.0))

        t = x + 1.0
        y = t * 2.0
        y.backward(retain_grad=True)

        self.assertIsNotNone(t.grad)
        self.assertEqual(float(t.grad.data), 2.0)


class TestUnchain(unittest.TestCase):

    def test_unchain_backward_cuts_graph(self):
        x = Container(np.array(1.0))

        t = x + 1.0
        y = t * 2.0
        y.unchain_backward()

        self.assertIsNone(y.creator)
        self.assertIsNone(t.creator)

        y.backward()
        self.assertIsNone(x.grad)


class TestConfiguration(unittest.TestCase):

    def test_test_mode_restores_train_flag(self):
        self.assertTrue(marquetry.config.train)

        with marquetry.test_mode():
            self.assertFalse(marquetry.config.train)

        self.assertTrue(marquetry.config.train)

    def test_using_config_restores_on_error(self):
        try:
            with marquetry.using_config("train", False):
                raise RuntimeError("boom")
        except RuntimeError:
            pass

        self.assertTrue(marquetry.config.train)


class TestContainerBasics(unittest.TestCase):

    def test_copy_is_independent(self):
        x = Container(np.array([1, 2, 3]))

        y = x.copy()
        y.data[0] = 99

        self.assertEqual(x.data[0], 1)

    def test_to_numpy_refuses_with_grad(self):
        x = Container(np.array([1.0, 2.0]))
        x.grad = Container(np.ones(2))

        with self.assertRaises(TypeError):
            x.to_numpy()

    def test_transpose_property(self):
        x = Container(np.random.randn(2, 3))

        self.assertEqual(x.T.shape, (3, 2))

    def test_len_and_size(self):
        x = Container(np.zeros((4, 5)))

        self.assertEqual(len(x), 4)
        self.assertEqual(x.size, 20)
        self.assertEqual(x.ndim, 2)

    def test_clear_grad(self):
        x = Container(np.array(2.0))
        y = x * 3.0
        y.backward()
        self.assertIsNotNone(x.grad)

        x.clear_grad()
        self.assertIsNone(x.grad)

    def test_container_method_chaining(self):
        x = Container(np.arange(6, dtype=np.float64))

        y = x.reshape(2, 3).sum(axis=1)
        y.backward()

        self.assertEqual(y.shape, (2,))
        self.assertTrue(array_equal(x.grad.data, np.ones(6)))

    def test_matmul_operator(self):
        a = np.random.randn(2, 3)
        b = np.random.randn(3, 4)

        y = Container(a) @ Container(b)

        self.assertTrue(array_close(y.data, a.dot(b)))


class TestGradientShapeCheck(unittest.TestCase):

    def test_mismatched_grad_shape_rejected(self):
        x = Container(np.zeros((2, 3)))

        with self.assertRaises(ValueError):
            x.grad = Container(np.zeros((3, 2)))

    def test_mismatched_grad_dtype_rejected(self):
        x = Container(np.zeros(3, dtype=np.float32))

        with self.assertRaises(TypeError):
            x.grad = Container(np.zeros(3, dtype=np.float64))
