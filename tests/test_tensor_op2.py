import unittest

import numpy as np

import marquetry.functions as funcs
from marquetry import Variable
from marquetry.utils import array_close, array_equal, gradient_check


class TestConcat(unittest.TestCase):

    def test_forward1(self):
        x0 = Variable(np.array([[1, 2], [3, 4], [5, 6]]))
        x1 = np.array([[4], [4], [4]])

        y = funcs.concat((x0, x1), axis=1)

        self.assertEqual(y.shape, (3, 3))
        self.assertTrue(array_equal(y[:, 2].data.reshape((-1, 1)), x1))

    def test_forward2(self):
        x0 = np.array([[1, 2, 3], [4, 5, 6]])
        x1 = np.array([[2, 4, 6]])

        y = funcs.concat(x0, x1)

        self.assertEqual(y.shape, (3, 3))
        self.assertTrue(array_equal(y[2].reshape(1, -1), x1))

    def test_backward1(self):
        x0 = np.random.randn(2, 3)
        x1 = np.random.randn(1, 3)

        f = lambda x: funcs.concat((x, x1), axis=0)

        self.assertTrue(gradient_check(f, x0))

    def test_backward2(self):
        x0 = np.random.randn(3, 2)
        x1 = np.random.randn(3, 2)

        f = lambda x: funcs.concat((x0, x), axis=1)

        self.assertTrue(gradient_check(f, x1))


class TestSplit(unittest.TestCase):

    def test_forward1(self):
        x = np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]])
        y0, y1 = funcs.split(x, 2, axis=1)

        self.assertTrue(array_equal(x[:, :2], y0.data))
        self.assertTrue(array_equal(x[:, 2:], y1.data))

    def test_forward2(self):
        x = np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7]])
        y0, y1, y2 = funcs.split(x, (1, 3), axis=1)

        self.assertTrue(array_equal(x[:, :1], y0.data))
        self.assertTrue(array_equal(x[:, 1:3], y1.data))
        self.assertTrue(array_equal(x[:, 3:], y2.data))

    def test_backward1(self):
        x = np.random.randn(2, 10)

        def func(x):
            y0, y1 = funcs.split(x, indices=5, axis=1)
            return y0 + y1

        self.assertTrue(gradient_check(func, x))

    def test_backward2(self):
        x = np.random.randn(2, 10)

        def func(x):
            y0, y1 = funcs.split(x, indices=1, axis=0)
            return y0 * y1

        self.assertTrue(gradient_check(func, x))

    def test_backward3(self):
        x = np.random.randn(3, 12)

        def func(x):
            y0, y1, y2 = funcs.split(x, indices=(4, 8), axis=1)
            return y0 - y1 - y2

        self.assertTrue(gradient_check(func, x))


class TestSqueeze(unittest.TestCase):

    def test_forward1(self):
        x = np.random.randn(3, 1, 4)
        y = funcs.squeeze(x, axis=1)

        self.assertEqual(y.shape, (3, 4))

    def test_forward2(self):
        x = np.random.randn(3, 2, 4)

        with self.assertRaises(ValueError):
            funcs.squeeze(x, axis=1)

    def test_backward1(self):
        x = np.random.randn(3, 1, 5)

        self.assertTrue(gradient_check(funcs.squeeze, x, axis=1))

    def test_shape_check(self):
        x = Variable(np.random.randn(3, 1, 5))
        y = funcs.squeeze(x, axis=1)
        y.backward()

        self.assertTrue(x.grad.shape, x.shape)


class TestUnSqueeze(unittest.TestCase):

    def test_forward1(self):
        x = np.random.randn(3, 4)
        y = funcs.unsqueeze(x, axis=1)

        self.assertEqual(y.shape, (3, 1, 4))

    def test_forward2(self):
        x = Variable(np.random.randn(3, 4))
        y = x.unsqueeze(axis=0)

        self.assertEqual(y.shape, (1, 3, 4))

    def test_backward1(self):
        x = np.random.randn(3, 4)

        self.assertTrue(gradient_check(funcs.unsqueeze, x, axis=1))

    def test_shape_check(self):
        x = Variable(np.random.randn(3, 4))
        y = x.unsqueeze(axis=2)
        y.backward()

        self.assertEqual(x.grad.shape, x.shape)
