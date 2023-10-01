import unittest

import numpy as np

import marquetry.container
from marquetry import Container
from marquetry.utils import gradient_check, array_equal


class TestAdd(unittest.TestCase):

    def test_forward1(self):
        x0 = np.array([1, 2, 3])
        x1 = Container(np.array([1, 2, 3]))
        y = x0 + x1

        res = y.data
        expected = np.array([2, 4, 6])

        self.assertTrue(array_equal(res, expected))

    def test_datatype(self):
        x = Container(np.array(2.0))
        y = x ** 2

        self.assertFalse(np.isscalar(y))

    def test_backward1(self):
        x = Container(np.random.randn(3, 3))
        y = np.random.randn(3, 3)
        f = lambda x: x + y

        self.assertTrue(gradient_check(f, x))

    def test_backward2(self):
        x = Container(np.random.randn(3, 3))
        y = np.random.randn(3, 1)
        f = lambda x: x + y

        self.assertTrue(gradient_check(f, x))

    def test_backward3(self):
        x = np.random.randn(3, 3)
        y = np.random.randn(3, 1)

        self.assertTrue(gradient_check(marquetry.functions.add, x, y))


class TestSub(unittest.TestCase):
    def test_forward1(self):
        x0 = np.array([1, 2, 3])
        x1 = Container(np.array([1, 2, 3]))
        y = x0 - x1

        res = y.data
        expected = np.array([0, 0, 0])

        self.assertTrue(array_equal(res, expected))

    def test_backward1(self):
        x = Container(np.random.randn(3, 3))
        y = np.random.randn(3, 3)
        f = lambda x: x + y

        self.assertTrue(gradient_check(f, x))

    def test_backward2(self):
        x = Container(np.random.randn(3, 3))
        y = np.random.randn(3, 1)
        f = lambda x: x + y

        self.assertTrue(gradient_check(f, x))

    def test_backward3(self):
        x = np.random.randn(3, 3)
        y = np.random.randn(3, 1)

        self.assertTrue(gradient_check(marquetry.functions.sub, x, y))


class TestMul(unittest.TestCase):

    def test_forward1(self):
        x0 = np.array([1, 2, 3])
        x1 = Container(np.array([1, 2, 3]))

        y = x0 * x1
        res = y.data
        expected = np.array([1, 4, 9])

        self.assertTrue(array_equal(res, expected))

    def test_backward1(self):
        x = np.random.randn(3, 3)
        y = np.random.randn(3, 3)
        f = lambda x: x * y

        self.assertTrue(gradient_check(f, x))

    def test_backward2(self):
        x = np.random.randn(3, 3)
        y = np.random.randn(3, 1)
        f = lambda x: x * y

        self.assertTrue(gradient_check(f, x))

    def test_backward3(self):
        x = np.random.randn(3, 1)
        y = np.random.randn(3, 3)
        f = lambda x: x * y

        self.assertTrue(gradient_check(f, x))


class TestDiv(unittest.TestCase):

    def test_forward1(self):
        x0 = np.array([1, 2, 3])
        x1 = Container(x0)
        y = x0 / x1

        res = y.data
        expected = np.array([1, 1, 1])

        self.assertTrue(array_equal(res, expected))

    def test_backward1(self):
        x = np.random.randn(3, 3)
        y = np.random.randn(3, 3)
        f = lambda x: x / y

        self.assertTrue(gradient_check(f, x))

    def test_backward2(self):
        x = np.random.randn(3, 3)
        y = np.random.randn(3, 1)
        f = lambda x: x / y

        self.assertTrue(gradient_check(f, x))

    def test_backward3(self):
        x = np.random.randn(3, 1)
        y = np.random.randn(3, 3)
        f = lambda x: x / y

        self.assertTrue(gradient_check(f, x))
