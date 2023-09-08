import unittest

import numpy as np

import marquetry.functions as funcs
from marquetry.utils import array_equal, gradient_check


class TestSin(unittest.TestCase):

    def test_forward1(self):
        x = np.array([1, 2, 3])
        y = funcs.sin(x)

        res = y.data
        expected = np.sin(x)

        self.assertTrue(array_equal(res, expected))

    def test_forward2(self):
        x = np.array(4)
        y = funcs.sin(x)

        res = y.data
        expected = np.sin(x)

        self.assertTrue(array_equal(res, expected))

    def test_backward1(self):
        x = np.random.randn(3, 4)

        self.assertTrue(gradient_check(funcs.sin, x))

    def test_backward2(self):
        x = np.random.randn(1)

        self.assertTrue(gradient_check(funcs.sin, x))


class TestCos(unittest.TestCase):

    def test_forward1(self):
        x = np.array([1, 2, 3])
        y = funcs.cos(x)

        res = y.data
        expected = np.cos(x)

        self.assertTrue(array_equal(res, expected))

    def test_forward2(self):
        x = np.array(4)
        y = funcs.cos(x)

        res = y.data
        expected = np.cos(x)

        self.assertTrue(array_equal(res, expected))

    def test_backward1(self):
        x = np.random.randn(3, 4)

        self.assertTrue(gradient_check(funcs.cos, x))

    def test_backward2(self):
        x = np.random.randn(1)

        self.assertTrue(gradient_check(funcs.cos, x))


class TestTan(unittest.TestCase):

    def test_forward1(self):
        x = np.array([1, 2, 3])
        y = funcs.tan(x)

        res = y.data
        expected = np.tan(x)

        self.assertTrue(array_equal(res, expected))

    def test_forward2(self):
        x = np.array(4)
        y = funcs.tan(x)

        res = y.data
        expected = np.tan(x)

        self.assertTrue(array_equal(res, expected))

    def test_backward1(self):
        x = np.random.rand(10, 30)

        self.assertTrue(gradient_check(funcs.tan, x))

    def test_backward2(self):
        x = np.random.randn(1)

        self.assertTrue(gradient_check(funcs.tan, x))


class TestTanh(unittest.TestCase):

    def test_forward1(self):
        x = np.array([1, 2, 3])
        y = funcs.tanh(x)

        res = y.data
        expected = np.tanh(x)

        self.assertTrue(array_equal(res, expected))

    def test_forward2(self):
        x = np.array(4)
        y = funcs.tanh(x)

        res = y.data
        expected = np.tanh(x)

        self.assertTrue(array_equal(res, expected))

    def test_backward1(self):
        x = np.random.randn(10, 20)
        self.assertTrue(gradient_check(funcs.tanh, x))

    def test_backward2(self):
        x = np.random.randn(1)

        self.assertTrue(gradient_check(funcs.tanh, x))


class TestExp(unittest.TestCase):

    def test_forward1(self):
        x = np.array([1, 2, 3])
        y = funcs.exp(x)

        res = y.data
        expected = np.exp(x)

        self.assertTrue(array_equal(res, expected))

    def test_forward2(self):
        x = np.array(4)
        y = funcs.exp(x)

        res = y.data
        expected = np.exp(x)

        self.assertTrue(array_equal(res, expected))

    def test_backward1(self):
        x = np.random.randn(3, 4)

        self.assertTrue(gradient_check(funcs.exp, x))

    def test_backward2(self):
        x = np.random.randn(1)

        self.assertTrue(gradient_check(funcs.exp, x))


class TestLog(unittest.TestCase):
    def test_forward1(self):
        x = np.array([1, 2, 3])
        y = funcs.log(x)

        res = y.data
        expected = np.log(x)

        self.assertTrue(array_equal(res, expected))

    def test_forward2(self):
        x = np.array(4)
        y = funcs.log(x)

        res = y.data
        expected = np.log(x)

        self.assertTrue(array_equal(res, expected))

    def test_backward1(self):
        x = np.abs(np.random.randn(3, 4))
        x = np.clip(x, 1e-8, 0.999)

        self.assertTrue(gradient_check(funcs.log, x))

    def test_backward2(self):
        x = np.abs(np.random.rand(1))
        x = np.clip(x, 1e-8, 0.999)

        self.assertTrue(gradient_check(funcs.log, x))


class TestAbsolute(unittest.TestCase):
    def test_forward1(self):
        x = np.array([[1, -2, 0], [-1, 1, 2]])
        expected = np.array([[1, 2, 0], [1, 1, 2]])

        y = funcs.absolute(x)

        self.assertTrue(array_equal(y.data, expected))

    def test_forward2(self):
        x = np.array(-1)
        expected = np.array(1)

        y = funcs.absolute(x)

        self.assertTrue(array_equal(y.data, expected))

    def test_backward1(self):
        x = np.random.randn(3, 4)

        self.assertTrue(gradient_check(funcs.absolute, x))

    def test_backward2(self):
        x = np.random.randn(1)

        self.assertTrue(gradient_check(funcs.absolute, x))


class TestAverage(unittest.TestCase):

    def test_forward1(self):
        x = np.array([[2, 4, 6], [1, 2, 3]])
        expected = np.array(3.)

        y = funcs.average(x)

        self.assertTrue(array_equal(y.data, expected))

    def test_forward2(self):
        x = np.array([[2, 4, 6], [1, 2, 3]])
        expected = np.array([[4.], [2.]])

        axis = 1
        keepdims = True

        y = funcs.average(x, axis=axis, keepdims=keepdims)

        self.assertTrue(array_equal(y.data, expected))

    def test_forward3(self):
        x = np.array(8)
        expected = np.array(8.)

        y = funcs.average(x)

        self.assertTrue(array_equal(y.data, expected))

    def test_backward1(self):
        x = np.random.randn(3, 4)

        self.assertTrue(gradient_check(funcs.average, x))

    def test_backward2(self):
        x = np.random.randn(3, 4)

        f = lambda x: funcs.average(x, axis=1)

        self.assertTrue(gradient_check(f, x))

    def test_backward3(self):
        x = np.random.randn(3, 4)

        f = lambda x: funcs.average(x, axis=0, keepdims=True)

        self.assertTrue(gradient_check(f, x))


class TestReciprocal(unittest.TestCase):
    def test_forward1(self):
        x = np.array([[2, 4, 6], [1, 2, 3]])
        expected = np.array([[.5, .25, 1/6], [1, .5, 1/3]])

        y = funcs.reciprocal(x, dtype=np.float64)

        self.assertTrue(array_equal(y.data, expected))

    def test_forward2(self):
        x = np.array(8)
        expected = np.array(0.125)

        y = funcs.reciprocal(x)

        self.assertTrue(array_equal(y.data, expected))

    def test_backward1(self):
        x = np.random.randn(3, 4)

        f = lambda x: funcs.reciprocal(x, dtype=np.float64)

        self.assertTrue(gradient_check(f, x))
