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
        x = np.random.randn(3, 4)
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

        self.assertTrue(gradient_check(funcs.log, x))

    def test_backward2(self):
        x = np.abs(np.random.randn(1))

        self.assertTrue(gradient_check(funcs.log, x))
