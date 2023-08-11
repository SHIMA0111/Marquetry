import unittest

import numpy as np

import marquetry.functions as funcs
from marquetry import Variable
from marquetry.utils import gradient_check, array_close


class TestSum(unittest.TestCase):

    def test_datatype(self):
        x = Variable(np.random.rand(10))
        y = funcs.sum(x)

        self.assertFalse(np.isscalar(y))

    def test_forward1(self):
        x = Variable(np.array(2.0))
        y = funcs.sum(x)

        res = y.data
        expected = np.sum(x.data)

        self.assertTrue(array_close(res, expected))

    def test_forward2(self):
        x = Variable(np.random.randn(10, 20, 30))
        y = funcs.sum(x, axis=1)

        res = y.data
        expected = np.sum(x.data, axis=1)

        self.assertTrue(array_close(res, expected))

    def test_forward3(self):
        x = Variable(np.random.randn(10, 20, 30))
        y = funcs.sum(x, axis=1, keepdims=True)

        res = y.data
        expected = np.sum(x.data, axis=1, keepdims=True)

        self.assertTrue(array_close(res, expected))

    def test_backward1(self):
        x_data = np.random.randn(10)
        f = lambda x: funcs.sum(x)

        self.assertTrue(gradient_check(f, x_data))

    def test_backward2(self):
        x_data = np.random.randn(10, 10)
        f = lambda x: funcs.sum(x, axis=1)

        self.assertTrue(gradient_check(f, x_data))

    def test_backward3(self):
        x_data = np.random.rand(10, 20, 20)
        f = lambda x: funcs.sum(x, axis=2)

        self.assertTrue(gradient_check(f, x_data))

    def test_backward4(self):
        x_data = np.random.randn(10, 20, 20)
        f = lambda x: funcs.sum(x, axis=None)

        self.assertTrue(gradient_check(f, x_data))

    def test_backward5(self):
        x_data = np.random.randn(10, 10, 10)
        f = lambda x: funcs.sum(x, axis=None, keepdims=True)

        self.assertTrue(gradient_check(f, x_data))


class TestSumTo(unittest.TestCase):

    def test_forward1(self):
        x = Variable(np.random.randn(10))
        y = funcs.sum_to(x, (1,))

        res = y.data
        expected = np.sum(x.data)

        self.assertTrue(array_close(res, expected))

    def test_forward2(self):
        x = Variable(np.array([[1., 2., 3.], [4., 5., 6.]]))
        y = funcs.sum_to(x, (1, 3))

        res = y.data
        expected = np.sum(x.data, axis=0, keepdims=True)

        self.assertTrue(array_close(res, expected))

    def test_forward3(self):
        x = Variable(np.random.randn(10))
        y = funcs.sum_to(x, (10,))

        res = y.data
        expected = x.data

        self.assertTrue(array_close(res, expected))

    def test_backward1(self):
        x_data = np.random.rand(10)
        f = lambda x: funcs.sum_to(x, (1,))

        self.assertTrue(gradient_check(f, x_data))

    def test_backward2(self):
        x_data = np.random.randn(10, 10) * 10
        f = lambda x: funcs.sum_to(x, (10,))

        self.assertTrue(gradient_check(f, x_data))

    def test_backward3(self):
        x_data = np.random.rand(10, 20, 20) * 100
        f = lambda x: funcs.sum_to(x, (10,))

        self.assertTrue(gradient_check(f, x_data))

    def test_backward4(self):
        x_data = np.random.randn(10)
        f = lambda x: funcs.sum_to(x, (10,)) + 1

        self.assertTrue(gradient_check(f, x_data))


class TestBroadcastTo(unittest.TestCase):

    def test_forward1(self):
        x = Variable(np.random.randn(10, 1, 4))
        shape = (10, 10, 4)

        y = funcs.broadcast_to(x, shape)

        self.assertEqual(y.shape, shape)

    def test_forward2(self):
        x = Variable(np.random.rand(1))
        shape = (5, 4)

        y = funcs.broadcast_to(x, shape)

        self.assertEqual(y.shape, shape)

    def test_forward3(self):
        x = Variable(np.random.randn(5, 4))
        shape = (5, 4)

        y = funcs.broadcast_to(x, shape)

        self.assertEqual(y.shape, shape)
        self.assertTrue(array_close(y.data, x.data))

    def test_backward1(self):
        x_data = np.random.rand(10, 1)
        f = lambda x: funcs.broadcast_to(x, (10, 5))

        self.assertTrue(gradient_check(f, x_data))

    def test_backward2(self):
        x_data = np.random.randn(1)
        f = lambda x: funcs.broadcast_to(x, (5, 5))

        self.assertTrue(gradient_check(f, x_data))

    def test_backward3(self):
        x_data = np.random.rand(5, 5)
        f = lambda x: funcs.broadcast_to(x, (5, 5))

        self.assertTrue(gradient_check(f, x_data))
