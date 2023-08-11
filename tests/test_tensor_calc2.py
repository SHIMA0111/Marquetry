import unittest

import torch
import numpy as np

import marquetry.functions as funcs
import marquetry.layers as layers
from marquetry import Variable
from marquetry.utils import gradient_check, array_close, array_equal


class TestMatmul(unittest.TestCase):

    def test_forward1(self):
        x0 = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        x1 = Variable(x0.data.T)
        y = funcs.matmul(x0, x1)

        res = y.data
        expected = np.array([[14, 32], [32, 77]])

        self.assertTrue(array_equal(res, expected))

    def test_backward1(self):
        x0 = np.random.randn(3, 2)
        x1 = np.random.randn(2, 3)

        f = lambda x: funcs.matmul(x, x1)

        self.assertTrue(gradient_check(f, x0))

    def test_backward2(self):
        x0 = np.random.randn(3, 2)
        x1 = np.random.randn(2, 3)

        f = lambda x: funcs.matmul(x0, x)

        self.assertTrue(gradient_check(f, x1))

    def test_backward3(self):
        x0_data = np.random.randn(10, 1)
        x1_data = np.random.randn(1, 5)

        f = lambda x: funcs.matmul(x, x1_data)

        self.assertTrue(gradient_check(f, x0_data))


class TestLinear(unittest.TestCase):

    def test_forward1(self):
        x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        w = Variable(x.data.T)
        b = None

        y = funcs.linear(x, w, b)

        res = y.data
        expected = np.array([[14, 32], [32, 77]])

        self.assertTrue(array_equal(res, expected))

    def test_forward2(self):
        x = np.array([[1, 2, 3], [4, 5, 6]]).astype("f")
        w = x.T
        b = None

        y = funcs.linear(x, w, b)
        ty = torch.nn.functional.linear(torch.tensor(x), torch.tensor(x), b)

        self.assertTrue(array_close(y.data, ty.data.numpy()))

    def test_forward3(self):
        tl = torch.nn.Linear(3, 2)
        x = np.array([[1, 2, 3], [4, 5, 6]]).astype("f")
        w = tl.weight.data.numpy().T
        b = tl.bias.data.numpy()

        l = layers.Linear(2)
        l.w.data = w
        l.b.data = b

        y = l(x)
        ty = tl(torch.tensor(x))

        self.assertTrue(array_close(y.data, ty.data.numpy()))

    def test_backward1(self):
        x = np.random.randn(3, 2)
        w = np.random.randn(2, 3)
        b = np.random.randn(3)

        f = lambda x: funcs.linear(x, w, b)

        self.assertTrue(gradient_check(f, x))

    def test_backward2(self):
        x = np.random.randn(3, 2)
        w = np.random.randn(2, 3)
        b = np.random.randn(3)

        f = lambda w: funcs.linear(x, w, b)

        self.assertTrue(gradient_check(f, w))

    def test_backward3(self):
        x = np.random.randn(3, 2)
        w = np.random.randn(2, 3)
        b = np.random.randn(3)

        f = lambda b: funcs.linear(x, w, b)

        self.assertTrue(gradient_check(f, b))

    def test_backward4(self):
        x = np.random.randn(100, 200)
        w = np.random.randn(200, 300)
        b = None

        f = lambda x: funcs.linear(x, w, b)

        self.assertTrue(gradient_check(f, x))
