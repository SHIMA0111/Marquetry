import unittest


import numpy as np
import torch

import marquetry.functions as funcs
from marquetry.utils import array_close, gradient_check


class TestMSE(unittest.TestCase):

    def test_forward1(self):
        x0 = np.array([0., 1., 2.])
        x1 = np.array([0., 1., 2.])
        y = funcs.mean_squared_error(x0, x1)

        res = y.data
        expected = ((x0 - x1) ** 2).sum() / x0.size

        self.assertTrue(array_close(res, expected))

    def test_backward1(self):
        x0 = np.random.randn(10)
        x1 = np.random.randn(10)

        f = lambda x: funcs.mean_squared_error(x, x1)

        self.assertTrue(gradient_check(f, x0))

    def test_backward2(self):
        x0 = np.random.randn(10)
        x1 = np.random.randn(10)

        f = lambda x: funcs.mean_squared_error(x0, x)

        self.assertTrue(gradient_check(f, x1))

    def test_backward3(self):
        x0 = np.random.rand(100)
        x1 = np.random.rand(100)

        f = lambda x: funcs.mean_squared_error(x, x1)

        self.assertTrue(gradient_check(f, x0))


class TestSoftmaxCrossEntropy(unittest.TestCase):

    def test_forward1(self):
        x = np.array([[-1, 0, 1, 2], [2, 0, 1, -1]], np.float32)
        t = np.array([3, 0]).astype(np.int32)
        y = funcs.softmax_cross_entropy(x, t)
        tl = torch.nn.CrossEntropyLoss()
        ty = tl(torch.tensor(x), torch.tensor(t, dtype=torch.long))

        self.assertTrue(array_close(y.data, ty.data.numpy()))

    def test_backward1(self):
        x = np.array([[-1, 0, 1, 2], [2, 0, 1, -1]], np.float32)
        t = np.array([3, 0]).astype(np.int32)

        f = lambda x: funcs.softmax_cross_entropy(x, t)

        self.assertTrue(gradient_check(f, x))

    def test_backward2(self):
        batch_size, data_dim = 10, 10
        x = np.random.randn(batch_size, data_dim)
        t = np.random.randint(0, data_dim, (batch_size,))

        f = lambda x: funcs.softmax_cross_entropy(x, t)

        self.assertTrue(gradient_check(f, x))

    def test_backward3(self):
        batch_size, data_dim = 100, 10
        x = np.random.randn(batch_size, data_dim)
        t = np.random.randint(0, data_dim, (batch_size,))

        f = lambda x: funcs.softmax_cross_entropy(x, t)

        self.assertTrue(gradient_check(f, x))


class TestSigmoidCrossEntropy(unittest.TestCase):

    def test_forward1(self):
        x = np.array([[3], [-1]], np.float32)
        t = np.array([1, 0]).astype(np.int32)
        y = funcs.sigmoid_cross_entropy(x, t)
        tl = torch.nn.BCELoss()
        ty = tl(input=torch.sigmoid(torch.tensor(x)), target=torch.tensor(t.reshape((-1, 1)), dtype=torch.float32))

        self.assertTrue(array_close(y.data, ty.data.numpy()))

    def test_backward1(self):
        x = np.array([[3], [-1]], np.float32)
        t = np.array([1, 0]).astype(np.int32)

        f = lambda x: funcs.sigmoid_cross_entropy(x, t)

        self.assertTrue(gradient_check(f, x))

    def test_backward2(self):
        batch_size, data_dim = 10, 1
        x = np.random.randn(batch_size, data_dim)
        t = np.random.randint(0, data_dim, (batch_size,))

        f = lambda x: funcs.sigmoid_cross_entropy(x, t)

        self.assertTrue(gradient_check(f, x))

    def test_backward3(self):
        batch_size, data_dim = 100, 1
        x = np.random.randn(batch_size, data_dim)
        t = np.random.randint(0, data_dim, (batch_size,))

        f = lambda x: funcs.sigmoid_cross_entropy(x, t)

        self.assertTrue(gradient_check(f, x))
