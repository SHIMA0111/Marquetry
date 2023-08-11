import unittest

import numpy as np
import torch

import marquetry.functions as funcs
from marquetry.utils import gradient_check, array_close


class TestSigmoid(unittest.TestCase):

    def test_forward1(self):
        x = np.array([[0, 1, 2], [0, 2, 4]], np.float32)
        y = funcs.sigmoid(x)
        ty = torch.nn.functional.sigmoid(torch.tensor(x))

        self.assertTrue(array_close(y.data, ty.data.numpy()))

    def test_forward2(self):
        x = np.random.randn(10, 10).astype(np.float32)
        y = funcs.sigmoid(x)
        ty = torch.nn.functional.sigmoid(torch.tensor(x))

        self.assertTrue(array_close(y.data, ty.data.numpy()))

    def test_backward1(self):
        x_data = np.array([[0, 1, 2], [0, 2, 4]])

        self.assertTrue(gradient_check(funcs.sigmoid, x_data))

    def test_backward2(self):
        x_data = np.random.rand(10, 10)

        self.assertTrue(gradient_check(funcs.sigmoid, x_data))

    def test_backward3(self):
        x_data = np.random.randn(10, 10, 10)

        self.assertTrue(gradient_check(funcs.sigmoid, x_data))


class TestReLU(unittest.TestCase):

    def test_forward1(self):
        x = np.array([[-1, 0], [2, -3], [-2, 1]], np.float32)
        y = funcs.relu(x)

        res = y.data
        expected = np.array([[0, 0], [2, 0], [0, 1]], np.float32)

        self.assertTrue(array_close(res, expected))

    def test_forward2(self):
        x = np.array([[-1, 0], [2, -3], [-2, 1]], np.float32)
        y = funcs.relu(x)
        ty = torch.relu(torch.tensor(x))

        self.assertTrue(array_close(y.data, ty.data.numpy()))

    def test_backward1(self):
        x_data = np.array([[-1, 1, 2], [-1, 2, 4]])

        self.assertTrue(gradient_check(funcs.relu, x_data))

    def test_backward2(self):
        x_data = np.random.randn(10, 10) * 100

        self.assertTrue(gradient_check(funcs.relu, x_data))

    def test_backward3(self):
        x_data = np.random.rand(10, 10, 10) * 100

        self.assertTrue(gradient_check(funcs.relu, x_data))


class TestSoftmax(unittest.TestCase):
    def test_forward1(self):
        x = np.array([[0, 1, 2], [0, 2, 4]], np.float32)
        y = funcs.softmax(x)
        ty = torch.softmax(torch.tensor(x), dim=1)

        self.assertTrue(array_close(y.data, ty.data.numpy()))

    def test_forward2(self):
        x = np.random.rand(10, 10).astype("f")
        y = funcs.softmax(x)
        ty = torch.softmax(torch.tensor(x), dim=1)

        self.assertTrue(array_close(y.data, ty.data.numpy()))

    def test_forward3(self):
        x = np.random.randn(10, 10, 10).astype("f")
        y = funcs.softmax(x, axis=2)
        ty = torch.softmax(torch.tensor(x), dim=2)

        self.assertTrue(array_close(y.data, ty.data.numpy()))

    def test_backward1(self):
        x_data = np.array([[0, 1, 2], [0, 2, 4]])
        f = lambda x: funcs.softmax(x, axis=1)

        self.assertTrue(gradient_check(f, x_data))

    def test_backward2(self):
        x_data = np.random.randn(10, 10)
        f = lambda x: funcs.softmax(x, axis=1)

        self.assertTrue(gradient_check(f, x_data))

    def test_backward3(self):
        x_data = np.random.rand(10, 10, 10)
        f = lambda x: funcs.softmax(x, axis=2)

        self.assertTrue(gradient_check(f, x_data))


class TestLogSoftmax(unittest.TestCase):

    def test_forward1(self):
        x = np.array([[-1, 0, 1, 2], [2, 0, 1, -1]], dtype=np.float32)
        y = funcs.log_softmax(x)
        ty = torch.log_softmax(torch.tensor(x), dim=1, dtype=torch.float32)

        self.assertTrue(array_close(y.data, ty.data.numpy()))

    def test_backward1(self):
        x_data = np.array([[-1, 0, 1, 2], [2, 0, 1, -1]])
        f = lambda x: funcs.log_softmax(x)

        self.assertTrue(gradient_check(f, x_data))

    def test_backward2(self):
        x_data = np.random.randn(10, 10)
        f = lambda x: funcs.log_softmax(x)

        self.assertTrue(gradient_check(f, x_data))

    def test_backward3(self):
        x_data = np.random.rand(10, 10, 10)
        f = lambda x: funcs.log_softmax(x, axis=2)

        self.assertTrue(gradient_check(f, x_data))


class TestLeakyReLU(unittest.TestCase):

    def test_forward1(self):
        x = np.array([[-1, 0], [2, -3], [-2, 1]], np.float32)
        y = funcs.leaky_relu(x)

        res = y.data
        expected = np.array([[-0.2, 0], [2, -0.6], [-0.4, 1]], np.float32)

        self.assertTrue(array_close(res, expected))

    def test_forward2(self):
        x = np.array([[-1, 0], [2, -3], [-2, 1]], np.float32)
        y = funcs.leaky_relu(x)
        ty = torch.nn.functional.leaky_relu(torch.tensor(x), negative_slope=0.2)

        self.assertTrue(array_close(y.data, ty.data.numpy()))

    def test_backward1(self):
        x_data = np.array([[-1, 1, 2], [-1, 2, 4]])

        self.assertTrue(gradient_check(funcs.leaky_relu, x_data))

    def test_backward2(self):
        x_data = np.random.randn(10, 10)

        self.assertTrue(gradient_check(funcs.leaky_relu, x_data))

    def test_backward3(self):
        x_data = np.random.rand(10, 10, 10) * 100

        self.assertTrue(gradient_check(funcs.leaky_relu, x_data))

