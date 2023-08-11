import unittest

import numpy as np
import torch

import marquetry.functions as funcs
from marquetry import Variable
from marquetry.utils import array_close, gradient_check


class TestMax(unittest.TestCase):

    def test_forward1(self):
        x = Variable(np.random.rand(10))
        y = funcs.max(x)

        res = y.data
        expected = np.max(x.data)

        self.assertTrue(array_close(res, expected))

    def test_forward2(self):
        shape = (10, 20, 30)
        axis = 1

        x = Variable(np.random.randn(*shape))
        y = funcs.max(x, axis=axis)

        res = y.data
        expected = np.max(x.data, axis=axis)

        self.assertTrue(array_close(res, expected))

    def test_forward3(self):
        shape = (10, 20, 30)
        axis = (0, 1)

        x = Variable(np.random.randn(*shape))
        y = funcs.max(x, axis=axis)

        res = y.data
        expected = np.max(x.data, axis=axis)

        self.assertTrue(array_close(res, expected))

    def test_forward4(self):
        shape = (10, 20, 30)
        axis = (0, 1)

        x = Variable(np.random.randn(*shape))
        y = funcs.max(x, axis=axis, keepdims=True)

        res = y.data
        expected = np.max(x.data, axis=axis, keepdims=True)

        self.assertTrue(array_close(res, expected))

    def test_backward1(self):
        x_data = np.random.randn(10)
        f = lambda x: funcs.max(x)

        self.assertTrue(gradient_check(f, x_data))

    def test_backward2(self):
        x_data = np.random.randn(10, 10)
        f = lambda x: funcs.max(x, axis=1)

        self.assertTrue(gradient_check(f, x_data))

    def test_backward3(self):
        x_data = np.random.randn(10, 20, 30)
        f = lambda x: funcs.max(x, axis=(1, 2))

        self.assertTrue(gradient_check(f, x_data))

    def test_backward4(self):
        x_data = np.random.randn(10, 20, 20)
        f = lambda x: funcs.max(x, axis=None)

        self.assertTrue(gradient_check(f, x_data))

    def test_backward5(self):
        x_data = np.random.randn(10, 20, 20)
        f = lambda x: funcs.max(x, axis=None, keepdims=True)

        self.assertTrue(gradient_check(f, x_data))


class TestMin(unittest.TestCase):

    def test_forward1(self):
        x = Variable(np.random.rand(10))
        y = funcs.min(x)

        res = y.data
        expected = np.min(x.data)

        self.assertTrue(array_close(res, expected))

    def test_forward2(self):
        shape = (10, 20, 30)
        axis = 1

        x = Variable(np.random.randn(*shape))
        y = funcs.min(x, axis=axis)

        res = y.data
        expected = np.min(x.data, axis=axis)

        self.assertTrue(array_close(res, expected))

    def test_forward3(self):
        shape = (10, 20, 30)
        axis = (0, 1)

        x = Variable(np.random.randn(*shape))
        y = funcs.min(x, axis=axis)

        res = y.data
        expected = np.min(x.data, axis=axis)

        self.assertTrue(array_close(res, expected))

    def test_forward4(self):
        shape = (10, 20, 30)
        axis = (0, 1)

        x = Variable(np.random.randn(*shape))
        y = funcs.min(x, axis=axis, keepdims=True)

        res = y.data
        expected = np.min(x.data, axis=axis, keepdims=True)

        self.assertTrue(array_close(res, expected))

    def test_backward1(self):
        x_data = np.random.randn(10)
        f = lambda x: funcs.min(x)

        self.assertTrue(gradient_check(f, x_data))

    def test_backward2(self):
        x_data = np.random.randn(10, 10)
        f = lambda x: funcs.min(x, axis=1)

        self.assertTrue(gradient_check(f, x_data))

    def test_backward3(self):
        x_data = np.random.randn(10, 20, 30)
        f = lambda x: funcs.min(x, axis=(1, 2))

        self.assertTrue(gradient_check(f, x_data))

    def test_backward4(self):
        x_data = np.random.randn(10, 20, 20)
        f = lambda x: funcs.min(x, axis=None)

        self.assertTrue(gradient_check(f, x_data))

    def test_backward5(self):
        x_data = np.random.randn(10, 20, 20)
        f = lambda x: funcs.min(x, axis=None, keepdims=True)

        self.assertTrue(gradient_check(f, x_data))


class TestClip(unittest.TestCase):

    def test_forward1(self):
        x = np.array([[0, .3, .8], [1.2, .3, .9], [0, .2, 1.5]])
        y = funcs.clip(x, 1e-5, 1.)

        res = y.data
        expected = np.array([[1e-5, .3, .8], [1.0, .3, .9], [1e-5, .2, 1.0]])

        self.assertTrue(array_close(res, expected))

    def test_forward2(self):
        x = np.array([[0, .3, .8], [1.2, .3, .9], [0, .2, 1.5]])
        y = funcs.clip(x, x_min=1e-5, x_max=1.)
        ty = torch.clip(torch.tensor(x), min=torch.tensor(1e-5), max=torch.tensor(1.))

        self.assertTrue(array_close(y.data, ty.data.numpy()))
