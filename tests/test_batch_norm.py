import unittest

import torch
import numpy as np

import marquetry
import marquetry.functions as funcs
from marquetry.utils import gradient_check, array_equal, array_close


def get_params(batch_size, channels, height=None, weight=None, dtype: any = "f"):
    if height is not None:
        x = np.random.randn(batch_size, channels, height, weight).astype(dtype)
    else:
        x = np.random.randn(batch_size, channels).astype(dtype)

    gamma = np.random.randn(channels).astype(dtype)
    beta = np.random.randn(channels).astype(dtype)
    mean = np.random.randn(channels).astype(dtype)
    var = np.abs(np.random.randn(channels).astype(dtype))

    return x, gamma, beta, mean, var


class TestFixedBatchNorm(unittest.TestCase):

    def test_type1(self):
        batch_size, channels = 8, 3
        x, gamma, beta, mean, var = get_params(batch_size, channels)
        with marquetry.test_mode():
            y = funcs.batch_norm(x, gamma, beta, mean, var, eps=2e-5)

        self.assertTrue(y.data.dtype == np.float32)

    def test_forward1(self):
        batch_size, channels = 8, 1
        x, gamma, beta, mean, var = get_params(batch_size, channels)

        ty = torch.batch_norm(torch.tensor(x), torch.tensor(gamma), torch.tensor(beta),
                              running_mean=torch.tensor(mean), running_var=torch.tensor(var),
                              training=False, momentum=0.9, eps=1e-15, cudnn_enabled=False)

        with marquetry.test_mode():
            y = funcs.batch_norm(x, gamma, beta, mean, var)
        self.assertTrue(array_close(y.data, ty.detach().numpy()))

    def test_forward2(self):
        batch_size, channels = 2, 10
        x, gamma, beta, mean, var = get_params(batch_size, channels)
        ty = torch.batch_norm(torch.tensor(x), torch.tensor(gamma), torch.tensor(beta),
                              running_mean=torch.tensor(mean), running_var=torch.tensor(var),
                              training=False, momentum=0.9, eps=1e-15, cudnn_enabled=False)
        with marquetry.test_mode():
            y = funcs.batch_norm(x, gamma, beta, mean, var)

        self.assertTrue(array_close(y.data, ty.detach().numpy()))

    def test_forward3(self):
        batch_size, channels = 20, 10
        x, gamma, beta, mean, var = get_params(batch_size, channels)
        ty = torch.batch_norm(torch.tensor(x), torch.tensor(gamma), torch.tensor(beta),
                              running_mean=torch.tensor(mean), running_var=torch.tensor(var),
                              training=False, momentum=0.9, eps=1e-15, cudnn_enabled=False)
        with marquetry.test_mode():
            y = funcs.batch_norm(x, gamma, beta, mean, var)

        self.assertTrue(array_close(y.data, ty.detach().numpy()))


class TestBatchNorm(unittest.TestCase):

    def test_type1(self):
        batch_size, channels = 8, 3
        x, gamma, beta, mean, var = get_params(batch_size, channels)
        y = funcs.batch_norm(x, gamma, beta, mean, var, eps=2e-5)

        self.assertTrue(y.data.dtype == np.float32)

    def test_forward1(self):
        batch_size, channels = 8, 1
        x, gamma, beta, mean, var = get_params(batch_size, channels)
        ty = torch.batch_norm(torch.tensor(x), torch.tensor(gamma), torch.tensor(beta),
                              running_mean=torch.tensor(mean), running_var=torch.tensor(var),
                              training=True, momentum=0.9, eps=1e-15, cudnn_enabled=False)
        y = funcs.batch_norm(x, gamma, beta, mean, var)

        self.assertTrue(array_close(y.data, ty.detach().numpy()))

    def test_forward2(self):
        batch_size, channels = 2, 10
        x, gamma, beta, mean, var = get_params(batch_size, channels)
        ty = torch.batch_norm(torch.tensor(x), torch.tensor(gamma), torch.tensor(beta),
                              running_mean=torch.tensor(mean), running_var=torch.tensor(var),
                              training=True, momentum=0.9, eps=1e-15, cudnn_enabled=False)
        y = funcs.batch_norm(x, gamma, beta, mean, var)

        self.assertTrue(array_close(y.data, ty.detach().numpy()))

    def test_forward3(self):
        batch_size, channels = 20, 10
        x, gamma, beta, mean, var = get_params(batch_size, channels)
        ty = torch.batch_norm(torch.tensor(x), torch.tensor(gamma), torch.tensor(beta),
                              running_mean=torch.tensor(mean), running_var=torch.tensor(var),
                              training=True, momentum=0.9, eps=1e-15, cudnn_enabled=False)
        y = funcs.batch_norm(x, gamma, beta, mean, var)

        self.assertTrue(array_close(y.data, ty.detach().numpy()))

    def test_forward4(self):
        batch_size, channels = 20, 10
        tl = torch.nn.BatchNorm1d(10, 1e-15, momentum=0.1, affine=False)
        l = marquetry.layers.BatchNorm()

        for _ in range(10):
            x = np.random.randn(batch_size, channels).astype("f")
            ty = tl(torch.tensor(x))
            y = l(x)
            self.assertTrue(array_close(y.data, ty.data))

        self.assertTrue(array_close(tl.running_mean.numpy(), l.avg_mean.data))
        self.assertTrue(array_close(tl.running_var.numpy(), l.avg_var.data))

    def test_backward1(self):
        batch_size, channels = 8, 3
        x, gamma, beta, mean, var = get_params(batch_size, channels, dtype=np.float64)
        f = lambda x: funcs.batch_norm(x, gamma, beta, mean, var)

        self.assertTrue(gradient_check(f, x))

    def test_backward2(self):
        batch_size, channels = 8, 3
        x, gamma, beta, mean, var = get_params(batch_size, channels, dtype=np.float64)
        f = lambda gamma: funcs.batch_norm(x, gamma, beta, mean, var)

        self.assertTrue(gradient_check(f, gamma))

    def test_backward3(self):
        batch_size, channels = 8, 3
        x, gamma, beta, mean, var = get_params(batch_size, channels, dtype=np.float64)
        f = lambda beta: funcs.batch_norm(x, gamma, beta, mean, var)

        self.assertTrue(gradient_check(f, beta))


class TestBatchNormLayer(unittest.TestCase):

    def test_forward1(self):
        batch_size, channels = 8, 3
        x, _, _, _, _ = get_params(batch_size, channels)
        ty = torch.nn.BatchNorm1d(3, eps=1e-15, affine=False)(torch.tensor(x))
        y = marquetry.layers.BatchNorm()(x)

        self.assertTrue(array_close(y.data, ty.data))

    def test_forward2(self):
        batch_size, channels = 8, 3
        tl = torch.nn.BatchNorm1d(3, eps=1e-15, affine=False)
        l = marquetry.layers.BatchNorm()

        for _ in range(10):
            x, _, _, _, _ = get_params(batch_size, channels)
            tl(torch.tensor(x))
            l(x)

        self.assertTrue(array_close(l.avg_mean.data, tl.running_mean.numpy()))
        self.assertTrue(array_close(l.avg_var.data, tl.running_var.numpy()))
