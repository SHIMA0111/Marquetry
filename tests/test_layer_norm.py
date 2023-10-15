import unittest

import torch
import numpy as np

import marquetry
from marquetry import functions
from marquetry.utils import gradient_check, array_close


def get_params(batch_size, channels, height=None, weight=None, dtype: any = "f"):
    if height is not None:
        x = np.random.randn(batch_size, channels, height, weight).astype(dtype)
        dim = channels * height * weight
    else:
        x = np.random.randn(batch_size, channels).astype(dtype)
        dim = channels

    gamma = np.random.randn(dim).astype(dtype)
    beta = np.random.randn(dim).astype(dtype)

    return x, gamma, beta


class TestFixedLayerNorm(unittest.TestCase):

    def test_type1(self):
        batch_size, channels = 8, 3
        x, gamma, beta = get_params(batch_size, channels)
        with marquetry.test_mode():
            y = functions.layer_normalization(x, gamma, beta)

        self.assertTrue(y.data.dtype == np.float32)

    def test_forward1(self):
        batch_size, channels = 2, 10
        x, gamma, beta = get_params(batch_size, channels, dtype=np.float64)
        ty = torch.layer_norm(torch.tensor(x), normalized_shape=(channels,),
                              weight=torch.tensor(gamma), bias=torch.tensor(beta), eps=1e-15, cudnn_enable=False)

        with marquetry.test_mode():
            y = functions.layer_normalization(x, gamma, beta)

        self.assertTrue(array_close(y.data, ty.detach().numpy()))

    def test_forward2(self):
        batch_size, channels, height, width = 20, 3, 10, 10
        x, gamma, beta = get_params(batch_size, channels, height, width)
        ty = torch.layer_norm(torch.tensor(x), normalized_shape=(channels, height, width,),
                              weight=torch.tensor(gamma.reshape(channels, height, width)),
                              bias=torch.tensor(beta.reshape(channels, width, height)), eps=1e-15,
                              cudnn_enable=False)

        with marquetry.test_mode():
            y = functions.layer_normalization(x, gamma, beta)

        self.assertTrue(array_close(y.data, ty.detach().numpy()))


class TestLayerNorm(unittest.TestCase):

    def test_forward1(self):
        batch_size, channels = 5, 10
        tl = torch.nn.LayerNorm(channels, eps=1e-15, elementwise_affine=False)
        l = marquetry.layers.LayerNormalization()

        for _ in range(10):
            x = np.random.randn(batch_size, channels)
            ty = tl(torch.tensor(x))
            y = l(x)
            self.assertTrue(array_close(y.data, ty.data.numpy()))

    def test_backward1(self):
        batch_size, channels = 8, 3
        x, gamma, beta = get_params(batch_size, channels, dtype=np.float64)
        f = lambda x: functions.layer_normalization(x, gamma, beta)

        self.assertTrue(gradient_check(f, x))

    def test_backward2(self):
        batch_size, channels = 8, 3
        x, gamma, beta = get_params(batch_size, channels, dtype=np.float64)
        f = lambda gamma: functions.layer_normalization(x, gamma, beta)

        self.assertTrue(gradient_check(f, gamma))

    def test_backward3(self):
        batch_size, channels = 8, 3
        x, gamma, beta = get_params(batch_size, channels, dtype=np.float64)
        f = lambda beta: functions.layer_normalization(x, gamma, beta)

        self.assertTrue(gradient_check(f, beta))

