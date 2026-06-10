import unittest

import numpy as np

import marquetry
import marquetry.functions as funcs
from marquetry import Container
from marquetry.utils import array_close, gradient_check


def get_params(channels, dtype=np.float64):
    gamma = np.random.randn(channels).astype(dtype)
    beta = np.random.randn(channels).astype(dtype)
    mean = np.random.randn(channels).astype(dtype)
    var = np.abs(np.random.randn(channels).astype(dtype)) + 0.5

    return gamma, beta, mean, var


class TestBatchRenormFunction(unittest.TestCase):

    def test_default_equals_batch_normalization(self):
        # rmax=1, dmax=0 force r=1, d=0, which reduces renormalization to plain BN
        x = np.random.randn(8, 3)
        gamma, beta, mean, var = get_params(3)

        y_renorm = funcs.batch_renormalization(x, gamma, beta, mean.copy(), var.copy())
        y_norm = funcs.batch_normalization(x, gamma, beta, mean.copy(), var.copy())

        self.assertTrue(array_close(y_renorm.data, y_norm.data))

    def test_running_statistics_update_equals_batch_normalization(self):
        x = np.random.randn(8, 3)
        gamma, beta, mean, var = get_params(3)

        renorm_mean, renorm_var = mean.copy(), var.copy()
        norm_mean, norm_var = mean.copy(), var.copy()

        funcs.batch_renormalization(x, gamma, beta, renorm_mean, renorm_var)
        funcs.batch_normalization(x, gamma, beta, norm_mean, norm_var)

        self.assertTrue(array_close(renorm_mean, norm_mean))
        self.assertTrue(array_close(renorm_var, norm_var))

    def test_forward_with_active_correction(self):
        eps = 1e-15
        x = np.random.randn(8, 3) * 2.0 + 1.0
        gamma, beta, mean, var = get_params(3)
        rmax, dmax = 2.0, 1.0

        y = funcs.batch_renormalization(x, gamma, beta, mean.copy(), var.copy(), rmax=rmax, dmax=dmax)

        batch_mean = x.mean(axis=0)
        batch_std = np.sqrt(x.var(axis=0) + eps)
        running_std = np.sqrt(var + eps)
        r = np.clip(batch_std / running_std, 1.0 / rmax, rmax)
        d = np.clip((batch_mean - mean) / running_std, -dmax, dmax)
        x_hat = (x - batch_mean) / batch_std
        expected = gamma * (x_hat * r + d) + beta

        self.assertTrue(array_close(y.data, expected))

    def test_eval_mode_uses_running_statistics(self):
        eps = 1e-15
        x = np.random.randn(8, 3)
        gamma, beta, mean, var = get_params(3)

        with marquetry.test_mode():
            y = funcs.batch_renormalization(x, gamma, beta, mean.copy(), var.copy(), rmax=3.0, dmax=5.0)

        expected = gamma * (x - mean) / np.sqrt(var + eps) + beta
        self.assertTrue(array_close(y.data, expected))

    def test_backward_default(self):
        x = np.random.randn(8, 3)
        gamma, beta, mean, var = get_params(3)
        f = lambda x_in: funcs.batch_renormalization(x_in, gamma, beta, mean.copy(), var.copy())

        self.assertTrue(gradient_check(f, x))

    def test_backward_4d(self):
        x = np.random.randn(3, 2, 4, 5)
        gamma, beta, mean, var = get_params(2)
        f = lambda x_in: funcs.batch_renormalization(x_in, gamma, beta, mean.copy(), var.copy())

        self.assertTrue(gradient_check(f, x))

    def test_backward_with_saturated_correction(self):
        # running stats far from batch stats -> r and d sit on their clip bounds,
        # where they are locally constant and the stop-gradient backward is exact
        x = np.random.randn(8, 3)
        gamma = np.random.randn(3)
        beta = np.random.randn(3)
        mean = np.full(3, 10.0)
        var = np.full(3, 100.0)
        f = lambda x_in: funcs.batch_renormalization(
            x_in, gamma, beta, mean.copy(), var.copy(), rmax=1.5, dmax=0.5)

        self.assertTrue(gradient_check(f, x))

    def test_backward_gamma(self):
        x = np.random.randn(8, 3)
        gamma, beta, mean, var = get_params(3)
        f = lambda gamma_in: funcs.batch_renormalization(
            x, gamma_in, beta, mean.copy(), var.copy(), rmax=2.0, dmax=1.0)

        self.assertTrue(gradient_check(f, gamma))

    def test_backward_eval_mode(self):
        x = Container(np.random.randn(8, 3))
        gamma, beta, mean, var = get_params(3)

        with marquetry.test_mode():
            y = funcs.batch_renormalization(x, gamma, beta, mean.copy(), var.copy())
        y.backward()

        expected = np.ones((8, 3)) * gamma / np.sqrt(var + 1e-15)
        self.assertTrue(array_close(x.grad.data, expected))


class TestBatchRenormLayer(unittest.TestCase):

    def test_default_equals_batch_norm_layer(self):
        x = np.random.randn(8, 4).astype(np.float32)

        renorm_layer = marquetry.layers.BatchRenormalization()
        norm_layer = marquetry.layers.BatchNormalization()

        y_renorm = renorm_layer(x)
        y_norm = norm_layer(x)

        self.assertTrue(array_close(y_renorm.data, y_norm.data))
        self.assertTrue(array_close(renorm_layer.avg_mean.data, norm_layer.avg_mean.data))
        self.assertTrue(array_close(renorm_layer.avg_var.data, norm_layer.avg_var.data))

    def test_eval_after_train(self):
        renorm_layer = marquetry.layers.BatchRenormalization(rmax=2.0, dmax=1.0)

        for _ in range(5):
            renorm_layer(np.random.randn(16, 4).astype(np.float32) * 2.0 + 1.0)

        x = np.random.randn(16, 4).astype(np.float32)
        with marquetry.test_mode():
            y = renorm_layer(x)

        expected = (x - renorm_layer.avg_mean.data) / np.sqrt(renorm_layer.avg_var.data + 1e-15)
        self.assertTrue(array_close(y.data, expected, atol=1e-4))

    def test_4d_shape(self):
        renorm_layer = marquetry.layers.BatchRenormalization(rmax=2.0, dmax=1.0)
        x = np.random.randn(2, 3, 4, 4).astype(np.float32)

        y = renorm_layer(x)

        self.assertEqual(y.shape, (2, 3, 4, 4))
