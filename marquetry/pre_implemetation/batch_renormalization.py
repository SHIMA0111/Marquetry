import numpy as np

from marquetry import configuration
from marquetry import cuda_backend
from marquetry import Function


class BatchRenormalization(Function):
    def __init__(self, mean, var, decay, eps, rmax, dmax, update_delta):
        self.running_mean = mean
        self.running_var = var
        self.rmax = rmax
        self.dmax = dmax
        self.r = None
        self.d = None
        self.decay = decay
        self.eps = eps
        self.update_delta = update_delta

        self.all_mean = None
        self.all_var = None

        self.std = None
        self.x_hat = None
        self.x_hat_renorm = None

    def forward(self, *input_data):
        x, gamma, beta = input_data[:3]

        assert x.ndim in (2, 4)

        x_ndim = x.ndim
        x_shape = x.shape
        if x_ndim == 4:
            batch_size, channels, height, width = x_shape
            x = x.transpose(0, 2, 3, 1).reshape(-1, channels)

        xp = cuda_backend.get_array_module(x)

        if len(input_data) != 5:
            assert configuration.config.train

        elif len(input_data) == 5:
            fixed_mean = input_data[3]
            fixed_var = input_data[4]

        if configuration.config.train:
            mean = x.mean(axis=0)
            var = x.var(axis=0)

            self.std = xp.sqrt(var, dtype=var.dtype)

            if self.r is not None:
                running_sigma = xp.sqrt(self.running_var + self.eps, dtype=self.running_mean.dtype)
                self.r = xp.clip(self.std / running_sigma, 1 / self.rmax, self.rmax)
                self.d = xp.clip((mean - self.running_mean) / running_sigma, -self.dmax, self.dmax)

                m = x.size // gamma.size
                self.running_mean *= self.decay
                temp_ar = xp.array(mean)
                temp_ar *= (1 - self.decay)
                self.running_mean += temp_ar
                del temp_ar

                self.running_var *= self.decay
                adjust = m / max(m - 1., 1.)
                temp_ar = xp.array(var)
                temp_ar *= (1 - self.decay) * adjust
                self.running_var += temp_ar
                del temp_ar

        else:
            mean = fixed_mean
            var = fixed_var

            self.std = xp.sqrt(var, dtype=var.dtype)

            if self.r is not None:
                self.r = xp.ones_like(gamma)
                self.d = xp.zeros_like(gamma)

        inv_std = 1 / (self.std + self.eps)
        normed_x = (x - mean) * inv_std
        renormed_x = normed_x * self.r + self.d

        if configuration.config.train:
            self.r

        y = gamma * renormed_x + beta

        if x_ndim == 4:
            batch_size, channels, height, width = x_shape
            y = y.reshape(batch_size, height, width, channels).transpose(0, 3, 1, 2)

        return y

    def backward(self, inputs, grad_outputs):
        x, gamma = inputs[:2]
        grad_y = grad_outputs[0]
        head_ndim = gamma.ndim + 1
        expander = (None, Ellipsis) + (None,) * (x.ndim - head_ndim)
        m = gamma.dtype.type(x.size // gamma.size)
        axis = (0,) + tuple(range(head_ndim, x.ndim))
        xp = cuda_backend.get_array_module(x)

        if len(inputs) == 5:
            mean = inputs[3]
            var = inputs[4] + self.eps
            std = xp.sqrt(var, dtype=var.dtype)
            gs = gamma / std
            grad_beta = grad_y.sum(axis=axis)
            x_hat = _xhat(x, mean, std, expander)
            grad_gamma = (grad_y * x_hat).sum(axis=axis)
            grad_mean = -gs * grad_beta
            grad_var = -0.5 * gamma / var * grad_gamma
            grad_x = gs[expander] * grad_y

            return grad_x, grad_gamma, grad_beta, grad_mean, grad_var

        assert configuration.config.train

        grad_beta = grad_y.sum(axis=axis)
        grad_gamma = (grad_y * self.x_hat_renorm).sum(axis=axis)
        grad_sigma_batch = (grad_y * self.x_hat).sum(axis=axis)

        if xp is np:
            scale = (self.r * gamma / self.std)[expander]
            grad_x = scale * (grad_y - (self.x_hat * grad_sigma_batch[expander] + grad_beta[expander]) / m)

        else:
            inv_m = np.float32(1) / m
            grad_x = xp.elementwise(
                'T grad_y, T x_hat, T gamma, T std, T grad_sigma_batch, T grad_beta, T inv_m, T r',
                'T grad_x'
                'grad_x = (r * gamma / std) * (grad_y - (x_hat * grad_sigma_batch + grad_beta) * inv_m)',
                'bn_bwd')(grad_y, self.x_hat, gamma[expander], self.std[expander],
                          grad_sigma_batch[expander], grad_beta[expander], inv_m, self.r[expander])

        return grad_x, grad_gamma, grad_beta


def batch_renormalization(x, gamma, beta, rmax, dmax, mean, var, eps=1e-15, decay=0.9):
    return BatchRenormalization(mean, var, decay, eps, rmax, dmax)(x, gamma, beta)
