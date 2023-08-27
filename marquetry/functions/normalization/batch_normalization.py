import marquetry
from marquetry import Function, cuda_backend, functions, configuration


class BatchNormalization(Function):
    def __init__(self, mean, var, decay, eps):
        self.avg_mean = mean
        self.avg_var = var
        self.decay = decay
        self.eps = eps

        self.inv_std = None

    def forward(self, x, gamma, beta):
        assert x.ndim in (2, 4)

        x_ndim = x.ndim
        x_shape = x.shape
        if x_ndim == 4:
            batch_size, channels, height, width = x_shape
            x = x.transpose(0, 2, 3, 1).reshape(-1, channels)

        xp = cuda_backend.get_array_module(x)

        if configuration.config.train:
            mean = x.mean(axis=0)
            var = x.var(axis=0)
            inv_std = 1 / xp.sqrt(var + self.eps)
            normed_x = (x - mean) * inv_std

            samples = x.size // gamma.size
            scale = samples - 1. if samples - 1. > 1. else 1.
            adjust = samples / scale
            self.avg_mean *= self.decay
            self.avg_mean += (1 - self.decay) * mean

            self.avg_var *= self.decay
            self.avg_var += (1 - self.decay) * adjust * var

            self.inv_std = inv_std
        else:
            inv_std = 1 / xp.sqrt(self.avg_var + self.eps)
            normed_x = (x - self.avg_mean) * inv_std

        y = gamma * normed_x + beta

        if x_ndim == 4:
            batch_size, channels, height, width = x_shape
            y = y.reshape(batch_size, height, width, channels).transpose(0, 3, 1, 2)

        self.retain_inputs((0, 1))
        return y

    def backward(self, inputs, grad_y):
        grad_y = grad_y[0]

        gy_ndim = grad_y.ndim
        gy_shape = grad_y.shape
        if gy_ndim == 4:
            batch_size, channels, height, width = gy_shape
            grad_y = grad_y.transpose(0, 2, 3, 1).reshape(-1, channels)

        x, gamma, _ = inputs
        batch_size = len(x)

        if x.ndim == 4:
            batch_size, channels, height, width = x.shape
            x = x.transpose(0, 2, 3, 1).reshape(-1, channels)

        mean = x.sum(axis=0) / batch_size
        xc = (x - mean) * self.inv_std

        grad_beta = functions.sum(grad_y, axis=0)
        grad_gamma = functions.sum(xc * grad_y, axis=0)

        grad_x = grad_y - grad_beta / batch_size - xc * grad_gamma / batch_size
        grad_x *= gamma * self.inv_std

        if gy_ndim == 4:
            batch_size, channels, height, width = gy_shape
            grad_x = grad_x.reshape(batch_size, height, width, channels).transpose(0, 3, 1, 2)

        return grad_x, grad_gamma, grad_beta


def batch_normalization(x, gamma, beta, mean, var, decay=0.9, eps=1e-15):
    return BatchNormalization(mean, var, decay, eps)(x, gamma, beta)
