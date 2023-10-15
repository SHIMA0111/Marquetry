from marquetry import cuda_backend
from marquetry import configuration
from marquetry import Function
from marquetry import functions


class LayerNormalization(Function):
    """Apply layer normalization to the input container.

        Layer normalization is a technique used in deep neural networks to stabilize and accelerate training.
        It normalizes the input data by normalization and  adjustable parameters called gamma(scale factor) and
        beta(shift factor).
        This doesn't depend on the batch size due to normalize by the only 1 layer values.
        In some use cases, the stability is better than :class:`marquetry.functions.BatchNormalization`.
        This helps in preventing issues like vanishing gradients, reduce co-variate shift,
        and allow for faster convergence.

        Note:
            Generally, you don't need to execute ``forward`` and ``backward`` method manually.
            You should use only ``__call__`` method.

    """

    def __init__(self, eps=1e-15):
        self.eps = eps
        self.inv_std = None

    def forward(self, x, gamma, beta):
        assert x.ndim in (2, 4)

        x_ndim = x.ndim
        x_shape = x.shape

        if x_ndim == 4:
            batch_size, channels, height, width = x_shape
            x = x.reshape(batch_size, -1)

        xp = cuda_backend.get_array_module(x)

        mean = x.mean(axis=1, keepdims=True)
        var = x.var(axis=1, keepdims=True)
        std = xp.sqrt(var + self.eps)
        inv_std = 1. / std
        x_hat = (x - mean) * inv_std

        scaled_x = x_hat * gamma
        y = scaled_x + beta

        if x_ndim == 4:
            y = y.reshape(*x_shape)

        if configuration.config.enable_backprop:
            self.inv_std = inv_std
            self.var = var

        self.retain_inputs((0, 1))
        return y

    def backward(self, inputs, grad_y):
        grad_y = grad_y[0]

        gy_ndim = grad_y.ndim
        gy_shape = grad_y.shape
        if gy_ndim == 4:
            batch_size, channels, height, width = gy_shape
            grad_y = grad_y.reshape(batch_size, -1)

        x, gamma, _ = inputs

        if x.ndim == 4:
            batch_size, channels, height, width = x.shape
            x = x.reshape(batch_size, -1)

        x_data_dim = x.shape[1]

        mean = x.mean(axis=1, keepdims=True)
        xc = (x - mean) * self.inv_std

        grad_beta = functions.sum(grad_y, axis=0)
        grad_gamma = functions.sum(xc * grad_y, axis=0)

        g_x_hat = grad_y * gamma

        g_inv_std = functions.sum(g_x_hat * (x - mean), axis=1, keepdims=True)
        g_x_mu_1 = g_x_hat * self.inv_std

        g_std = g_inv_std * (- 1. / (self.var + self.eps))
        g_var = g_std * 0.5 * self.inv_std

        g_squ_x_mu = g_var * (1. / x_data_dim)
        g_x_mu_2 = g_squ_x_mu * 2 * (x - mean)

        g_x_1 = g_x_mu_1 + g_x_mu_2
        g_mu = functions.sum(g_x_1, axis=1, keepdims=True) * (- 1.)

        g_x_2 = g_mu * (1. / x_data_dim)

        grad_x = g_x_1 + g_x_2

        if gy_ndim == 4:
            grad_x = grad_x.reshape(*gy_shape)

        return grad_x, grad_gamma, grad_beta


def layer_normalization(x, gamma, beta, eps=1e-15):
    """Apply layer normalization to the input container.

        Layer normalization is a technique used in deep neural networks to stabilize and accelerate training.
        It normalizes the input data by normalization and  adjustable parameters called gamma(scale factor) and
        beta(shift factor).
        This doesn't depend on the batch size due to normalize by the only 1 layer values.
        In some use cases, the stability is better than :class:`marquetry.functions.BatchNormalization`.
        This helps in preventing issues like vanishing gradients, reduce co-variate shift,
        and allow for faster convergence.

        Args:
            x (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The input tensor.
            gamma (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The scale factor. It allows the network to learn the optimal scaling for each feature.
            beta (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The shift factor. It allows the network to learn the optimal mean for each feature.
            eps (float): A small value to prevent division by zero.
                Default is 1e-15.

        Caution:
            Generally use case, you can use LayerNormalization in :mod:`marquetry.layers`.
            The layer components manage the params like gamma, beta itself.
            Therefore, we suggest to use it for your network if you have no special meaning to use this function.

        Returns:
            marquetry.Container: The normalized and scaled input container.

        References:
            Layer Normalization
            (https://arxiv.org/abs/1607.06450)

    """

    return LayerNormalization(eps)(x, gamma, beta)
