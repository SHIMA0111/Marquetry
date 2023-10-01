from marquetry import configuration
from marquetry import cuda_backend
from marquetry import Function
from marquetry import functions


class BatchNormalization(Function):
    """Apply batch normalization to the input tensor.

        Batch normalization is a technique used in
        deep neural networks to stabilize and accelerate training.
        It normalizes the input data by normalization and
        adjustable parameters called gamma(scale factor) and
        beta(shift factor).
        This helps in preventing issues like vanishing gradients,
        reduce co-variate shift, and allows for faster convergence.

        Note:
            Generally, you don't need to execute ``forward`` and ``backward`` method manually.
            You should use only ``__call__`` method.
    """
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
    """Apply batch normalization to the input tensor.

        Batch normalization is a technique used in
        deep neural networks to stabilize and accelerate training.
        It normalizes the input data by normalization and
        adjustable parameters called gamma(scale factor) and beta(shift factor).
        This helps in preventing issues like vanishing gradients,
        reduce co-variate shift, and allows for faster convergence.

        Args:
            x (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The input tensor.
            gamma (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The scale factor. It allows the network to learn the optimal scaling for each feature.
            beta (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The shift factor. It allows the network to learn the optimal mean for each feature.
            mean (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The running mean of the batch. It is updated during training and used for inference.
            var (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The running variance of the batch. It is updated during training and used for inference.
            decay (float): The decay rate for the running statistics. It controls how fast the running
                statistics adapt to changes in the input distribution.
                Default is 0.9.
            eps (float): A small value to prevent division by zero.
                Default is 1e-15.

        Caution:
            Generally use case, you can use BatchNormalization in :mod:`marquetry.layers`.
            The layer components manage the params like gamma, beta, running_mean and running_var itself.
            Therefore, we suggest to use it for your network if you have no special meaning to use this function.

        Returns:
            marquetry.Container: The normalized and scaled input tensor.

        References:
            Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
            (https://arxiv.org/abs/1502.03167)

    """

    return BatchNormalization(mean, var, decay, eps)(x, gamma, beta)
