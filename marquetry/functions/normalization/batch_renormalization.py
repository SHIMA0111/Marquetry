from marquetry import configuration
from marquetry import cuda_backend
from marquetry import Function
from marquetry import functions


class BatchRenormalization(Function):
    """Apply batch renormalization to the input container.

        Batch renormalization extends batch normalization so that train-time
        normalization stays close to the inference-time normalization even with
        small or non-i.i.d. mini-batches. The batch statistics are corrected toward
        the running statistics by the clipped factors ``r`` and ``d``:

        ``r = clip(sigma_batch / sigma_running, 1 / rmax, rmax)``
        ``d = clip((mu_batch - mu_running) / sigma_running, -dmax, dmax)``

        and the output is ``y = gamma * (x_hat * r + d) + beta`` where
        ``x_hat`` is the batch-normalized input. ``r`` and ``d`` are treated
        as constants in the backward computation as defined in the paper.

        With ``rmax=1`` and ``dmax=0`` this is mathematically identical to
        :class:`marquetry.functions.BatchNormalization`.

        Note:
            Generally, you don't need to execute ``forward`` and ``backward`` method manually.
            You should use only ``__call__`` method.

        References:
            Batch Renormalization: Towards Reducing Minibatch Dependence
            in Batch-Normalized Models (https://arxiv.org/abs/1702.03275)
    """

    def __init__(self, mean, var, decay, eps, rmax, dmax):
        self.avg_mean = mean
        self.avg_var = var
        self.decay = decay
        self.eps = eps
        self.rmax = rmax
        self.dmax = dmax

        self.inv_std = None
        self.r = None
        self.d = None
        self._train_mode = None

    def forward(self, x, gamma, beta):
        assert x.ndim in (2, 4)

        x_ndim = x.ndim
        x_shape = x.shape
        if x_ndim == 4:
            batch_size, channels, height, width = x_shape
            x = x.transpose(0, 2, 3, 1).reshape(-1, channels)

        xp = cuda_backend.get_array_module(x)

        self._train_mode = configuration.config.train
        if self._train_mode:
            mean = x.mean(axis=0)
            var = x.var(axis=0)
            std = xp.sqrt(var + self.eps)
            running_std = xp.sqrt(self.avg_var + self.eps)

            self.r = xp.clip(std / running_std, 1.0 / self.rmax, self.rmax)
            self.d = xp.clip((mean - self.avg_mean) / running_std, -self.dmax, self.dmax)
            self.inv_std = 1.0 / std

            normed_x = (x - mean) * self.inv_std
            renormed_x = normed_x * self.r + self.d

            samples = x.size // gamma.size
            scale = samples - 1. if samples - 1. > 1. else 1.
            adjust = samples / scale
            self.avg_mean *= self.decay
            self.avg_mean += (1 - self.decay) * mean

            self.avg_var *= self.decay
            self.avg_var += (1 - self.decay) * adjust * var
        else:
            self.inv_std = 1.0 / xp.sqrt(self.avg_var + self.eps)
            self.r = xp.ones_like(gamma)
            self.d = xp.zeros_like(gamma)

            renormed_x = (x - self.avg_mean) * self.inv_std

        y = gamma * renormed_x + beta

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

        if x.ndim == 4:
            batch_size, channels, height, width = x.shape
            x = x.transpose(0, 2, 3, 1).reshape(-1, channels)

        batch_size = len(x)

        if self._train_mode:
            mean = x.sum(axis=0) / batch_size
            xc = (x - mean) * self.inv_std
            xc_renormed = xc * self.r + self.d

            grad_beta = functions.sum(grad_y, axis=0)
            grad_gamma = functions.sum(xc_renormed * grad_y, axis=0)
            grad_sigma = functions.sum(xc * grad_y, axis=0)

            grad_x = grad_y - grad_beta / batch_size - xc * grad_sigma / batch_size
            grad_x *= gamma * self.r * self.inv_std
        else:
            xc = (x - self.avg_mean) * self.inv_std

            grad_beta = functions.sum(grad_y, axis=0)
            grad_gamma = functions.sum(xc * grad_y, axis=0)

            grad_x = grad_y * (gamma * self.inv_std)

        if gy_ndim == 4:
            batch_size, channels, height, width = gy_shape
            grad_x = grad_x.reshape(batch_size, height, width, channels).transpose(0, 3, 1, 2)

        return grad_x, grad_gamma, grad_beta


def batch_renormalization(x, gamma, beta, mean, var, rmax=1.0, dmax=0.0, decay=0.9, eps=1e-15):
    """Apply batch renormalization to the input tensor.

        Batch renormalization corrects the batch statistics toward the running statistics
        using the clipped factors ``r`` and ``d``, which keeps the train-time behavior
        close to the inference-time behavior even with small mini-batches.

        Args:
            x (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The input tensor.
            gamma (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The scale factor.
            beta (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The shift factor.
            mean (:class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The running mean which is updated in train mode and used in test mode.
            var (:class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The running variance which is updated in train mode and used in test mode.
            rmax (float): The clipping bound of the std correction factor ``r``.
                ``1.0`` (default) disables the correction which means the behavior is
                identical to batch normalization. The paper suggests increasing this
                gradually from 1 during training.
            dmax (float): The clipping bound of the mean correction factor ``d``.
                ``0.0`` (default) disables the correction.
            decay (float): The decay rate of the running statistics. Default is 0.9.
            eps (float): A small value to prevent division by zero. Default is 1e-15.

        Caution:
            Generally use case, you can use BatchRenormalization in :mod:`marquetry.layers`.
            The layer component manages gamma, beta, and the running statistics itself.

        Returns:
            marquetry.Container: The renormalized and scaled input tensor.

        References:
            Batch Renormalization: Towards Reducing Minibatch Dependence
            in Batch-Normalized Models (https://arxiv.org/abs/1702.03275)
    """

    return BatchRenormalization(mean, var, decay, eps, rmax, dmax)(x, gamma, beta)
