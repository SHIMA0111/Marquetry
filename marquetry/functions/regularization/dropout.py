import marquetry
from marquetry import configuration
from marquetry import cuda_backend
from marquetry import Function


class Dropout(Function):
    """Apply dropout to the input tensor.

        Dropout is a regularization technique used during training to prevent overfitting.
        It works by randomly deactivating neurons in the input tensor during each forward pass.

        Note:
            Generally, you don't need to execute ``forward`` and ``backward`` method manually.
            You should use only ``__call__`` method.
    """

    def __init__(self, dropout_rate):
        self.dropout_rate = dropout_rate

        self.mask = None
        self.scale = None
        self._train_mode = None

    def forward(self, x):
        self._train_mode = configuration.config.train
        if self._train_mode:
            xp = cuda_backend.get_array_module(x)
            mask = xp.random.rand(*x.shape) > self.dropout_rate
            self.mask = mask
            scale = xp.array(1.0 - self.dropout_rate).astype(x.dtype)
            self.scale = scale
            y = x * mask / scale
        else:
            y = x

        self.retain_inputs(())

        return y

    def backward(self, x, grad_y):
        if self._train_mode:
            grad_x = grad_y[0] * (self.mask / self.scale)
        else:
            grad_x = grad_y[0]

        return grad_x


def dropout(x, dropout_rate=0.5):
    """
        Apply dropout to the input tensor.

        Dropout is a regularization technique used during training to prevent overfitting. It works by randomly
        deactivating neurons in the input tensor during each forward pass.

        Args:
            x (:class:marquetry.Container or :class:numpy.ndarray or :class:cupy.ndarray):
                The input tensor.
            dropout_rate (float): The neurons to deactivate during each forward pass.
                It should be a float between 0 and 1, representing the probability of deactivation.
                Default is 0.5.

        Returns:
            marquetry.Container: The result of applying dropout to the input tensor.

        Note:
            During training (when `marquetry.configuration.config.train` is `True`),
                this function deactivates neurons in `x` randomly based on the specified `dropout_rate`.
                It also scales the remaining activations to maintain the expected value of the output.
            During inference (when `marquetry.configuration.config.train` is `False`),
                this function returns `x` unchanged.

    """

    return Dropout(dropout_rate)(x)
