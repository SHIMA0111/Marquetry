from marquetry import cuda_backend
from marquetry import Function


class ReLU(Function):
    """Rectifier Linear Unit.

        Note:
            Generally, you don't need to execute ``forward`` and ``backward`` method manually.
            You should use only ``__call__`` method.
    """
    def forward(self, x):
        xp = cuda_backend.get_array_module(x)
        y = xp.maximum(x, 0.0)

        return y

    def backward(self, x, grad_y):
        x, = x
        mask = x > 0
        grad_x = grad_y[0] * mask

        return grad_x


def relu(x):
    """Rectified Linear Unit function.

        if x >= 0, :math:`f(x) = x` || if x < 0, :math:`f(x) = 0`

        Args:
            x (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                Input container that is float array.

        Returns:
            marquetry.Container: Output container. A float array.

        Examples:

            >>> x = np.array([[-1, 0], [2, -3], [-2, 1]], 'f')
            >>> x
            array([[-1.,  0.],
                   [ 2., -3.],
                   [-2.,  1.]], dtype=float32)
            >>> relu(x)
            container([[0. 0.]
                       [2. 0.]
                       [0. 1.]])

    """
    return ReLU()(x)
