from marquetry import Function


class LeakyReLU(Function):
    """Leaky Rectifier Linear Unit.

        Note:
            Generally, you don't need to execute ``forward`` and ``backward`` method manually.
            You should use only ``__call__`` method.
    """

    def __init__(self, slope):
        self.slope = slope

    def forward(self, x):
        y = x.copy()
        y[x <= 0] *= self.slope

        return y

    def backward(self, x, grad_y):
        mask = (x[0] > 0).astype(grad_y[0].dtype)
        mask[mask <= 0] = self.slope

        grad_x = grad_y[0] * mask

        return grad_x


def leaky_relu(x, slope=0.2):
    """Leaky Rectified Linear Unit function.

        This function is improved version of the ReLU(Rectified Linear Unit).

        if x >= 0, :math:`f(x) = x` || if x < 0, :math:`f(x) = slope * x`
            - The ``slope`` is a small constant value the default value is `0.2`.

        Args:
            x (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                Input container that is float array.
            slope (float): small constant value. The default is 0.2.

        Returns:
            marquetry.Container: Output container. A float array.

        Examples:

            >>> x = np.array([[-1, 0], [2, -3], [-2, 1]], 'f')
            >>> x
            array([[-1.,  0.],
                   [ 2., -3.],
                   [-2.,  1.]], dtype=float32)
            >>> leaky_relu(x, slope=0.2)
            container([[-0.2  0. ]
                       [ 2.  -0.6]
                       [-0.4  1. ]])

    """

    return LeakyReLU(slope)(x)
