from marquetry import cuda_backend
from marquetry import Function


class Sigmoid(Function):
    """Logistic Sigmoid Function.

        Note:
            Generally, you don't need to execute ``forward`` and ``backward`` method manually.
            You should use only ``__call__`` method.
    """
    def forward(self, x):
        xp = cuda_backend.get_array_module(x)

        y = xp.exp(xp.minimum(0, x)) / (1 + xp.exp(-xp.abs(x)))

        self.retain_inputs(())
        self.retain_outputs((0,))
        return y

    def backward(self, x, grad_y):
        y = self.output_data[0]
        grad_x = y * (1 - y) * grad_y[0]

        return grad_x


def sigmoid(x):
    """Logistic sigmoid function.

        This function's result is obtained 0.0 ~ 1.0.

        :math:`f(x) = {1 / (1 + exp(-x))}`

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
            >>> sigmoid(x)
            container([[0.26894143 0.5       ]
                       [0.880797   0.04742587]
                       [0.11920292 0.7310586 ]])

    """

    return Sigmoid()(x)
