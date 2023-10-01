from marquetry import cuda_backend
from marquetry import Function


class Softmax(Function):
    """Softmax activation function.

        Note:
            Generally, you don't need to execute ``forward`` and ``backward`` method manually.
            You should use only ``__call__`` method.
    """
    def __init__(self, axis):
        self.axis = axis

    def forward(self, x):
        xp = cuda_backend.get_array_module(x)

        y = x - x.max(axis=self.axis, keepdims=True)
        y = xp.exp(y)
        y /= y.sum(axis=self.axis, keepdims=True)

        self.retain_inputs(())
        self.retain_outputs((0,))
        return y

    def backward(self, x, grad_y):
        y = self.output_data[0]
        grad_x = y * grad_y[0]
        sum_grad_x = grad_x.sum(axis=self.axis, keepdims=True)
        grad_x -= y * sum_grad_x

        return grad_x


def softmax(x, axis=1):
    """Softmax function.

        :math:`f(x) = exp(x) / Σexp(x)`

        Args:
            x (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                Input container that is float array.
            axis (int): The softmax sum axis

        Returns:
            marquetry.Container: Output container. A float array.

        Examples:

            >>> x = np.array([[-1, 0], [2, -3], [-2, 1]], 'f')
            >>> x
            array([[-1.,  0.],
                   [ 2., -3.],
                   [-2.,  1.]], dtype=float32)
            >>> softmax(x, axis=1)
            container([[0.26894143 0.7310586 ]
                       [0.9933072  0.00669285]
                       [0.04742587 0.95257413]])

    """

    return Softmax(axis)(x)
