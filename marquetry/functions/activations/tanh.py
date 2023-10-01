from marquetry import cuda_backend
from marquetry import Function


class Tanh(Function):
    """Hyperbolic Tangent function.

        Note:
            Generally, you don't need to execute ``forward`` and ``backward`` method manually.
            You should use only ``__call__`` method.
    """
    def forward(self, x):
        xp = cuda_backend.get_array_module(x)
        y = xp.tanh(x)

        self.retain_inputs(())
        self.retain_outputs((0,))
        return y

    def backward(self, x, grad_y):
        y = self.output_data[0]
        grad_x = grad_y[0] * (1 - y ** 2)

        return grad_x


def tanh(x):
    """Hyperbolic Tangent function.

        This function's result is obtained -1.0 ~ 1.0.

        :math:`f(x) = {(exp(x) - exp(-x)) / (exp(x) + exp(-x))}`

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
            >>> tanh(x, axis=1)
            container([[-0.7615942  0.       ]
                       [ 0.9640276 -0.9950548]
                       [-0.9640276  0.7615942]])

    """

    return Tanh()(x)
