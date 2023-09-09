from marquetry import cuda_backend
from marquetry import Function


class Tanh(Function):
    """Hyperbolic Tangent function."""
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

        f(x) = {(exp(x) - exp(-x)) / (exp(x) + exp(-x))}

        Args:
            x (:class:`marquetry.Variable` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                Input variable that is float array.

        Returns:
            marquetry.Variable: Output variable. A float array.

        Examples:

            >>> x = np.array([[-1, 0], [2, -3], [-2, 1]], 'f')
            >>> x
            array([[-1.,  0.],
                   [ 2., -3.],
                   [-2.,  1.]], dtype=float32)
            >>> tanh(x, axis=1)
            matrix([[-0.7615942  0.       ]
                    [ 0.9640276 -0.9950548]
                    [-0.9640276  0.7615942]])

    """

    return Tanh()(x)
