from marquetry import cuda_backend
from marquetry import Function
from marquetry import functions


class Softplus(Function):
    """Softplus activation function.

        Note:
            Generally, you don't need to execute ``forward`` and ``backward`` method manually.
            You should use only ``__call__`` method.

    """
    def __init__(self, beta):
        self.beta = beta

    def forward(self, x):
        xp = cuda_backend.get_array_module(x)

        # y = xp.log(1 + xp.exp(x * self.beta)) / self.beta
        y = xp.maximum(x, 0) + xp.log(1 + xp.exp(-xp.abs(x * self.beta))) / self.beta

        return y

    def backward(self, x, grad_y):
        grad_x = functions.sigmoid(x[0] * self.beta) * grad_y[0]

        return grad_x


def softplus(x, beta=1):
    """Softplus function.

        :math:`f(x) = \\frac {1}{ \\beta} \log (1 + exp( \\beta \cdot x))`

        Args:
            x (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                Input container that is float array.
            beta (int): The hyperparameter of the fuction's sharpness.
                Default is `1`.

        Returns:
            marquetry.Container: Output container. A float array.

        Examples:

            >>> x = np.array([[-1, 0], [2, -3], [-2, 1]], 'f')
            >>> x
            array([[-1.,  0.],
                   [ 2., -3.],
                   [-2.,  1.]], dtype=float32)
            >>> softplus(x)
            container([[0.31326166 0.6931472 ]
                       [2.126928   0.04858733]
                       [0.12692805 1.3132616 ]])

    """

    return Softplus(beta)(x)
