from marquetry import cuda_backend
from marquetry import Function
from marquetry import functions


class Mish(Function):
    """Mish Function.

        This class implements the Mish activation function.
        Mish is anticipated as one of successors of the ReLU(:class:`marquetry.functions.relu`).
        And it can take some negative values unlike ReLU.

        Note:
            Generally, you don't need to execute ``forward`` and ``backward`` method manually.
            You should use only ``__call__`` method.
    """

    def __init__(self):
        self.tanh_softplus = None

    def forward(self, x):
        xp = cuda_backend.get_array_module(x)

        softplus = xp.maximum(x, 0) + xp.log(1 + xp.exp(-xp.abs(x)))
        tanh_softpuls = xp.tanh(softplus)

        y = x * tanh_softpuls

        self.tanh_softplus = tanh_softpuls

        return y

    def backward(self, x, grad_y):
        x = x[0]

        sigmoid = functions.sigmoid(x)

        grad_x = (self.tanh_softplus + (sigmoid * (1 - self.tanh_softplus ** 2) * x)) * grad_y[0]

        return grad_x


def mish(x):
    """Mish Function.

        This functioncd implements the Mish activation function.
        Mish is anticipated as one of successors of the ReLU(:class:`marquetry.functions.relu`).
        And it can take some negative values unlike ReLU.

        This function is obtained by:

        :math:`f(x) = x * Tanh(Softplus(x))`

        Args:
            x (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                Input container or ndarray that is float array.

        Returns:
            marquetry.Container: Output container. A float array.

        References:
            - Mish: A Self Regularized Non-Monotonic Activation Function (https://arxiv.org/abs/1908.08681)

        Examples:

            >>> x = np.array([[-1, 0], [2, -3], [-2, 1]], 'f')
            >>> x
            array([[-1.,  0.],
                   [ 2., -3.],
                   [-2.,  1.]], dtype=float32)
            >>
            >>> mish(x)
            container([[-0.30340144  0.        ]
                       [ 1.943959   -0.14564739]
                       [-0.25250155  0.86509836]])

    """

    return Mish()(x)
