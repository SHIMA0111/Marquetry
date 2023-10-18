from marquetry import Function
from marquetry import functions
from marquetry.utils import sigmoid_array


class Swish(Function):
    """Swish Function.

        This class implements the Swish activation function,
        which is also known as ``SiLU (Sigmoid-weighted Linear Unit)``.
        Swish is anticipated as a successor of the ReLU(:class:`marquetry.function.relu`).

        Note:
            Generally, you don't need to execute ``forward`` and ``backward`` method manually.
            You should use only ``__call__`` method.
    """

    def __init__(self, beta):
        self.beta = beta
        self.sigmoid = None

    def forward(self, x):
        sigmoid = sigmoid_array(self.beta * x)
        self.sigmoid = sigmoid

        y = x * sigmoid

        return y

    def backward(self, x, grad_y):
        grad_x = self.sigmoid * (1 + self.beta * (1 - self.sigmoid) * x[0]) * grad_y[0]

        return grad_x


class DynamicSwish(Function):
    """Dynamic Swish Function.

        This class implements the Swish activation function with learnable `beta`.

        Note:
            Generally, you don't need to execute ``forward`` and ``backward`` method manually.
            You should use only ``__call__`` method.
    """

    def __init__(self):
        self.beta = None
        self.sigmoid = None

    def forward(self, x, beta):
        self.beta = beta

        sigmoid = sigmoid_array(self.beta * x)
        self.sigmoid = sigmoid

        y = x * sigmoid

        self.retain_inputs((0,))
        return y

    def backward(self, x, grad_y):
        x, _ = x

        grad_x = self.sigmoid * (1 + self.beta * (1 - self.sigmoid) * x) * grad_y[0]
        grad_beta = self.sigmoid * (1 - self.sigmoid) * x ** 2 * grad_y[0]
        grad_beta = functions.sum(grad_beta)

        return grad_x, grad_beta


def swish(x, beta=1):
    """Swish Function.

        This function implements the Swish activation function,
        which is also known as ``SiLU (Sigmoid-weighted Linear Unit)``.
        Swish is anticipated as a successor of the ReLU(:class:`marquetry.function.relu`).

        This function is obtained by:

        :math:`f(x) = x * \sigma (beta \cdot x)`

        Args:
            x (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                Input container or ndarray that is float array.
            beta (int or float): A learnable parameter for swish function.

        Caution:
            This activation has learnable parameter
            (if the ``beta`` is constant value, it is the same as :meth:`swish`).
            In generally, we suggest to use :class:`marquetry.layers.DynamicSwish` instead of this.

            The layers object manages the leanable parameter itself so in Swish, the `beta` is managed by the object.

        Returns:
            marquetry.Container: Output container. A float array.

        References:
            - Searching for Activation Functions (https://arxiv.org/abs/1710.05941)
            - Swish: a Self-Gated Activation Function (https://arxiv.org/abs/1710.05941v1)

        Examples:

            >>> x = np.array([[-1, 0], [2, -3], [-2, 1]], 'f')
            >>> x
            array([[-1.,  0.],
                   [ 2., -3.],
                   [-2.,  1.]], dtype=float32)
            >>
            >>> dynamic_swish(x, beta=0.3)
            container([[-0.42555746  0.        ]
                       [ 1.2913126  -0.8671515 ]
                       [-0.70868737  0.5744425 ]])

    """

    return Swish(beta)(x)


def dynamic_swish(x, beta):
    """Dynamic Swish Function.

        This function implements the Swish activation function with learnable `beta`.

        This function is obtained by:

        :math:`f(x) = x * \sigma (beta \cdot x)`
            - ``beta`` is learnable value

        Args:
            x (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                Input container or ndarray that is float array.
            beta (int or float): A hyperparameter for swish function.

        Returns:
            marquetry.Container: Output container. A float array.

        References:
            - Searching for Activation Functions (https://arxiv.org/abs/1710.05941)
            - Swish: a Self-Gated Activation Function (https://arxiv.org/abs/1710.05941v1)

        Examples:

            >>> x = np.array([[-1, 0], [2, -3], [-2, 1]], 'f')
            >>> x
            array([[-1.,  0.],
                   [ 2., -3.],
                   [-2.,  1.]], dtype=float32)
            >>
            >>> swish(x)
            container([[-0.26894143  0.        ]
                       [ 1.761594   -0.14227763]
                       [-0.23840584  0.7310586 ]])

    """

    return DynamicSwish()(x, beta)
