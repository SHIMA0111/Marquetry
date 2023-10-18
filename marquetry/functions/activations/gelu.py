import numpy as np
import scipy

from marquetry import cuda_backend
from marquetry import Function
from marquetry import functions
from marquetry.utils import sigmoid_array


class GELU(Function):
    """Gaussian Error Linear Units (GELU) Function.

        This class implements the Gaussian Error Linear Units activation function,
        which is used in neural networks.
        GELU is often used with Large Transformer models.

        Note:
            Generally, you don't need to execute ``forward`` and ``backward`` method manually.
            You should use only ``__call__`` method.

    """

    approximate_methods = ("none", "tanh", "sigmoid")

    def __init__(self, approximate: str):
        if approximate.lower() not in self.approximate_methods:
            raise ValueError("GELU allows {} as approximating methods but you input {}."
                             .format(", ".join(self.approximate_methods), approximate))

        self.approximate = approximate.lower()
        self.gauss = None
        self.tanh = None
        self.sigmoid = None

    def forward(self, x):
        xp = cuda_backend.get_array_module(x)

        if self.approximate == "tanh":
            tanh = xp.tanh(xp.sqrt(2 / xp.pi) * (x + 0.044715 * xp.power(x, 3)))
            y = 0.5 * x * (1 + tanh)

            self.tanh = tanh

        elif self.approximate == "sigmoid":
            sigmoid = sigmoid_array(1.702 * x)
            y = x * sigmoid

            self.sigmoid = sigmoid

        else:
            if xp is np:
                gauss = (1 + scipy.special.erf(x / xp.sqrt(2))) / 2
            else:
                gauss = (1 + cuda_backend.cpx.scipy.special.erf(x / xp.sqrt(2))) / 2

            y = x * gauss

            self.gauss = gauss

        return y

    def backward(self, x, grad_y):
        x = x[0]
        grad_y = grad_y[0]

        if self.approximate == "tanh":
            grad_x = 1 + self.tanh
            grad_x += (functions.sqrt(2 / np.pi) * x * (1 - functions.square(self.tanh))
                       * (1 + 0.044715 * 3 * functions.square(x)))
            grad_x /= 2

        elif self.approximate == "sigmoid":
            grad_x = self.sigmoid * (1 + 1.702 * (1 - self.sigmoid) * x)

        else:
            grad_x = self.gauss + x * functions.exp(-(x ** 2 / 2)) / functions.sqrt(2 * np.pi)

        grad_x = grad_x * grad_y

        return grad_x


def gelu(x, approximate="none"):
    """Gaussian Error Linear Units (GELU) Function.

        This function implements the Gaussian Error Linear Units activation function,
        which is used in neural networks.
        GELU is often used with Large Transformer models.
        It provides the GELU function with different approximation options:

        - "none"
        - "tanh"
        - "sigmoid"

        The choice of approximation affects the computational performance and accuracy.

        This function is obtained by:

        :math:`f(x) = x \cdot \Phi (x)`
        (:math:`\Phi (x)` is the standard Gaussian cumulative distribution function.)

        The :math:`\Phi (x)` is a special function so sometimes the function is approximated by the below functions.

        :math:`f(x) = \\frac{1}{2} x \cdot (1 + Tanh[ \sqrt{ \\frac{2}{\pi}} \cdot (x + 0.04715x^3)])`

        or

        :math:`f(x) = x \cdot \sigma (1.702x)`

        Args:
            x (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                Input container or ndarray that is float array.
            approximate (str): The approximation method to use. Options are "none" for the
                exact GELU function, "tanh" for the tanh-based approximation, and "sigmoid"
                for the sigmoid-based approximation.

        Examples:
            >>> x = np.array([[1, 2, 3], [2, 4, 6]])
            >>> output = gelu(x, approximate="tanh")
            >>> print(output)
            container([[0.84119199 1.95459769 2.99636261]
                       [1.95459769 3.99992975 6.        ]])
            >>> output = gelu(x, approximate="none")
            >>> print(output)
            container([[0.84134475 1.95449974 2.99595031]
                       [1.95449974 3.99987332 5.99999999]])
            >>> output = gelu(x, approximate="sigmoid")
            >>> print(output)
            container([[0.84579577 1.93565862 2.98192869]
                       [1.93565862 3.99558528 5.99977965]])

    """

    return GELU(approximate)(x)
