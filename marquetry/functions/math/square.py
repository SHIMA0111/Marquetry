import numpy as np

from marquetry import cuda_backend
from marquetry import Function


class Square(Function):
    """Calculate the squared value of an input.

        This class compute the squared value of the input.

        Note:
            Generally, you don't need to execute ``forward`` and ``backward`` method manually.
            You should use only ``__call__`` method.
    """

    def forward(self, x):
        xp = cuda_backend.get_array_module(x)

        data_type = str(x.dtype)
        if "int" in data_type:
            bit_num = data_type.split("int")[1]
            data_type = "float" + bit_num

        return xp.square(x, dtype=data_type)

    def backward(self, x, grad_y):
        grad_x = grad_y[0] * 2. * x[0]

        return grad_x


def square(x):
    """Calculate the squared value of an input array.

        This function computes the squared value of the input.

        Args:
            x (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The input values.

        Returns:
            :class:`marquetry.Container`: The squared result of the input values.

        Examples:
            >>> x = np.array([[2, 4, 6], [1, 2, 3]])
            >>> sqrt(x)
            container([[ 4. 16. 36.]
                       [ 1.  4.  9.]])

    """

    return Square()(x)
