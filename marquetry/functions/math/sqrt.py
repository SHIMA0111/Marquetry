from marquetry import cuda_backend
from marquetry import Function


class Sqrt(Function):
    """Calculate the square root value of an input.

        This class compute the square root value of the input.

        Note:
            Generally, you don't need to execute ``forward`` and ``backward`` method manually.
            You should use only ``__call__`` method.
    """
    def __init__(self, eps):
        self.eps = eps

    def forward(self, x):
        xp = cuda_backend.get_array_module(x)

        self.retain_outputs((0,))
        self.retain_inputs(())
        data_type = str(x.dtype)
        if "int" in data_type:
            bit_num = data_type.split("int")[1]
            data_type = "float" + bit_num

        return xp.sqrt(x, dtype=data_type)

    def backward(self, x, grad_y):
        y = self.output_data[0]
        grad_x = grad_y[0] / (2. * y + self.eps)

        return grad_x


def sqrt(x, eps=1e-15):
    """Calculate the square root value of an input array.

        This function computes the square root value of the input.

        Args:
            x (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The input values.
            eps (float): A small constant value to prevent zero-division

        Returns:
            :class:`marquetry.Container`: The square root result of the input values.

        Examples:
            >>> x = np.array([[2, 4, 6], [1, 2, 3]])
            >>> sqrt(x)
            container([[1.41421356 2.         2.44948974]
                       [1.         1.41421356 1.73205081]])

    """

    return Sqrt(eps)(x)
