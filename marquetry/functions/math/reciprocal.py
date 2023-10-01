from marquetry import cuda_backend
from marquetry import Function


class Reciprocal(Function):
    """Calculate the reciprocal of the input tensor.

        Note:
            Generally, you don't need to execute ``forward`` and ``backward`` method manually.
            You should use only ``__call__`` method.
    """

    def __init__(self, dtype="f"):
        self.dtype = dtype

    def forward(self, x):
        xp = cuda_backend.get_array_module(x)
        y = xp.reciprocal(x, dtype=self.dtype)

        return y

    def backward(self, x, grad_y):
        grad_x = -(1 / x[0] ** 2) * grad_y[0]

        return grad_x


def reciprocal(x, dtype="f"):
    """Calculate the reciprocal of the input tensor.

        Reciprocal is calculated as:
            reciprocal = 1 / x

        Args:
            x (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The input tensor.

        Returns:
            :class:`marquetry.Container`: The reciprocal of the input tensor.

        Examples:
                >>> x = np.array([[1, 3, 2], [5, 2, 4]])
                >>> x
                array([[1, 3, 2],
                       [5, 2, 4]])
                >>> reciprocal(x, dtype=np.float64)
                container([[1.         0.33333333 0.5       ]
                           [0.2        0.5        0.25      ]])
    """

    return Reciprocal(dtype)(x)
