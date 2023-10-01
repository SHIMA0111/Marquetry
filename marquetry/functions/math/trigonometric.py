from marquetry import cuda_backend
from marquetry import Function


class Sin(Function):
    """Calculate the sine(Sin) of the input tensor.

        Note:
            Generally, you don't need to execute ``forward`` and ``backward`` method manually.
            You should use only ``__call__`` method.
    """
    def forward(self, x):
        xp = cuda_backend.get_array_module(x)
        y = xp.sin(x)

        return y

    def backward(self, x, grad_y):
        grad_x = cos(x[0]) * grad_y[0]

        return grad_x


def sin(x):
    """Calculate the sine(Sin) of the input tensor.

        Args:
            x (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The input tensor.

        Returns:
            :class:`marquetry.Container`: The sine of the input tensor.

        Examples:
                >>> x = np.array([[1, 3, 2], [5, 2, 4]])
                >>> x
                array([[1, 3, 2],
                       [5, 2, 4]])
                >>> sin(x)
                container([[ 0.84147098  0.14112001  0.90929743]
                           [-0.95892427  0.90929743 -0.7568025 ]])
    """

    return Sin()(x)


class Cos(Function):
    """Calculate the cosine(Cos) of the input tensor."""

    def forward(self, x):
        xp = cuda_backend.get_array_module(x)
        y = xp.cos(x)

        return y

    def backward(self, x, grad_y):
        grad_x = -sin(x[0]) * grad_y[0]

        return grad_x


def cos(x):
    """Calculate the cosine(Cos) of the input tensor.

        Args:
            x (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The input tensor.

        Returns:
            :class:`marquetry.Container`: The cosine of the input tensor.

        Examples:
                >>> x = np.array([[1, 3, 2], [5, 2, 4]])
                >>> x
                array([[1, 3, 2],
                       [5, 2, 4]])
                >>> cos(x)
                container([[ 0.54030231 -0.9899925  -0.41614684]
                           [ 0.28366219 -0.41614684 -0.65364362]])
    """

    return Cos()(x)


class Tan(Function):
    """Calculate the tangent(Tan) of the input tensor."""

    def forward(self, x):
        xp = cuda_backend.get_array_module(x)
        y = xp.tan(x)

        self.retain_inputs(())
        self.retain_outputs((0,))

        return y

    def backward(self, x, grad_y):
        y = self.output_data[0]
        grad_x = 1 + y ** 2
        grad_x *= grad_y[0]

        return grad_x


def tan(x):
    """Calculate the tangent(Tan) of the input tensor.

        Args:
            x (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The input tensor.

        Returns:
            :class:`marquetry.Container`: The tangent of the input tensor.

        Examples:
                >>> x = np.array([[1, 3, 2], [5, 2, 4]])
                >>> x
                array([[1, 3, 2],
                       [5, 2, 4]])
                >>> tan(x)
                container([[ 1.55740772 -0.14254654 -2.18503986]
                           [-3.38051501 -2.18503986  1.15782128]])
    """

    return Tan()(x)
