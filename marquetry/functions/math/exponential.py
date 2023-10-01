from marquetry import cuda_backend
from marquetry import Function


class Exp(Function):
    """Calculate the exponential of an input.

        This class computes the exponential of the input.

        Note:
            Generally, you don't need to execute ``forward`` and ``backward`` method manually.
            You should use only ``__call__`` method.
    """

    def forward(self, x):
        xp = cuda_backend.get_array_module(x)
        y = xp.exp(x)

        self.retain_inputs(())
        self.retain_outputs((0,))
        return y

    def backward(self, x, grad_y):
        y = self.output_data[0]
        grad_x = grad_y[0] * y

        return grad_x


def exp(x):
    """Calculate the exponential of an input.

        This function computes the exponential of the input.

        Args:
            x (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The input values.

        Returns:
            :class:`marquetry.Container`: The exponential of the input.

        Examples:
            >>> x = np.array([1, 2, 3])
            >>> exp(x)
            container([ 2.71828183  7.3890561  20.08553692])
    """

    return Exp()(x)


class Log(Function):
    """Calculate the natural logarithm of an input.

        This class computes the natural logarithm of the input.

        Note:
            Generally, you don't need to execute ``forward`` and ``backward`` method manually.
            You should use only ``__call__`` method.
    """
    def forward(self, x):
        xp = cuda_backend.get_array_module(x)
        y = xp.log(x)

        return y

    def backward(self, x, grad_y):
        grad_x = grad_y[0] / x[0]

        return grad_x


def log(x):
    """Calculate the natural logarithm of an input.

        This function computes the natural logarithm of the input.

        Args:
            x (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The input values.

        Returns:
            :class:`marquetry.Container`: The natural logarithm of the input.

        Examples:
            >>> x = np.array([1, 2, 3])
            >>> log(x)
            container([0.         0.69314718 1.09861229])

    """

    return Log()(x)


class Log2(Function):
    """Calculate the base-2 logarithm of an input.

        This class computes the base-2 logarithm of the input.

        Note:
            Generally, you don't need to execute ``forward`` and ``backward`` method manually.
            You should use only ``__call__`` method.
    """

    def forward(self, x):
        xp = cuda_backend.get_array_module(x)
        y = xp.log2(x)

        return y

    def backward(self, x, grad_y):
        grad_x = grad_y[0] / (x[0] * log(2))

        return grad_x


def log2(x):
    """Calculate the base-2 logarithm of an input.

        This function computes the base-2 logarithm of the input.

        Args:
            x (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The input values.

        Returns:
            :class:`marquetry.Container`: The base-2 logarithm of the input.

        Examples:
            >>> x = np.array([1, 2, 3])
            >>> log2(x)
            container([0.        1.        1.5849625])
    """
    return Log2()(x)


class Log10(Function):
    """Calculate the base-10 logarithm of an input.

        This function computes the base-10 logarithm of the input.

        Note:
            Generally, you don't need to execute ``forward`` and ``backward`` method manually.
            You should use only ``__call__`` method.
    """

    def forward(self, x):
        xp = cuda_backend.get_array_module(x)
        y = xp.log10(x)

        return y

    def backward(self, x, grad_y):
        grad_x = grad_y[0] / (x[0] * log(10))

        return grad_x


def log10(x):
    """Calculate the base-10 logarithm of an input.

        This function computes the base-10 logarithm of the input.

        Args:
            x (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The input values.

        Returns:
            :class:`marquetry.Container`: The base-10 logarithm of the input.

        Examples:
            >>> x = np.array([1, 2, 3])
            >>> log10(x)
            container([0.         0.30103    0.47712125])
    """

    return Log10()(x)
