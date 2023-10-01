from marquetry import Function
from marquetry import functions


class Linear(Function):
    """Perform a linear transformation on an input array.

        This class defines a function that performs a linear transformation on an input array using
        the provided weight matrix and optional bias.

        Note:
            Generally, you don't need to execute ``forward`` and ``backward`` method manually.
            You should use only ``__call__`` method.
    """
    def forward(self, x, w, b):
        y = x.dot(w)
        if b is not None:
            y += b
        return y

    def backward(self, inputs, grad_y):
        x, w, b = inputs
        grad_y = grad_y[0]

        grad_b = None if b is None else functions.sum_to(grad_y, b.shape)

        grad_x = functions.matmul(grad_y, w.T)
        grad_w = functions.matmul(x.T, grad_y)

        return grad_x, grad_w, grad_b


def linear(x, w, b=None):
    """Perform a linear transformation on an input array or container.


        This class defines a function that performs a linear transformation on an input array using
        the provided weight matrix and optional bias.

        Args:
            x (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The input array to be linearly transformed.
            w (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The weight matrix for the linear transformation.
            b (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray` or None):
                The bias term. If None, no bias is added.

        Returns:
            :class:`marquetry.Container`: The result of the linear transformation on
                the input array using the provided weight matrix and bias (if any).

        Examples:
            >>> x = np.arange(0, 20).reshape(4, 5)
            >>> w = np.arange(0, 30).reshape(5, 6)
            >>> b = np.arange(0, 6)
            >>> linear(x, w, b)
            container([[ 180  191  202  213  224  235]
                       [ 480  516  552  588  624  660]
                       [ 780  841  902  963 1024 1085]
                       [1080 1166 1252 1338 1424 1510]])
            >>> linear(x, w, b).shape
            (4, 6)
    """
    return Linear()(x, w, b)
