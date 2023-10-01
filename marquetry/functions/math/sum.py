from marquetry import Function
from marquetry import functions
from marquetry import utils


class Sum(Function):
    """Calculate the sum of the input tensor along the specified axis or axes.

        Note:
            Generally, you don't need to execute ``forward`` and ``backward`` method manually.
            You should use only ``__call__`` method.
    """

    def __init__(self, axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims

        self.x_shape = None

    def forward(self, x):
        self.x_shape = x.shape
        y = x.sum(axis=self.axis, keepdims=self.keepdims)

        self.retain_inputs(())
        return y

    def backward(self, x, grad_y):
        grad_y = utils.reshape_sum_backward(grad_y[0], self.x_shape, self.axis, self.keepdims)
        grad_x = functions.broadcast_to(grad_y, self.x_shape)
        return grad_x


def sum(x, axis=None, keepdims=False):
    """Calculate the sum of the input tensor along the specified axis or axes.

        Args:
            x (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The input tensor.
            axis (None or int or tuple of ints): The axis or axes along which to sum.
                If None, sum all elements.
            keepdims (bool): If True, the output will have the same number of dimensions as the input,
                with size 1 in the reduced dimensions.
                If False, the output will have reduced dimensions which size is 1 removed.

        Returns:
            :class:`marquetry.Container`: The sum of the input tensor along the specified axis or axes.

        Examples:
                >>> x = np.array([[1, 3, 2], [5, 2, 4]])
                >>> x
                array([[1, 3, 2],
                       [5, 2, 4]])
                >>> sum(x)
                container(17)
                >>> sum(x, axis=1)
                container([ 6 11])
                >>> sum(x, axis=1, keepdims=True)
                container([[ 6]
                           [11]])
    """

    return Sum(axis, keepdims)(x)
