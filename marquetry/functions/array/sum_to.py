from marquetry import Function
from marquetry import functions
from marquetry import utils


class SumTo(Function):
    """SumTo an input array to a target shape.

        Note:
            Generally, you don't need to execute ``forward`` and ``backward`` method manually.
            You should use only ``__call__`` method.
    """

    def __init__(self, shape):
        self.shape = shape

        self.x_shape = None

    def forward(self, x):
        if x.shape == self.shape:
            return x

        self.x_shape = x.shape
        y = utils.sum_to(x, self.shape)

        self.retain_inputs(())
        return y

    def backward(self, x, grad_y):
        if self.x_shape is None:
            return grad_y[0]

        grad_x = functions.broadcast_to(grad_y[0], self.x_shape)

        return grad_x


def sum_to(x, shape):
    """SumTo an input array to a target shape.

        Args:
            x (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The input array to be summed to the target shape.
            shape (tuple of ints):
                The target shape to which the input array should be summed.

        Returns:
            :class:`marquetry.Container`: The result of SumTo the input
                array to the target shape by summing.

        Examples:
            >>> x = np.arange(1, 9).reshape(2, 4)
            array([[1, 2, 3, 4],
                   [5, 6, 7, 8]])
            >>> sum_to(x, (1, 1))
            container([[36]])
    """
    return SumTo(shape)(x)
