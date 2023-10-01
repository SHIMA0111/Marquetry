import numpy as np

import marquetry
from marquetry import Function


class Absolute(Function):
    """Calculate the absolute value of an input.

        This class computes the absolute value of the input.

        Note:
            Generally, you don't need to execute ``forward`` and ``backward`` method manually.
            You should use only ``__call__`` method.
    """

    def forward(self, x):
        y = np.abs(x)

        return y

    def backward(self, x, grad_y):
        mask = (x[0] >= 0).astype("f")
        mask -= .5
        mask *= 2.

        grad_y = grad_y[0]
        if mask.shape != grad_y.shape:
            marquetry.functions.broadcast_to(grad_y, x[0].shape)

        grad_x = grad_y * mask

        return grad_x


def absolute(x):
    """Calculate the absolute value of an input.

        This function computes the absolute value of the input.

        Args:
            x (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The input values.

        Returns:
            :class:`marquetry.Container`: The absolute value of the input.

        Examples:
            >>> x = np.array([[-2, 4, 0], [-1, 2, -3]])
            >>> x
            array([[-2, 4, 0],
                   [-1, 2, -3]])
            >>> absolute(x)
            container([[2 4 6]
                       [1 2 3]])
    """

    return Absolute()(x)
