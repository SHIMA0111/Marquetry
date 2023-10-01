from marquetry import cuda_backend
from marquetry import Function


class Clip(Function):
    """Clip the values of an array within a specified range.

        This class clips the values of the input array `x`
        so that they fall within the specified range [`x_min`, `x_max`].
        Values less than `x_min` are set to `x_min`,
        and values greater than `x_max` are set to `x_max`.

        Note:
            Generally, you don't need to execute ``forward`` and ``backward`` method manually.
            You should use only ``__call__`` method.
    """

    def __init__(self, x_min, x_max):
        self.x_min = x_min
        self.x_max = x_max

    def forward(self, x):
        xp = cuda_backend.get_array_module(x)

        y = xp.clip(x, self.x_min, self.x_max)

        return y

    def backward(self, x, grad_y):
        x = x[0]

        mask = (x >= self.x_min) * (x <= self.x_max)
        grad_x = grad_y[0] * mask

        return grad_x


def clip(x, x_min, x_max):
    """Clip the values of an array within a specified range.

        This function clips the values of the input array `x`
        so that they fall within the specified range [`x_min`, `x_max`]. \n
        Values less than `x_min` are set to `x_min`,
        and values greater than `x_max` are set to `x_max`.

        Args:
            x (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The input array or container.
            x_min (float or int): The minimum value for clipping.
            x_max (float or int): The maximum value for clipping.

        Returns:
            :class:`marquetry.Container`: The result of clipping, with values in the range [`x_min`, `x_max`].

        Examples:
            >>> x = np.array([[-0.2, 10.9, 3.2], [-1.2,  4.2, 0]])
            >>> x
            array([[-0.2, 10.9,  3.2],
                   [-1.2,  4.2,  0. ]])
            >>> clip(x, 0.1, 5.)
            container([[0.1 5.  3.2]
                       [0.1 4.2 0.1]])
            >>> clip(x, -1., 9.)
            container([[-0.2  9.   3.2]
                       [-1.   4.2  0. ]])
    """

    return Clip(x_min, x_max)(x)
