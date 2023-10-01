import numpy as np

from marquetry import cuda_backend
from marquetry import Function
from marquetry import functions


class Split(Function):
    """Split an input array or container into multiple parts along the specified axis and indices.

        Note:
            Generally, you don't need to execute ``forward`` and ``backward`` method manually.
            You should use only ``__call__`` method.
    """

    def __init__(self, indices, axis):
        self.axis = axis
        if np.isscalar(indices):
            indices = (indices,)
        self.indices = indices

    def forward(self, x):
        xp = cuda_backend.get_array_module(x)
        y = xp.split(x, self.indices, axis=self.axis)

        self.retain_inputs(())
        return tuple(y)

    def backward(self, x, grad_ys):
        grad_x = functions.concat(grad_ys, axis=self.axis)

        return grad_x


def split(x, indices, axis):
    """Split an input array or container into multiple parts along the specified axis and indices.

        Args:
            x (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The input array to be split.
            indices (int or tuple of ints):
                The indices at which to split the input array or container along the specified axis.
            axis (int): The axis along which the input array or container should be split.

        Returns:
            list of :class:`marquetry.Container` a tuple containing the result of
            splitting the input array into multiple parts along the specified axis.

        Examples:
            >>> x = np.arange(1, 9).reshape(2, 4)
            array([[1, 2, 3, 4],
                   [5, 6, 7, 8]])
            >>> split(x, indices=(2, 3), axis=1)
            [container([[1 2]
                        [5 6]]),
             container([[3]
                        [7]]),
             container([[4]
                        [8]])]
    """

    return Split(indices, axis)(x)
