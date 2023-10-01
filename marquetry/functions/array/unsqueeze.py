from marquetry import Function
from marquetry import functions


class UnSqueeze(Function):
    """Add a single-dimensional entry (axis with size 1) belong to the specify axis to the shape of an input array.

        Note:
            Generally, you don't need to execute ``forward`` and ``backward`` method manually.
            You should use only ``__call__`` method.
    """

    def __init__(self, axis):
        if isinstance(axis, int):
            axis = (axis,)
        elif isinstance(axis, (tuple, list)) and all(isinstance(x, int) for x in axis):
            axis = tuple(axis)
        else:
            raise ValueError("axis expected int or sequence of ints, but got invalid type data.")

        self.axis = axis

    def forward(self, x):
        x_shape = x.shape

        new_shape = list(x_shape)

        for index in sorted(self.axis):
            new_shape.insert(index, 1)
        new_shape = tuple(new_shape)

        y = x.reshape(new_shape)

        self.retain_inputs(())
        return y

    def backward(self, x, grad_y):
        grad_x = functions.squeeze(grad_y[0], self.axis)

        return grad_x


def unsqueeze(x, axis):
    """Add a single-dimensional entry (axis with size 1) belong to the specify axis to the shape of an input array.

        Args:
            x (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The input array to which a single-dimensional entry should be added along the specified axis.
            axis (int or tuple of ints or list of ints): The axis along which to add the single-dimensional entry.

        Returns:
            :class:`marquetry.Container`:
                The result of adding a single-dimensional entry to the input array along the specified axis.

        Examples:
            >>> x = np.arange(1, 9).reshape(1, 8)
            array([[1, 2, 3, 4, 5, 6, 7, 8]])
            >>> unsqueeze(x, axis=0)
            container([[[1 2 3 4 5 6 7 8]]])
            >>> unsqueeze(x, axis=0).shape
            (1, 1, 8)
        """

    return UnSqueeze(axis)(x)
