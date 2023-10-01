from marquetry import cuda_backend
from marquetry import Function
from marquetry import functions


class Squeeze(Function):
    """Remove single-dimensional entries from the shape of an input array.

        Note:
            Generally, you don't need to execute ``forward`` and ``backward`` method manually.
            You should use only ``__call__`` method.
    """
    def __init__(self, axis):
        if isinstance(axis, int):
            axis = (axis,)
        elif isinstance(axis, (tuple, list)) and all(isinstance(x, int) for x in axis):
            axis = tuple(axis)
        elif axis is None:
            axis = axis
        else:
            raise ValueError("axis expected int or sequence of ints, but got invalid type data.")

        self.axis = axis

    def forward(self, x):
        xp = cuda_backend.get_array_module(x)

        if self.axis is not None:
            for axis in self.axis:
                if x.shape[axis] != 1:
                    raise ValueError("You can't squeeze non-one size axis element, {}-axis is {}."
                                     .format(axis, x.shape[axis]))

        y = xp.squeeze(x, axis=self.axis)

        return y

    def backward(self, x, grad_y):
        axis = self.axis
        if axis is None:
            tmp_axis = []
            for i, axis_num in enumerate(x[0].shape):
                if axis_num == 1:
                    tmp_axis.append(i)

            axis = tuple(tmp_axis)

        grad_x = functions.unsqueeze(grad_y[0], axis)

        return grad_x


def squeeze(x, axis):
    """Remove single-dimensional entries from the shape of an input array.

        Args:
            x (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The input array  from which single-dimensional entries should be removed.
            axis (int or tuple of ints or list of ints): The axis along which single-dimensional entries should be removed.

        Returns:
            :class:`marquetry.Container`: The result of removing single-dimensional
            entries from the input array or container along the specified axis (or all axes if axis is None).

        Raises:
            ValueError: If the specified axis does not have a size of 1,
                indicating that there are no single-dimensional entries to remove.

        Examples:
            >>> x = np.arange(1, 9).reshape(1, 8)
            array([[1, 2, 3, 4, 5, 6, 7, 8]])
            >>> squeeze(x, axis=0)
            container([1 2 3 4 5 6 7 8])

    """

    return Squeeze(axis)(x)
