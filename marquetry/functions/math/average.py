from marquetry import as_container
from marquetry import functions
from marquetry import Function


class Average(Function):
    """Calculate the average value of an input.

        This class computes the average value of the input along the specified axis.
        If the axis is None, this calculates average value overall input values.

        Note:
            Generally, you don't need to execute ``forward`` and ``backward`` method manually.
            You should use only ``__call__`` method.
    """
    def __init__(self, axis=None, keepdims=False):
        if axis is None:
            self.axis = None
        elif isinstance(axis, int):
            self.axis = (axis,)
        elif isinstance(axis, tuple) and all(isinstance(axis_num, int) for axis_num in axis):
            if len(set(axis)) != len(axis):
                raise ValueError("your input axis has duplicated values: ({})"
                                 .format(", ".join(map(str, axis))))
            self.axis = axis
        else:
            if not isinstance(axis, tuple):
                type_info_print = type(axis).__name__
            else:
                type_info = [str(type(axis_num).__name__) for axis_num in axis]
                type_info_print = "(" + ", ".join(type_info) + ")" if len(type_info) != 1 \
                    else "(" + ", ".join(type_info) + ",)"
            raise TypeError("Valid axis are None, int or tuple of int, but got {}"
                            .format(type_info_print))

        self.keepdims = keepdims

        self.multiplier = None
        self.ndim = None
        self.shape = None

    def forward(self, x):
        self.ndim = x.ndim
        self.shape = x.shape

        if self.axis is None:
            self.multiplier = 1. / x.size

        else:
            divider = 1
            for axis in self.axis:
                divider *= x.shape[axis]
            self.multiplier = 1. / divider

        y = x.mean(axis=self.axis, keepdims=self.keepdims)

        self.retain_inputs(())
        return y

    def backward(self, inputs, grad_y):
        grad_y, = grad_y
        grad_y *= self.multiplier

        if not (self.ndim == 0 or self.axis is None or self.keepdims):
            actual_axis = [
                axis if axis >= 0 else axis + self.ndim
                for axis in self.axis
            ]
            shape = list(grad_y.shape)

            for axis in sorted(actual_axis):
                shape.insert(axis, 1)

            grad_y = grad_y.reshape(shape)

        grad_x = functions.broadcast_to(grad_y, self.shape)

        return grad_x


def average(x, axis=None, keepdims=False):
    """Calculate the average of an input array along the specified axis or axes.

        This function computes the average value of the input along the specified axis.
        If the axis is None, this calculates average value overall input values.

        Args:
            x (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The input values.
            axis (int or tuple of ints or None):
                The axis or axes along which the average is computed.
                If None (default), it computes the average of all elements.
                If int, it specifies the axis along which the average is computed.
                If tuple of ints, it specifies multiple axes along which the average is computed.
            keepdims (bool): If True, the output has the same dimensions as the input,
                but the reduced axes are retained with size 1.
                If False (default), the reduced axes are removed from the output.

        Returns:
            :class:`marquetry.Container`: The average of the input values computed along the specified axis or axes.

        Note:
            `axis` and `keepdims` follow NumPy conventions.

        Examples:
            >>> x = np.array([[2, 4, 6], [1, 2, 3]])
            >>> x
            array([[2, 4, 6],
                   [1, 2, 3]])
            >>> average(x)
            container(3.0)
            >>> average(x, axis=0)
            container([1.5 3.  4.5])
            >>> average(x, axis=1, keepdims=True)
            container([[4.]
                       [2.]])
    """

    return Average(axis, keepdims)(x)


def simple_average(x, axis=None, keepdims=False):
    x = as_container(x)
    y = functions.sum(x, axis, keepdims)

    return y * (y.data.size / x.data.size)


mean = average
