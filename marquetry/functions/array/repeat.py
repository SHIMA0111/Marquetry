from marquetry import cuda_backend
from marquetry import Function


class Repeat(Function):
    """Copy and expand the array to the specify number and axis.

        Note:
            Generally, you don't need to execute ``forward`` and ``backward`` method manually.
            You should use only ``__call__`` method.
    """

    def __init__(self, repeats, axis):
        if isinstance(repeats, int):
            self.repeats = (repeats,)
        elif isinstance(repeats, (tuple, list)) and all(isinstance(x, int) for x in repeats):
            self.repeats = tuple(repeats)
        else:
            raise TypeError("repeats must be int or tuple if ints, but got {}".format(repeats))

        if not all(x >= 0 for x in self.repeats):
            raise ValueError("all repeats elements must be larger than 0.")

        if axis is not None and not isinstance(axis, int) or isinstance(axis, bool):
            raise TypeError("axis must be int or None.")

        self.axis = axis

    def forward(self, x):
        xp = cuda_backend.get_array_module(x)
        repeats = self.repeats

        y = xp.repeat(x, repeats, self.axis)
        return y

    def backward(self, x, grad_y):
        x, = x
        grad_y, = grad_y

        grad_x = RepeatGrad(self.repeats, self.axis, x.shape, x.dtype)(grad_y)
        return grad_x


class RepeatGrad(Function):
    def __init__(self, repeats, axis, in_shape, in_dtype):
        self.repeats = repeats
        self.axis = axis
        if axis is not None and axis < 0:
            self.axis += len(in_shape)

        self.in_shape = in_shape
        self.in_dtype = in_dtype

    def forward(self, grad_y):
        xp = cuda_backend.get_array_module(grad_y)

        repeats = self.repeats
        axis = self.axis
        shape = list(self.in_shape)
        dtype = self.in_dtype

        if len(grad_y) == 0:
            grad_x = xp.zeros(shape, dtype)
            return grad_x

        if len(repeats) == 1:
            repeats = int(repeats[0])
            if axis is None:
                grad_x = grad_y.reshape(-1, repeats).sum(axis=1).reshape(shape)
            else:
                shape[axis:axis + 1] = [-1, repeats]
                grad_x = grad_y.reshape(shape).sum(axis=axis + 1)

            return grad_x

        if axis is None:
            position = 0
            grad_x = xp.zeros(shape, dtype).reshape(-1)

            for i, repeat_num in enumerate(repeats):
                grad_x[i] = xp.sum(grad_y[position:position + repeat_num])
                position += repeat_num

            grad_x = grad_x.reshape(shape)

        else:
            position = 0
            grad_x = xp.zeros(shape, dtype)
            source = [slice(None)] * axis + [None]
            distance = [slice(None)] * axis + [None]

            for i, repeat_num in enumerate(repeats):
                source[-1] = slice(position, position + repeat_num)
                distance[-1] = slice(i, i + 1)
                grad_x[tuple(distance)] = grad_y[tuple(source)].sum(axis=axis, keepdims=True)
                position += repeat_num

        self.retain_inputs(())
        return grad_x

    def backward(self, x, grad_grad_y):
        grad_grad_x = repeat(grad_grad_y[0], self.repeats, self.axis)

        return grad_grad_x


def repeat(x, repeats, axis):
    """Repeat and expand an input array along a specified axis.

        Args:
            x (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The input array to be repeated and expanded.
            repeats (int): The number of times to repeat the input along the specified axis.
            axis (int): The axis along which the input should be repeated and expanded.

        Returns:
            A :class:`marquetry.Container` object which is the result of repeating and
            expanding the input array along the specified axis.

        Examples:
            >>> x = np.arange(0, 3).reshape(1, 3)
            array([[0, 1, 2]])
            >>> repeat(x, repeats=3, axis=0)
            container([[0 1 2]
                       [0 1 2]
                       [0 1 2]])
    """
    return Repeat(repeats, axis)(x)
