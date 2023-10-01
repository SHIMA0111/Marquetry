from marquetry import cuda_backend
from marquetry import Function
from marquetry import functions


class Concat(Function):
    """Concatenate multiple tensors towards specified axis.

        Note:
            Generally, you don't need to execute ``forward`` and ``backward`` method manually.
            You should use only ``__call__`` method.
    """
    def __init__(self, axis):
        self.axis = axis

    def forward(self, *inputs):
        if len(inputs) == 1 and isinstance(inputs[0], (tuple, list)):
            inputs = inputs[0]

        xp = cuda_backend.get_array_module(inputs[0])
        y = xp.concatenate(inputs, axis=self.axis)

        return y

    def backward(self, inputs, grad_y):
        pre_index = 0
        indices = []
        for i, data in enumerate(inputs):
            if i == len(inputs) - 1:
                continue
            index = data.shape[self.axis]
            pre_index += index
            indices.append(pre_index)

        grad_x = functions.split(grad_y[0], indices, axis=self.axis)

        return grad_x


def concat(*inputs, axis=0):
    """Concatenates given containers to specify an axis.

        Args:
            inputs (tuple or enumeration of :class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                Input containers to be concatenated. The containers must have the
                same shape, except in the dimension corresponding to the concat axis.
            axis (int): The axis along which the arrays will be concatenated. Default is 0.

        Returns:
            marquetry.Container: The concatenated container.

        Examples:

            >>> x = np.arange(0, 12).reshape(3, 4)
            >>> x
            array([[ 0,  1,  2,  3],
                   [ 4,  5,  6,  7],
                   [ 8,  9, 10, 11]])
            >>> y = np.arange(0, 3).reshape(3, 1)
            >>> y
            array([[0],
                   [1],
                   [2]])
            >>> concat(x, y, axis=1)
            container([[ 0  1  2  3  0]
                       [ 4  5  6  7  1]
                       [ 8  9 10 11  2]])

        """

    if len(inputs) == 1 and isinstance(inputs[0], (tuple, list)):
        inputs = tuple(inputs[0])

    return Concat(axis)(*inputs)

