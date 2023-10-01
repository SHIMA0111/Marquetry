from marquetry import cuda_backend
from marquetry import Function
from marquetry import functions
from marquetry import utils


class Max(Function):
    """Calculate the maximum along the specified axis.

        Note:
            Generally, you don't need to execute ``forward`` and ``backward`` method manually.
            You should use only ``__call__`` method.
    """
    def __init__(self, axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        y = x.max(axis=self.axis, keepdims=self.keepdims)

        self.retain_outputs((0,))
        return y

    def backward(self, x, grad_y):
        x = x[0]
        y = self.output_data[0]
        grad_y = grad_y[0]

        xp = cuda_backend.get_array_module(x)

        shape = utils.max_backward_shape(x, self.axis)
        grad_y = functions.reshape(grad_y, shape)
        y = functions.reshape(y, shape)
        cond = xp.array(x == y.data)
        grad_y = functions.broadcast_to(grad_y, cond.shape)

        return grad_y * cond


def max(x, axis=None, keepdims=False):
    """Calculate the maximum along the specified axis.

        Args:
            x (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The input tensor.
            axis (int or tuple of ints or None): The axis or axes along which to find the maximum.
            keepdims (bool): If True, the output has the same number of dimensions as the input,
                otherwise size 1's dimension is reduced.

        Returns:
            :class:`marquetry.Container`: The maximum value along the specified axis.

        Examples:
            >>> x = np.array([[1, 3, 2], [5, 2, 4]])
            >>> x
            array([[1, 3, 2],
                   [5, 2, 4]])
            >>> max(x)
            container(5)
            >>> max(x, axis=1)
            container([3 5])
            >>> max(x, axis=1, keepdims=True)
            container([[3]
                       [5]])

    """
    return Max(axis, keepdims)(x)
