import numpy as np

from marquetry import cuda_backend
from marquetry import Function


class GetItem(Function):
    """The slices array and extruct element from array.

        Note:
            Generally, you don't need to execute ``forward`` and ``backward`` method manually.
            You should use only ``__call__`` method.
    """
    def __init__(self, slices):
        self.slices = slices

    def forward(self, x):
        y = x[self.slices]

        return y

    def backward(self, x, grad_y):
        f = GetItemGrad(self.slices, x[0].shape)
        return f(grad_y[0])


class GetItemGrad(Function):
    def __init__(self, slices, in_shape):
        self.slices = slices
        self.in_shape = in_shape

    def forward(self, grad_y):
        xp = cuda_backend.get_array_module(grad_y)
        grad_x = xp.zeros(self.in_shape, dtype=grad_y.dtype)

        if xp is np:
            np.add.at(grad_x, self.slices, grad_y)
        else:
            grad_x.scatter_add(self.slices, grad_y)

        self.retain_inputs(())
        return grad_x

    def backward(self, x, grad_grad_y):
        return get_item(grad_grad_y[0], self.slices)


def get_item(x, slices):
    """Extract specified index array from original array.

    Args:
        x (marquetry.Container): A container to be sliced.
        slices (int, slice, Ellipsis, None, integer array-like, boolean, array-like or tuple of them):
            It is an integer, a slice, an ellipsis, a numpy.newaxis, an integer array-like, a boolean array-like
            or tuple of them.

    Returns:
        A :class:`marquetry.Container` object which contains sliced array of ``x``.
    """

    f = GetItem(slices)
    return f(x)
