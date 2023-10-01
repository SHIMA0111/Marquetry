from marquetry import cuda_backend
from marquetry import Function


class Transpose(Function):
    """Transpose the dimensions of an input array or container.

        Note:
            Generally, you don't need to execute ``forward`` and ``backward`` method manually.
            You should use only ``__call__`` method.
    """

    def __init__(self, axes=None):
        self.axes = axes

    def forward(self, x):
        y = x.transpose(self.axes)
        self.retain_inputs(())
        return y

    def backward(self, x, grad_y):
        if self.axes is None:
            return transpose(grad_y[0])

        xp = cuda_backend.get_array_module(grad_y[0])

        axes_len = len(self.axes)
        inv_axes = tuple(xp.argsort([ax % axes_len for ax in self.axes]))
        return transpose(grad_y[0], inv_axes)


def transpose(x, axes=None):
    """Transpose the dimensions of an input array or container.

        Args:
            x (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The input array to be transposed.
            axes (tuple of ints or None): The permutation of axes to transpose the input array or container.
                If None, the dimensions are reversed.

        Returns:
            :class:`marquetry.Container`: The result of transposing the input array according to the specified axes.

        Examples:
            >>> x = np.arange(1, 25).reshape(2, 3, 4)
            >>> x
            array([[[ 1,  2,  3,  4],
                    [ 5,  6,  7,  8],
                    [ 9, 10, 11, 12]],
                   [[13, 14, 15, 16],
                    [17, 18, 19, 20],
                    [21, 22, 23, 24]]])
            >>> x.shape
            (2, 3, 4)
            >>> transpose(x)
            container([[[ 1 13]
                        [ 5 17]
                        [ 9 21]]
                       [[ 2 14]
                        [ 6 18]
                        [10 22]]
                       [[ 3 15]
                        [ 7 19]
                        [11 23]]
                       [[ 4 16]
                        [ 8 20]
                        [12 24]]])
            >>> transpose(x).shape
            (4, 3, 2)
            >>> transpose(x, (0, 2, 1))
            container([[[ 1  5  9]
                        [ 2  6 10]
                        [ 3  7 11]
                        [ 4  8 12]]
                       [[13 17 21]
                        [14 18 22]
                        [15 19 23]
                        [16 20 24]]])
            >>> transpose(x, (0, 2, 1)).shape
            (2, 4, 3)
    """
    return Transpose(axes)(x)
