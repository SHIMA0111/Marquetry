from marquetry import functions


class Min(functions.Max):
    """Calculate the minimum along the specified axis.

        Note:
            Generally, you don't need to execute ``forward`` and ``backward`` method manually.
            You should use only ``__call__`` method.
    """

    def forward(self, x):
        y = x.min(axis=self.axis, keepdims=self.keepdims)

        self.retain_outputs((0,))
        return y


def min(x, axis=None, keepdims=False):
    """Calculate the minimum along the specified axis.

            Args:
                x (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                    The input tensor.
                axis (int or tuple of ints or None): The axis or axes along which to find the minimum.
                keepdims (bool): If True, the output has the same number of dimensions as the input,
                    otherwise size 1's dimension is reduced.

            Returns:
                :class:`marquetry.Container`: The minimum value along the specified axis.

            Examples:
                >>> x = np.array([[1, 3, 2], [5, 2, 4]])
                >>> x
                array([[1, 3, 2],
                       [5, 2, 4]])
                >>> min(x)
                container(1)
                >>> min(x, axis=1)
                container([1 2])
                >>> min(x, axis=1, keepdims=True)
                container([[1]
                           [2]])

        """

    return Min(axis, keepdims)(x)
