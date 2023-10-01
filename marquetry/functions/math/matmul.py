from marquetry import Function


class MatMul(Function):
    """Calculate dot product between two inputs.

        This class calculates dot product between the two input matrices.

        Note:
            Generally, you don't need to execute ``forward`` and ``backward`` method manually.
            You should use only ``__call__`` method.
    """

    def forward(self, x0, x1):
        y = x0.dot(x1)

        return y

    def backward(self, xs, grad_y):
        x0, x1 = xs
        grad_y = grad_y[0]

        grad_x0 = matmul(grad_y, x1.T)
        grad_x1 = matmul(x0.T, grad_y)

        return grad_x0, grad_x1


def matmul(x0, x1):
    """Calculate dot product between two inputs.

        This function calculates dot product between the two input matrices.

        Args:
            x0 (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The first input matrix.
            x1 (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The second input matrix.

        Returns:
            :class:`marquetry.Container`: The result of the dot product.

        Examples:
            >>> x0 = np.arange(0, 20).reshape(5, 4)
            >>> x0
            array([[ 0,  1,  2,  3],
                   [ 4,  5,  6,  7],
                   [ 8,  9, 10, 11],
                   [12, 13, 14, 15],
                   [16, 17, 18, 19]])
            >>> x1 = np.arange(0, 24).reshape(4, 6)
            >>> x1
            array([[ 0,  1,  2,  3,  4,  5],
                   [ 6,  7,  8,  9, 10, 11],
                   [12, 13, 14, 15, 16, 17],
                   [18, 19, 20, 21, 22, 23]])
            >>> matmul(x0, x1)
            container([[  84   90   96  102  108  114]
                       [ 228  250  272  294  316  338]
                       [ 372  410  448  486  524  562]
                       [ 516  570  624  678  732  786]
                       [ 660  730  800  870  940 1010]])
            >>> matmul(x0, x1).shape
            (5, 6)

    """

    return MatMul()(x0, x1)
