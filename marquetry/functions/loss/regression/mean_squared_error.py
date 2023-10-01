from marquetry import Function


class MeanSquaredError(Function):
    """Calculate the Mean Squared Error (MSE) between two input arrays.

        This class computes the Mean Squared Error (MSE) between two input arrays `x0` and `x1`.
        The MSE is a measure of the average squared difference between corresponding elements  of `x0` and `x1`.
        It is often used as a loss function in regression problems.

        Note:
            Generally, you don't need to execute ``forward`` and ``backward`` method manually.
            You should use only ``__call__`` method.
    """
    def forward(self, x0, x1):
        diff = x0 - x1
        y = (diff ** 2).sum() / diff.dtype.type(diff.size)

        return y

    def backward(self, inputs, grad_y):
        x0, x1 = inputs
        diff = x0 - x1
        grad_x0 = 2. * diff / diff.size * grad_y[0]
        grad_x1 = -grad_x0 * grad_y[0]

        return grad_x0, grad_x1


def mean_squared_error(x0, x1):
    """Calculate the Mean Squared Error (MSE) between two input arrays.

        This function computes the Mean Squared Error (MSE) between two input arrays `x0` and `x1`.
        The MSE is a measure of the average squared difference between corresponding elements of `x0` and `x1`.
        It is often used as a loss function in regression problems.

        Parameters:
            x0 (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The first input array.
            x1 (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The second input array.

        Returns:
            :class:`marquetry.Container`: The Mean Squared Error (MSE).
    """

    return MeanSquaredError()(x0, x1)
