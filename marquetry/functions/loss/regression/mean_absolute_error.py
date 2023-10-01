from marquetry import cuda_backend
from marquetry import Function


class MeanAbsoluteError(Function):
    """Calculate the Mean Absolute Error (MAE) between two sets of values as loss function.

        This class computes the MAE between two sets of values.
        The MAE is a measure of the average absolute difference between corresponding elements of the input arrays.
        MAE is used as loss function when you want to use a data having outlier.

        Note:
            Generally, you don't need to execute ``forward`` and ``backward`` method manually.
            You should use only ``__call__`` method.

    """
    def forward(self, x0, x1):
        xp = cuda_backend.get_array_module(x0)
        diff = x0 - x1
        y = xp.absolute(diff).sum() / diff.dtype.type(diff.size)

        return y

    def backward(self, inputs, grad_y):
        xp = cuda_backend.get_array_module(inputs[0])
        x0, x1 = inputs

        diff = x0 - x1
        sign_map = xp.asarray(diff >= 0, dtype="f")
        sign_map -= .5
        sign_map *= 2

        coefficient = grad_y[0] * grad_y[0].dtype.type(1. / diff.size)
        grad_x = coefficient * sign_map

        return grad_x, -grad_x


def mean_absolute_error(x0, x1):
    """
        Calculate the Mean Absolute Error (MAE) between two sets of values.

        This function computes the MAE between two sets of values.
        The MAE is a measure of the average absolute　difference between corresponding elements of the input arrays.
        MAE is used as loss function when you want to use a data having outlier.

        Args:
            x0 (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The first set of values.
            x1 (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The second set of values.

        Returns:
            :class:`marquetry.Container`: The Mean Absolute Error (MAE).
    """

    return MeanAbsoluteError()(x0, x1)
