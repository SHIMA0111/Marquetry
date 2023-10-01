from marquetry import cuda_backend
from marquetry import Function


class RelativeAbsoluteError(Function):
    """Calculate the Relative Absolute Error (RAE) between predicted values and true values.

        This class defines a function that calculates the Relative Absolute Error (RAE),
        a measure of the relative difference between predicted values and true values.
        RAE quantifies the average absolute difference between predicted and true
        values relative to the average absolute deviation of true values from their mean.
        It is used to assess the accuracy of regression models.

        Note:
            Generally, you don't need to execute ``forward`` and ``backward`` method manually.
            You should use only ``__call__`` method.
    """

    def __init__(self, multi_output):
        if multi_output in ["uniform_average", "raw_values"]:
            self.multi_output = multi_output
        else:
            raise ValueError("invalid multi_output argument")

    def forward(self, y, t):
        assert y.size == t.size

        xp = cuda_backend.get_array_module(y)

        if t.ndim == 1:
            t = t.reshape((-1, 1))

        y = y.reshape(t.shape)

        correct_average = xp.mean(t, axis=0)

        absolute_error = xp.sum(xp.absolute(y - t), axis=0)
        absolute_deviation = xp.sum(xp.absolute(t - correct_average), axis=0)

        relative_absolute_error_value = xp.where(absolute_deviation != 0,
                                                 absolute_error / absolute_deviation,
                                                 1.0)
        if self.multi_output == "uniform_average":
            return xp.asarray(relative_absolute_error_value.mean(), dtype=y.dtype)
        else:
            return xp.asarray(relative_absolute_error_value, dtype=y.dtype)


def relative_absolute_error(y, t, multi_output="uniform_average"):
    """Calculate the Relative Absolute Error (RAE) between predicted values and true values.

        This function defines that calculates the Relative Absolute Error (RAE),
        a measure of the relative difference between predicted values and true values.
        RAE quantifies the average absolute difference between predicted and true
        values relative to the average absolute deviation of true values from their mean.
        It is used to assess the accuracy of regression models.

        Args:
            y (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The predicted values.
            t (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The true values.
            multi_output (str): Specifies how to calculate the RAE for multi-output.

        Note:
            multi_output:
                "uniform_average": Compute the uniform average RAE over all samples. This is the default option.

                "raw_values": Return the raw RAE values for each sample.

        Returns:
            :class:`marquetry.Container`: The RAE based on the predicted values
                and true values. Lower values indicate better accuracy.
    """

    return RelativeAbsoluteError(multi_output)(y, t)
