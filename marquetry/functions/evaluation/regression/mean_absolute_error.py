from marquetry import cuda_backend
from marquetry import Function


class MeanAbsoluteError(Function):
    """Calculate the Mean Absolute Error (MAE) between predicted values and true values.

        This class defines that calculates the Mean Absolute Error (MAE) between predicted values and true Values.
        MAE measures the average absolute difference between the predicted and true values.
        It can be calculated either as a uniform average over all samples or as raw values for each column.

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

        mae_value = xp.mean(xp.absolute(y - t), axis=0)

        self.retain_inputs(())

        if self.multi_output == "uniform_average":
            return xp.asarray(mae_value.mean(), dtype=y.dtype)
        else:
            return xp.asarray(mae_value, dtype=y.dtype)


def mean_absolute_error(y, t, multi_output="uniform_average"):
    """Calculate the Mean Absolute Error (MAE) between predicted values and true values.

        This function defines that calculates the Mean Absolute Error (MAE) between predicted values and true Values.
        MAE measures the average absolute difference between the predicted and true values.
        It can be calculated either as a uniform average over all samples or as raw values for each column.

        Args:
            y (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The predicted values.
            t (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The true values.
            multi_output (str): Specifies how to calculate the MAE for multi-output.

        Note:
            multi_output:
                "uniform_average": Compute the uniform average MAE over all samples. This is the default option.

                "raw_values": Return the raw MAE values for each sample.

        Returns:
            :class:`marquetry.Container`: The MAE values based on the predicted values
                and true values.
    """
    return MeanAbsoluteError(multi_output)(y, t)


mae = mean_absolute_error
