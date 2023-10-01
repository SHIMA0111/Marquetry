from marquetry import cuda_backend
from marquetry import Function


class RootMeanSquaredError(Function):
    """Calculate the Root Mean Squared Error (RMSE) between predicted values and true values.

        This class defines a function that calculates the Root Mean Squared Error (RMSE),
        which is a measure of the average squared difference between predicted and true values.
        RMSE is widely used to assess the accuracy of regression models.

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

        mean_squared_error = xp.square(y - t).mean(axis=0)
        root_mean_squared_error_value = xp.sqrt(mean_squared_error)

        self.retain_inputs(())

        if self.multi_output == "uniform_average":
            return xp.asarray(root_mean_squared_error_value.mean(), dtype=y.dtype)
        else:
            return xp.asarray(root_mean_squared_error_value, dtype=y.dtype)


def root_mean_squared_error(y, t, multi_output="uniform_average"):
    """Calculate the Root Mean Squared Error (RMSE) between predicted values and true values.

        This function defines that calculates the Root Mean Squared Error (RMSE),
        which is a measure of the average squared difference between predicted and true values.
        RMSE is widely used to assess the accuracy of regression models.

        Args:
            y (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The predicted values.
            t (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The true values.
            multi_output (str): Specifies how to calculate the RMSE for multi-output.

        Note:
            multi_output:
                "uniform_average": Compute the uniform average RMSE over all samples. This is the default option.

                "raw_values": Return the raw RMSE values for each sample.

        Returns:
            :class:`marquetry.Container`: The RMSE based on the predicted values
                and true values. Lower values indicate better accuracy.
    """

    return RootMeanSquaredError(multi_output)(y, t)


rmse = root_mean_squared_error
