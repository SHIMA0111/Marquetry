from marquetry import cuda_backend
from marquetry import Function


class MeanSquaredError(Function):
    """Calculate the Mean Squared Error (MSE) between predicted values and true values.

        This class defines that calculates the Mean Squared Error (MSE) between predicted values and true values.
        MSE measures the average squared difference between the predicted and true values.
        It can be calculated either as a uniform average over all samples or as raw values for each sample.

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

        mse_value = xp.mean(xp.square(y - t), axis=0)

        self.retain_inputs(())
        if self.multi_output == "uniform_average":
            return xp.asarray(mse_value.mean(), dtype=y.dtype)
        else:
            return xp.asarray(mse_value, dtype=y.dtype)


def mean_squared_error(y, t, multi_output="uniform_average"):
    """Calculate the Mean Squared Error (MSE) between predicted values and true values.

        This function defines that calculates the Mean Squared Error (MSE) between predicted values and true values.
        MSE measures the average squared difference between the predicted and true values.
        It can be calculated either as a uniform average over all samples or as raw values for each sample.

        Args:
            y (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The predicted values.
            t (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The true values.
            multi_output (str): Specifies how to calculate the MSE for multi-output.

        Note:
            multi_output:
                "uniform_average": Compute the uniform average MSE over all samples. This is the default option.

                "raw_values": Return the raw MSE values for each sample.

        Returns:
            :class:`marquetry.Container`: The MSE values based on the predicted values
                and true values.
    """
    return MeanSquaredError(multi_output)(y, t)


mse = mean_squared_error
