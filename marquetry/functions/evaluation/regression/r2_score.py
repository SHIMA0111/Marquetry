from marquetry import cuda_backend
from marquetry import Function


class R2Score(Function):
    """Calculate the R-squared (R2) score, which measures the goodness of fit of a model to the true values.

        This class defines a function that calculates the R-squared (R2) score,
        a statistical measure that indicates the goodness of fit of a model to the true values.
        R2 score measures the proportion of the variance in the dependent container that is predictable
        from the independent containers. It is a value between 0 and 1, where higher values indicate a better fit.

        Note:
            Generally, you don't need to execute ``forward`` and ``backward`` method manually.
            You should use only ``__call__`` method.
    """

    def __init__(self, multi_output):
        # TODO: implement sample weight to follow scikit-learn implementation
        if multi_output in ["uniform_average", "raw_values"]:
            self.multi_output = multi_output
        else:
            raise ValueError("invalid multi_output argument")

    def forward(self, y, t):
        xp = cuda_backend.get_array_module(y)
        if y.size != t.size:
            raise ValueError("target shape is {} but predict size is (), these aren't match."
                             .format(t.shape, y.shape))

        if t.ndim == 1:
            t = t.reshape((-1, 1))

        y = y.reshape(t.shape)

        target_sum_squared_deviations = xp.sum((t - xp.mean(t, axis=0)) ** 2, axis=0)
        pred_target_squared_deviations = xp.sum((y - t) ** 2, axis=0)

        r2_score_value = xp.where(target_sum_squared_deviations != 0,
                                  1 - pred_target_squared_deviations / target_sum_squared_deviations,
                                  0.0)

        self.retain_inputs(())
        if self.multi_output == "uniform_average":
            return xp.asarray(r2_score_value.mean(), dtype=y.dtype)
        else:
            return xp.asarray(r2_score_value, dtype=y.dtype)


def r2_score(y, t, multi_output="uniform_average"):
    """Calculate the R-squared (R2) score, which measures the goodness of fit of a model to the true values.

        This function defines that calculates the R-squared (R2) score,
        a statistical measure that indicates the goodness of fit of a model to the true values.
        R2 score measures the proportion of the variance in the dependent container that is predictable
        from the independent containers. It is a value between 0 and 1, where higher values indicate a better fit.

        R2-score is calculated as:
            :math:`R2 = 1 - {(y(pred) - t)^2 / (t - t(ave))^2}`

        Args:
            y (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The predicted values.
            t (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The true values.
            multi_output (str): Specifies how to calculate the R2 score for multi-output.

        Note:
            multi_output:
                "uniform_average": Compute the uniform average R2 score over all samples. This is the default.

                "raw_values": Return the raw R2 score values for each sample.

        Returns:
            :class:`marquetry.Container`: The R2 score based on the predicted values
                and true values. Higher values indicate a better fit.
        """
    return R2Score(multi_output)(y, t)


class AdjustR2Score(Function):
    def __init__(self):
        raise NotImplementedError()
