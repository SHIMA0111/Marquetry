from marquetry import cuda_backend
from marquetry import Function


class R2Score(Function):
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
    return R2Score(multi_output)(y, t)


class AdjustR2Score(Function):
    def __init__(self):
        raise NotImplementedError()
