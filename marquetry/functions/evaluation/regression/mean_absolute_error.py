from marquetry import cuda_backend
from marquetry import Function


class MeanAbsoluteError(Function):
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
    return MeanAbsoluteError(multi_output)(y, t)


mae = mean_absolute_error
