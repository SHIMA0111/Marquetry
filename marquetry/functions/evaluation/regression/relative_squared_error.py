from marquetry import cuda_backend
from marquetry import Function


class RelativeSquaredError(Function):
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

        squared_error = xp.sum(xp.square(y - t), axis=0)
        squared_deviation = xp.sum(xp.square(t - correct_average), axis=0)

        relative_squared_error_value = xp.where(squared_deviation != 0,
                                                squared_error / squared_deviation,
                                                1.0)

        if self.multi_output == "uniform_average":
            return xp.asarray(relative_squared_error_value.mean())
        else:
            return relative_squared_error_value


def relative_squared_error(y, t, multi_output="uniform_average"):
    return RelativeSquaredError(multi_output)(y, t)
