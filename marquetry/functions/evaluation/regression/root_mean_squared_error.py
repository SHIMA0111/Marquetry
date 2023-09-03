from marquetry import cuda_backend
from marquetry import Function


class RootMeanSquaredError(Function):
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
        return root_mean_squared_error_value


def root_mean_squared_error(y, t, multi_output="uniform_average"):
    return RootMeanSquaredError(multi_output)(y, t)


rmse = root_mean_squared_error
