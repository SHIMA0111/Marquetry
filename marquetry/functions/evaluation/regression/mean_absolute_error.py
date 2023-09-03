from marquetry import cuda_backend
from marquetry import Function


class MeanAbsoluteError(Function):
    def forward(self, y, t):
        assert y.size == t.size

        xp = cuda_backend.get_array_module(y)
        if t.ndim == 1:
            t = t.reshape((-1, 1))
        y = y.reshape(t.shape)

        mae_value = xp.mean(xp.absolute(y - t), axis=0)

        self.retain_inputs(())
        return mae_value


def mean_absolute_error(y, t):
    return MeanAbsoluteError()(y, t)


mae = mean_absolute_error
