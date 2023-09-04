from marquetry import cuda_backend
from marquetry import Function


class MeanAbsoluteError(Function):
    def forward(self, x0, x1):
        xp = cuda_backend.get_array_module(x0)
        diff = x0 - x1
        y = xp.absolute(diff).sum() / diff.dtype.type(diff.size)

        return y

    def backward(self, inputs, grad_y):
        xp = cuda_backend.get_array_module(inputs[0])
        x0, x1 = inputs

        diff = x0 - x1
        sign_map = xp.asarray(diff >= 0, dtype="f")
        sign_map -= .5
        sign_map *= 2

        coefficient = grad_y[0] * grad_y[0].dtype.type(1. / diff.size)
        grad_x = coefficient * sign_map

        return grad_x, -grad_x


def mean_absolute_error(x0, x1):
    return MeanAbsoluteError()(x0, x1)
