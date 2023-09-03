from marquetry import Function


class MeanSquaredError(Function):
    def forward(self, x0, x1):
        diff = x0 - x1
        y = (diff ** 2).sum() / diff.dtype.type(diff.size)

        return y

    def backward(self, inputs, grad_y):
        x0, x1 = inputs
        diff = x0 - x1
        grad_x0 = 2. * diff / diff.size * grad_y[0]
        grad_x1 = -grad_x0 * grad_y[0]

        return grad_x0, grad_x1


def mean_squared_error(x0, x1):
    return MeanSquaredError()(x0, x1)
