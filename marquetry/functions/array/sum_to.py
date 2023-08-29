from marquetry import Function
from marquetry import functions
from marquetry import utils


class SumTo(Function):
    def __init__(self, shape):
        self.shape = shape

        self.x_shape = None

    def forward(self, x):
        if x.shape == self.shape:
            return x

        self.x_shape = x.shape
        y = utils.sum_to(x, self.shape)

        self.retain_inputs(())
        return y

    def backward(self, x, grad_y):
        if self.x_shape is None:
            return grad_y[0]

        grad_x = functions.broadcast_to(grad_y[0], self.x_shape)

        return grad_x


def sum_to(x, shape):
    return SumTo(shape)(x)
