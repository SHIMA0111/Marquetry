from marquetry import Function
from marquetry import functions
from marquetry import utils


class Sum(Function):
    def __init__(self, axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims

        self.x_shape = None

    def forward(self, x):
        self.x_shape = x.shape
        y = x.sum(axis=self.axis, keepdims=self.keepdims)

        self.retain_inputs(())
        return y

    def backward(self, x, grad_y):
        grad_y = utils.reshape_sum_backward(grad_y[0], self.x_shape, self.axis, self.keepdims)
        grad_x = functions.broadcast_to(grad_y, self.x_shape)
        return grad_x


def sum(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)
