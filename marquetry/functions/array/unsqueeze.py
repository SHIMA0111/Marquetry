from marquetry import Function
from marquetry import functions


class UnSqueeze(Function):
    def __init__(self, axis):
        self.axis = axis

    def forward(self, x):
        x_shape = x.shape

        new_shape = list(x_shape)
        new_shape.insert(self.axis, 1)
        new_shape = tuple(new_shape)

        y = x.reshape(new_shape)

        self.retain_inputs(())
        return y

    def backward(self, x, grad_y):
        grad_x = functions.squeeze(grad_y[0], self.axis)

        return grad_x


def unsqueeze(x, axis):
    return UnSqueeze(axis)(x)
