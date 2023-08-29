from marquetry import cuda_backend
from marquetry import Function
from marquetry import functions


class Squeeze(Function):
    def __init__(self, axis):
        self.axis = axis

    def forward(self, x):
        xp = cuda_backend.get_array_module(x)
        if x.shape[self.axis] != 1:
            raise ValueError("You can't squeeze non-one size axis element.")

        y = xp.squeeze(x, axis=self.axis)

        self.retain_inputs(())
        return y

    def backward(self, x, grad_y):
        grad_x = functions.unsqueeze(grad_y[0], self.axis)

        return grad_x


def squeeze(x, axis):
    return Squeeze(axis)(x)