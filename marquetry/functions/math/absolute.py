import numpy as np

import marquetry
from marquetry import Function


class Absolute(Function):
    def forward(self, x):
        y = np.abs(x)

        return y

    def backward(self, x, grad_y):
        mask = (x[0] >= 0).astype("f")
        mask -= .5
        mask *= 2.

        grad_y = grad_y[0]
        if mask.shape != grad_y.shape:
            marquetry.functions.broadcast_to(grad_y, x[0].shape)

        grad_x = grad_y * mask

        return grad_x


def absolute(x):
    return Absolute()(x)
