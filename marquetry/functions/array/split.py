import numpy as np

from marquetry import Function, cuda_backend, functions


class Split(Function):
    def __init__(self, indices, axis):
        self.axis = axis
        if np.isscalar(indices):
            indices = (indices,)
        self.indices = indices

    def forward(self, x):
        xp = cuda_backend.get_array_module(x)
        y = xp.split(x, self.indices, axis=self.axis)

        self.retain_inputs(())
        return tuple(y)

    def backward(self, x, grad_ys):
        print(grad_ys)
        grad_x = functions.concat(grad_ys, axis=self.axis)

        return grad_x


def split(x, indices, axis):
    return Split(indices, axis)(x)