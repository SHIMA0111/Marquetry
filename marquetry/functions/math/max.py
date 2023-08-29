from marquetry import cuda_backend
from marquetry import Function
from marquetry import functions
from marquetry import utils


class Max(Function):
    def __init__(self, axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        y = x.max(axis=self.axis, keepdims=self.keepdims)

        self.retain_outputs((0,))
        return y

    def backward(self, x, grad_y):
        x = x[0]
        y = self.output_data[0]
        grad_y = grad_y[0]

        xp = cuda_backend.get_array_module(x)

        shape = utils.max_backward_shape(x, self.axis)
        grad_y = functions.reshape(grad_y, shape)
        y = functions.reshape(y, shape)
        cond = xp.array(x == y)
        grad_y = functions.broadcast_to(grad_y, cond.shape)

        return grad_y * cond


def max(x, axis=None, keepdims=False):
    return Max(axis, keepdims)(x)