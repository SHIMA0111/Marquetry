from marquetry import cuda_backend
from marquetry import Function


class ReLU(Function):
    def forward(self, x):
        xp = cuda_backend.get_array_module(x)
        y = xp.maximum(x, 0.0)

        return y

    def backward(self, x, grad_y):
        x, = x
        mask = x > 0
        grad_x = grad_y[0] * mask

        return grad_x


def relu(x):
    return ReLU()(x)
