from marquetry import cuda_backend
from marquetry import Function


class Clip(Function):
    def __init__(self, x_min, x_max):
        self.x_min = x_min
        self.x_max = x_max

    def forward(self, x):
        xp = cuda_backend.get_array_module(x)

        y = xp.clip(x, self.x_min, self.x_max)

        return y

    def backward(self, x, grad_y):
        x = x[0]

        mask = (x >= self.x_min) * (x <= self.x_max)
        grad_x = grad_y[0] * mask

        return grad_x


def clip(x, x_min, x_max):
    return Clip(x_min, x_max)(x)