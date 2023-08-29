from marquetry import cuda_backend
from marquetry import Function


class Sigmoid(Function):
    def forward(self, x):
        xp = cuda_backend.get_array_module(x)

        y = xp.exp(xp.minimum(0, x)) / (1 + xp.exp(-xp.abs(x)))

        self.retain_inputs(())
        self.retain_outputs((0,))
        return y

    def backward(self, x, grad_y):
        y = self.output_data[0]
        grad_x = y * (1 - y) * grad_y[0]

        return grad_x


def sigmoid(x):
    return Sigmoid()(x)
