from marquetry import cuda_backend
from marquetry import Function


class Tanh(Function):
    def forward(self, x):
        xp = cuda_backend.get_array_module(x)
        y = xp.tanh(x)

        self.retain_inputs(())
        self.retain_outputs((0,))
        return y

    def backward(self, x, grad_y):
        y = self.output_data[0]
        grad_x = grad_y[0] * (1 - y ** 2)

        return grad_x


def tanh(x):
    return Tanh()(x)
