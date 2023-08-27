from marquetry import Function
from marquetry import cuda_backend


class Reciprocal(Function):
    def forward(self, x):
        xp = cuda_backend.get_array_module(x)
        y = xp.reciprocal(x)

        self.retain_inputs(())
        self.retain_outputs((0,))

        return y

    def backward(self, x, grad_y):
        y = self.output_data[0]

        grad_x = -(1 / y ** 2) * grad_y

        return grad_x


def reciprocal(x):
    return Reciprocal()(x)
