from marquetry import cuda_backend
from marquetry import Function


class Sin(Function):
    def forward(self, x):
        xp = cuda_backend.get_array_module(x)
        y = xp.sin(x)

        return y

    def backward(self, x, grad_y):
        grad_x = cos(x[0]) * grad_y[0]

        return grad_x


def sin(x):
    return Sin()(x)


class Cos(Function):
    def forward(self, x):
        xp = cuda_backend.get_array_module(x)
        y = xp.cos(x)

        return y

    def backward(self, x, grad_y):
        grad_x = -sin(x[0]) * grad_y[0]

        return grad_x


def cos(x):
    return Cos()(x)


class Tan(Function):
    def forward(self, x):
        xp = cuda_backend.get_array_module(x)
        y = xp.tan(x)

        self.retain_inputs(())
        self.retain_outputs((0,))

        return y

    def backward(self, x, grad_y):
        y = self.output_data[0]
        grad_x = 1 + y ** 2
        grad_x *= grad_y[0]

        return grad_x


def tan(x):
    return Tan()(x)
