from marquetry import cuda_backend
from marquetry import Function


class Exp(Function):
    def forward(self, x):
        xp = cuda_backend.get_array_module(x)
        y = xp.exp(x)

        self.retain_inputs(())
        self.retain_outputs((0,))
        return y

    def backward(self, x, grad_y):
        y = self.output_data[0]
        grad_x = grad_y[0] * y

        return grad_x


def exp(x):
    return Exp()(x)


class Log(Function):
    def forward(self, x):
        xp = cuda_backend.get_array_module(x)
        y = xp.log(x)

        return y

    def backward(self, x, grad_y):
        grad_x = grad_y[0] / x[0]

        return grad_x


def log(x):
    return Log()(x)


class Log2(Function):
    def forward(self, x):
        xp = cuda_backend.get_array_module(x)
        y = xp.log2(x)

        return y

    def backward(self, x, grad_y):
        grad_x = grad_y[0] / (x[0] * log(2))

        return grad_x


def log2(x):
    return Log2()(x)


class Log10(Function):
    def forward(self, x):
        xp = cuda_backend.get_array_module(x)
        y = xp.log10(x)

        return y

    def backward(self, x, grad_y):
        grad_x = grad_y[0] / (x[0] * log(10))

        return grad_x


def log10(x):
    return Log10()(x)
