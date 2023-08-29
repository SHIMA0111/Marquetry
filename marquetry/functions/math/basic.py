import marquetry
from marquetry import as_array
from marquetry import Function


class Add(Function):
    def __init__(self):
        self.x0_shape = None
        self.x1_shape = None

    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 + x1

        self.retain_inputs(())

        return y

    def backward(self, x, grad_y):
        grad_x0, grad_x1 = grad_y[0], grad_y[0]
        if self.x0_shape != self.x1_shape:
            grad_x0 = marquetry.functions.sum_to(grad_x0, self.x0_shape)
            grad_x1 = marquetry.functions.sum_to(grad_x1, self.x1_shape)

        return grad_x0, grad_x1


def add(x0, x1):
    x1 = as_array(x1, marquetry.cuda_backend.get_array_module(x0.data))
    return Add()(x0, x1)


class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y

    def backward(self, inputs, grad_y):
        x0, x1 = inputs
        grad_x0 = grad_y[0] * x1
        grad_x1 = grad_y[0] * x0
        if x0.shape != x1.shape:
            grad_x0 = marquetry.functions.sum_to(grad_x0, x0.shape)
            grad_x1 = marquetry.functions.sum_to(grad_x1, x1.shape)

        return grad_x0, grad_x1


def mul(x0, x1):
    x1 = as_array(x1, marquetry.cuda_backend.get_array_module(x0.data))
    return Mul()(x0, x1)


class Neg(Function):
    def forward(self, x):
        self.retain_inputs(())
        return -x

    def backward(self, x, grad_y):
        return -grad_y[0]


def neg(x):
    return Neg()(x)


class Sub(Function):
    def __init__(self):
        self.x0_shape = None
        self.x1_shape = None

    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 - x1

        self.retain_inputs(())
        return y

    def backward(self, inputs, grad_x):
        grad_x0 = grad_x[0]
        grad_x1 = -grad_x[0]
        if self.x0_shape != self.x1_shape:
            grad_x0 = marquetry.functions.sum_to(grad_x0, self.x0_shape)
            grad_x1 = marquetry.functions.sum_to(grad_x1, self.x1_shape)

        return grad_x0, grad_x1


def sub(x0, x1):
    x1 = as_array(x1, marquetry.cuda_backend.get_array_module(x0.data))
    return Sub()(x0, x1)


def rsub(x0, x1):
    x1 = as_array(x1, marquetry.cuda_backend.get_array_module(x0.data))
    return Sub()(x1, x0)


class Div(Function):
    def forward(self, x0, x1):
        y = x0 / x1
        return y

    def backward(self, inputs, grad_y):
        x0, x1 = inputs
        grad_x0 = grad_y[0] / x1
        grad_x1 = grad_y[0] * (-x0 / x1 ** 2)
        if x0.shape != x1.shape:
            grad_x0 = marquetry.functions.sum_to(grad_x0, x0.shape)
            grad_x1 = marquetry.functions.sum_to(grad_x1, x1.shape)

        return grad_x0, grad_x1


def div(x0, x1):
    x1 = as_array(x1, marquetry.cuda_backend.get_array_module(x0.data))
    return Div()(x0, x1)


def rdiv(x0, x1):
    x1 = as_array(x1, marquetry.cuda_backend.get_array_module(x0.data))
    return Div()(x1, x0)


class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x):
        y = x ** self.c
        return y

    def backward(self, x, grad_y):
        c = self.c
        grad_x = c * x[0] ** (c - 1) * grad_y[0]

        return grad_x


def pow(x, c):
    return Pow(c)(x)
