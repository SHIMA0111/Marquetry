from marquetry import cuda_backend
from marquetry import Function


class Repeat(Function):
    def __init__(self, repeat_num, axis):
        self.repeat_num = repeat_num
        self.axis = axis

    def forward(self, x):
        xp = cuda_backend.get_array_module(x)
        y = xp.repeat(x, self.repeat_num, self.axis)

        return y

    def backward(self, x, grad_y):
        x_shape = x[0].shape

        grad_x = RepeatGrad(x_shape, self.repeat_num, self.axis)(grad_y[0])

        return grad_x


class RepeatGrad(Function):
    def __init__(self, in_shape, repeat_num, axis):
        self.in_shape = in_shape
        self.repeat_num = repeat_num
        self.axis = axis

    def forward(self, grad_y):
        xp = cuda_backend.get_array_module(grad_y)

        original_num = self.in_shape[self.axis]
        grad_shape = list(grad_y.shape)
        grad_shape[self.axis - 1] *= original_num
        grad_shape[self.axis] = int(grad_shape[self.axis] / original_num)
        grad_shape = tuple(grad_shape)

        grad_y = grad_y.reshape(grad_shape)
        grad_y = xp.sum(grad_y, axis=self.axis)
        grad_x = grad_y.reshape(self.in_shape)

        self.retain_inputs(())
        return grad_x

    def backward(self, x, grad_grad_y):
        grad_grad_x = repeat(grad_grad_y[0], self.repeat_num, self.axis)

        return grad_grad_x


def repeat(x, repeats, axis):
    return Repeat(repeats, axis)(x)
