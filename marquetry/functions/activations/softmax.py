from marquetry import cuda_backend
from marquetry import Function


class Softmax(Function):
    def __init__(self, axis):
        self.axis = axis

    def forward(self, x):
        xp = cuda_backend.get_array_module(x)

        y = x - x.max(axis=self.axis, keepdims=True)
        y = xp.exp(y)
        y /= y.sum(axis=self.axis, keepdims=True)

        self.retain_inputs(())
        self.retain_outputs((0,))
        return y

    def backward(self, x, grad_y):
        y = self.output_data[0]
        grad_x = y * grad_y[0]
        sum_grad_x = grad_x.sum(axis=self.axis, keepdims=True)
        grad_x -= y * sum_grad_x

        return grad_x


def softmax(x, axis=1):
    return Softmax(axis)(x)