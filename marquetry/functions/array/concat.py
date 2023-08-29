from marquetry import cuda_backend
from marquetry import Function
from marquetry import functions


class Concat(Function):
    def __init__(self, axis):
        self.axis = axis

    def forward(self, *inputs):
        if len(inputs) == 1 and isinstance(inputs[0], (tuple, list)):
            inputs = inputs[0]

        xp = cuda_backend.get_array_module(inputs[0])
        y = xp.concatenate(inputs, axis=self.axis)

        return y

    def backward(self, inputs, grad_y):
        pre_index = 0
        indices = []
        for i, data in enumerate(inputs):
            if i == len(inputs) - 1:
                continue
            index = data.shape[self.axis]
            pre_index += index
            indices.append(pre_index)

        grad_x = functions.split(grad_y[0], indices, axis=self.axis)

        return grad_x


def concat(*inputs, axis=0):
    if len(inputs) == 1 and isinstance(inputs[0], (tuple, list)):
        inputs = tuple(inputs[0])

    return Concat(axis)(*inputs)

