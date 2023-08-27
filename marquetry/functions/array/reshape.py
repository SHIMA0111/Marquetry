from marquetry import Function


class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape

        self.x_shape = None

    def forward(self, x):
        self.x_shape = x.shape
        y = x.reshape(self.shape)

        self.retain_inputs(())
        return y

    def backward(self, x, grad_y):
        return reshape(grad_y[0], self.x_shape)


def reshape(x, shape):
    return Reshape(shape)(x)
