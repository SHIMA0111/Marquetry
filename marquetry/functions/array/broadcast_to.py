from marquetry import Function, cuda_backend, functions


class BroadcastTo(Function):
    def __init__(self, shape):
        self.shape = shape

        self.x_shape = None

    def forward(self, x):
        xp = cuda_backend.get_array_module(x)
        if x.shape == self.shape:
            return x

        self.x_shape = x.shape
        y = xp.broadcast_to(x, self.shape)

        self.retain_inputs(())
        return y

    def backward(self, x, grad_y):
        if self.x_shape is None:
            return grad_y[0]

        grad_x = functions.sum_to(grad_y[0], self.x_shape)

        return grad_x


def broadcast_to(x, shape):
    return BroadcastTo(shape)(x)
