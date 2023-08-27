import marquetry
from marquetry import Function, cuda_backend, configuration


class Dropout(Function):
    def __init__(self, dropout_rate):
        self.dropout_rate = dropout_rate

        self.mask = None

    def forward(self, x):
        if marquetry.configuration.config.train:
            xp = cuda_backend.get_array_module(x)
            mask = xp.random.rand(*x.shape) > self.dropout_rate
            self.mask = mask
            scale = xp.array(1.0 - self.dropout_rate).astype(x.dtype)
            y = x * mask / scale
        else:
            y = x

        self.retain_inputs(())

        return y

    def backward(self, x, grad_y):
        if configuration.config.train_mode:
            grad_x = grad_y[0] * self.mask
        else:
            raise Exception("You execute non-train mode so you can't do backward.")

        return grad_x


def dropout(x, dropout_rate=0.5):
    return Dropout(dropout_rate)(x)