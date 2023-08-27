from marquetry import Function


class LeakyReLU(Function):
    def __init__(self, slope):
        self.slope = slope

    def forward(self, x):
        y = x.copy()
        y[x <= 0] *= self.slope

        return y

    def backward(self, x, grad_y):
        mask = (x[0] > 0).astype(grad_y.dtype)
        mask[mask <= 0] = self.slope

        grad_x = grad_y * mask

        return grad_x


def leaky_relu(x, slope=0.2):
    return LeakyReLU(slope)(x)
