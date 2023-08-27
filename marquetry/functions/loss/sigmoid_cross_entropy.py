from marquetry import Function, cuda_backend, as_variable, functions


class SigmoidCrossEntropy(Function):
    def __init__(self):
        self.batch_size = None

    def forward(self, x, t):
        if x.ndim != t.ndim:
            t = t.reshape(*x.shape)

        xp = cuda_backend.get_array_module(x)

        batch_size = x.shape[0] if x.ndim != 1 else len(x)
        p = xp.exp(xp.minimum(0, x)) / (1 + xp.exp(-xp.abs(x)))
        p = xp.clip(p, 1e-15, .999)
        tlog_p = t * xp.log(p) + (1 - t) * xp.log(1 - p)
        y = -1 * tlog_p.sum() / batch_size

        self.batch_size = batch_size

        return y

    def backward(self, inputs, grad_y):
        x, t = inputs
        if x.ndim != t.ndim:
            t = t.reshape(*x.shape)
        y = functions.sigmoid(x)

        batch_size = self.batch_size

        # grad_x = -(1 / batch_size) * ((t / y) - ((1 - t) / (1 - y))) * (y * (1 - y)) * grad_y
        grad_x = -(1 / batch_size) * (t * (1 - y) - y * (1 - t)) * grad_y[0]

        return grad_x


def sigmoid_cross_entropy(x, t):
    return SigmoidCrossEntropy()(x, t)


def simple_sigmoid_cross_entropy(x, t):
    if x.ndim != t.ndim:
        t = t.reshape(*x.shape)

    x, t = as_variable(x), as_variable(t)
    batch_size = len(x)
    p = functions.sigmoid(x)
    p = functions.clip(p, 1e-15, 0.99)
    tlog_p = t * functions.log(p) + (1 - t) * functions.log(1 - p)
    y = -1 * sum(tlog_p) / batch_size
    return y
