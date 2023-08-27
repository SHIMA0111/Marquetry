from marquetry import Function, utils, functions


class LogSoftmax(Function):
    def __init__(self, axis):
        self.axis = axis

    def forward(self, x):
        log_z = utils.logsumexp(x, self.axis)
        y = x - log_z

        self.retain_inputs(())
        self.retain_outputs((0,))
        return y

    def backward(self, x, grad_y):
        y = self.output_data[0]
        grad_y = grad_y[0]

        grad_x = grad_y - functions.exp(y) * grad_y.sum(axis=self.axis, keepdims=True)

        return grad_x


def log_softmax(x, axis=1):
    return LogSoftmax(axis)(x)