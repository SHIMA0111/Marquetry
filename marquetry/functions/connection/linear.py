from marquetry import Function, functions


class Linear(Function):
    def forward(self, x, w, b):
        y = x.dot(w)
        if b is not None:
            y += b
        return y

    def backward(self, inputs, grad_y):
        x, w, b = inputs
        grad_y = grad_y[0]

        grad_b = None if b is None else functions.sum_to(grad_y, b.shape)

        grad_x = functions.matmul(grad_y, w.T)
        grad_w = functions.matmul(x.T, grad_y)

        return grad_x, grad_w, grad_b


def linear(x, w, b=None):
    return Linear()(x, w, b)
