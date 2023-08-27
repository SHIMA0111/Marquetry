from marquetry import Function


class MatMul(Function):
    def forward(self, x1, x2):
        y = x1.dot(x2)

        return y

    def backward(self, xs, grad_y):
        x1, x2 = xs
        grad_y = grad_y[0]

        grad_x1 = matmul(grad_y, x2.T)
        grad_x2 = matmul(x1.T, grad_y)

        return grad_x1, grad_x2


def matmul(x1, x2):
    return MatMul()(x1, x2)