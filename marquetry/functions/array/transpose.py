from marquetry import Function, cuda_backend


class Transpose(Function):
    def __init__(self, axes=None):
        self.axes = axes

    def forward(self, x):
        y = x.transpose(self.axes)
        self.retain_inputs(())
        return y

    def backward(self, x, grad_y):
        if self.axes is None:
            return transpose(grad_y[0])

        xp = cuda_backend.get_array_module(grad_y[0])

        axes_len = len(self.axes)
        inv_axes = tuple(xp.argsort([ax % axes_len for ax in self.axes]))
        return transpose(grad_y[0], inv_axes)


def transpose(x, axes=None):
    return Transpose(axes)(x)