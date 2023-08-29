from marquetry import cuda_backend
from marquetry import Function
from marquetry import functions
from marquetry import utils
from marquetry.functions.connection.convolution_2d_grad_w import Conv2DGradW


class Convolution2D(Function):
    def __init__(self, stride=1, pad=0):
        super().__init__()
        self.stride = utils.pair(stride)
        self.pad = utils.pair(pad)

    def forward(self, x, w, b):
        xp = cuda_backend.get_array_module(x)

        kernel_height, kernel_width = w.shape[2:]
        col = utils.im2col_array(x, (kernel_height, kernel_width), self.stride, self.pad, to_matrix=False)

        y = xp.tensordot(col, w, ((1, 2, 3), (1, 2, 3)))
        if b is not None:
            y += b

        y = xp.rollaxis(y, 3, 1)

        return y

    def backward(self, inputs, grad_y):
        x, w, b = inputs
        grad_y = grad_y[0]

        grad_x = functions.deconvolution_2d(
            grad_y, w, b=None, stride=self.stride, pad=self.pad, out_size=(x.shape[2], x.shape[3]))

        grad_w = Conv2DGradW(self)(x, grad_y)

        grad_b = None

        if b is not None:
            grad_b = grad_y.sum(axis=(0, 2, 3))

        return grad_x, grad_w, grad_b


def convolution_2d(x, w, b=None, stride=1, pad=0):
    return Convolution2D(stride, pad)(x, w, b)