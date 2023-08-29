from marquetry import cuda_backend
from marquetry import Function
from marquetry import functions
from marquetry import utils
from marquetry.functions.connection.convolution_2d_grad_w import Conv2DGradW


class Deconvolution2D(Function):
    def __init__(self, stride=1, pad=0, out_size=None):
        super().__init__()
        self.stride = utils.pair(stride)
        self.pad = utils.pair(pad)
        self.out_size = out_size

        self.no_bias = False

    def forward(self, x, w, b):
        xp = cuda_backend.get_array_module(x)

        stride_height, stride_width = self.stride
        padding_height, padding_width = self.pad
        channels, out_channels, kernel_height, kernel_width = w.shape

        batch_size, channels, height, width = x.shape

        if self.out_size is None:
            out_height = utils.get_deconvolution_outsize(height, kernel_height, stride_height, padding_height)
            out_width = utils.get_deconvolution_outsize(width, kernel_width, stride_width, padding_width)
        else:
            out_height, out_width = utils.pair(self.out_size)

        img_shape = (batch_size, out_channels, out_height, out_width)
        grad_col = xp.tensordot(w, x, (0, 1))
        grad_col = xp.rollaxis(grad_col, 3)

        y = utils.col2im_array(
            grad_col, img_shape, (kernel_height, kernel_width), self.stride, self.pad, to_matrix=False)

        if b is not None:
            self.no_bias = True
            y += b.reshape((1, b.size, 1, 1))

        return y

    def backward(self, inputs, grad_y):
        x, w, b = inputs
        grad_y = grad_y[0]

        grad_x = functions.convolution_2d(grad_y, w, b=None, stride=self.stride, pad=self.pad)

        grad_w = Conv2DGradW(self)(grad_y, x)

        grad_b = None
        if b is not None:
            grad_b = grad_y.sum(axis=(0, 2, 3))

        return grad_x, grad_w, grad_b


def deconvolution_2d(x, w, b=None, stride=1, pad=0, out_size=None):
    return Deconvolution2D(stride, pad, out_size=out_size)(x, w, b)
