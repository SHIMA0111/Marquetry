from marquetry import cuda_backend
from marquetry import Function
from marquetry import functions
from marquetry import utils


class Conv2DGradW(Function):
    """Weight gradient calculation class for Convolution2D backward."""
    def __init__(self, conv2d_instance):
        w = conv2d_instance.inputs[1]
        kernel_height, kernel_width = w.shape[2:]
        self.kernel_size = (kernel_height, kernel_width)
        self.stride = conv2d_instance.stride
        self.pad = conv2d_instance.pad

    def forward(self, x, grad_y):
        xp = cuda_backend.get_array_module(x)

        col = utils.im2col_array(x, self.kernel_size, self.stride, self.pad, to_matrix=False)
        grad_w = xp.tensordot(grad_y, col, ((0, 2, 3), (0, 4, 5)))

        self.retain_outputs((0,))
        return grad_w

    def backward(self, inputs, grad_ys):
        x, grad_y = inputs
        grad_w, = self.output_data

        x_height, x_width = x.shape[2:]
        grad_x = functions.deconvolution_2d(
            grad_y, grad_w, stride=self.stride, pad=self.pad, out_size=(x_height, x_width))
        grad_grad_y = functions.convolution_2d(x, grad_w, stride=self.stride, pad=self.pad)

        return grad_x, grad_grad_y
