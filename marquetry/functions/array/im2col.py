from marquetry import Function, utils, functions


class Im2col(Function):
    def __init__(self, kernel_size, stride, pad, to_matrix):
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.to_matrix = to_matrix

        self.input_shape = None
        self.x_shape = None

    def forward(self, x):
        self.input_shape = x.shape

        y = utils.im2col_array(
            x, kernel_size=self.kernel_size, stride=self.stride, pad=self.pad, to_matrix=self.to_matrix)

        self.retain_inputs(())
        return y

    def backward(self, grad_y):
        grad_x = functions.col2im(grad_y[0], self.input_shape, self.kernel_size, self.stride, self.pad, self.to_matrix)

        return grad_x


def im2col(img, kernel_size, stride=1, pad=0, to_matrix=True):
    return Im2col(kernel_size, stride, pad, to_matrix)(img)