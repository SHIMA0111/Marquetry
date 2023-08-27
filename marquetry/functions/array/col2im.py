from marquetry import Function, utils, functions


class Col2im(Function):
    def __init__(self, input_shape, kernel_size, stride, pad, to_matrix):
        super().__init__()

        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.to_matrix = to_matrix

    def forward(self, x):
        y = utils.col2im_array(x, self.input_shape, self.kernel_size, self.stride, self.pad, self.to_matrix)

        self.retain_inputs(())
        return y

    def backward(self, grad_y):
        grad_x = functions.im2col(grad_y[0], self.kernel_size, self.stride, self.pad, self.to_matrix)

        return grad_x


def col2im(col, input_shape, kernel_size, stride=1, pad=0, to_matrix=True):
    return Col2im(input_shape, kernel_size, stride, pad, to_matrix)(col)
