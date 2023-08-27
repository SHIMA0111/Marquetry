from marquetry import Function, cuda_backend, utils


class MaxPooling2D(Function):
    def __init__(self, kernel_size, stride=1, pad=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.input_shape = None
        self.input_dtype = None

        self.indexes = None

    def forward(self, x):
        self.input_shape = x.shape
        self.input_dtype = x.dtype

        col = utils.im2col_array(x, self.kernel_size, self.stride, self.pad, to_matrix=False)
        batch_size, channels, kernel_height, kernel_weight, out_height, out_width = col.shape
        col = col.reshape((batch_size, channels, kernel_height * kernel_weight, out_height, out_width))

        self.indexes = col.argmax(axis=2)
        y = col.max(axis=2)

        self.retain_inputs(())
        return y

    def backward(self, x, grad_y):
        return MaxPooling2DGrad(self)(grad_y[0])


class MaxPooling2DGrad(Function):
    def __init__(self, pooling2d):
        self.pooling2d = pooling2d
        self.kernel_size = pooling2d.kernel_size
        self.stride = pooling2d.stride
        self.pad = pooling2d.pad
        self.input_shape = pooling2d.input_shape
        self.dtype = pooling2d.input_dtype
        self.indexes = pooling2d.indexes

        self.shape = None

    def forward(self, grad_y):
        self.shape = grad_y.shape
        self.dtype = grad_y.dtype

        xp = cuda_backend.get_array_module(grad_y)

        batch_size, channels, output_height, output_width = grad_y.shape
        batch_size, channels, height, width = self.input_shape
        kernel_height, kernel_width = utils.pair(self.kernel_size)

        grad_col = xp.zeros(
            (batch_size * channels * output_height * output_width * kernel_height * kernel_width), dtype=self.dtype)

        indexes = (self.indexes.ravel() + xp.arange(
            0, self.indexes.size * kernel_height * kernel_width, kernel_height * kernel_width))
        grad_col[indexes] = grad_y.ravel()
        grad_col = grad_col.reshape((batch_size, channels, output_height, output_width, kernel_height, kernel_width))
        grad_col = xp.swapaxes(grad_col, 2, 4)
        grad_col = xp.swapaxes(grad_col, 3, 5)

        grad_x = utils.col2im_array(grad_col, (batch_size, channels, height, width),
                                    self.kernel_size, self.stride, self.pad, to_matrix=False)

        self.retain_inputs(())
        return grad_x

    def backward(self, x, grad_grad_y):
        f = Pooling2DWithIndexes(self)
        return f(grad_grad_y[0])


class Pooling2DWithIndexes(Function):
    def __init__(self, pooling2d):
        self.kernel_size = pooling2d.kernel_size
        self.stride = pooling2d.stride
        self.pad = pooling2d.pad
        self.input_shape = pooling2d.shape
        self.dtype = pooling2d.dtype
        self.indexes = pooling2d.indexes

    def forward(self, x):
        xp = cuda_backend.get_array_module(x)
        col = utils.im2col_array(x, self.kernel_size, self.stride, self.pad, to_matrix=False)
        batch_size, channels, kernel_height, kernel_width, out_height, out_width = col.shape

        col = col.reshape((batch_size, channels, kernel_height * kernel_width, out_height, out_width))
        col = col.transpose((0, 1, 3, 4, 2)).reshape(-1, kernel_height * kernel_width)
        indexes = self.indexes.ravel()
        col = col[xp.arange(len(indexes)), indexes]

        self.retain_inputs(())
        return col.reshape(batch_size, channels, out_height, out_width)


def max_pooling_2d(x, kernel_size, stride=1, pad=0):
    return MaxPooling2D(kernel_size, stride, pad)(x)