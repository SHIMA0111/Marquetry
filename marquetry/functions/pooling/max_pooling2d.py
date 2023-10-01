from marquetry import cuda_backend
from marquetry import Function
from marquetry import utils


class MaxPooling2D(Function):
    """Apply 2D Max Pooling to the input tensor.

        Max Pooling is a down-sampling operation that extracts the maximum value from each local region
        of the input tensor, defined by the kernel size, and produces a smaller output tensor.

        Note:
            Generally, you don't need to execute ``forward`` and ``backward`` method manually.
            You should use only ``__call__`` method.
    """

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
    """Compute gradients for the 2D Max Pooling operation.

        This class computes the gradient of the loss with respect to the input of the Max Pooling operation.
    """

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

        # the indexes of the argmax in flatten of the col
        indexes = (self.indexes.ravel() + xp.arange(
            0, self.indexes.size * kernel_height * kernel_width, kernel_height * kernel_width))
        # assignment and other is 0
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
    """Compute gradients for the 2D Max Pooling Gradient.

        This class computes the gradient of the loss with respect to the input of the Max Pooling operation.
        In the second-order differential should extract the same element as the first MaxPooling from
        the gradient array.
        So that, this class extract the element by the original argmax indexes.
    """
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


def max_pooling_2d(x, kernel_size, stride=None, pad=0):
    """
        Apply 2D Max Pooling to the input tensor.

        Max Pooling is a downsampling operation that extracts the maximum value from each local region
        of the input tensor, defined by the kernel size, and produces a smaller output tensor.

        Args:
            x (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The input tensor.
            kernel_size (int or tuple): The size of the pooling kernel. If int, the same size is used for both
                height and width.
            stride (int): The stride of the pooling operation.
                If stride is None, stride is the same size as the kernel_size.
            pad (int): The amount of zero-padding around the input.
                Default is 0.

        Examples:
            >>> x = np.random.randn(1, 3, 20, 20)
            >>> x.shape
            (1, 3, 20, 20)
            >>> pool_data = max_pooling_2d(x, kernel_size=(2, 2))
            >>> pool_data.shape
            (1, 3, 10, 10)

        Returns:
            marquetry.Container: The result of the Max Pooling operation.

    """
    if stride is None:
        stride = kernel_size

    return MaxPooling2D(kernel_size, stride, pad)(x)
