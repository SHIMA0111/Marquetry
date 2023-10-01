from marquetry import Function
from marquetry import functions
from marquetry import utils


class Im2col(Function):
    """Transform an image array to matrix to easy use it convolution.

        This function transform image to the useful for the convolution process.

        Note:
            Generally, you don't need to execute ``forward`` and ``backward`` method manually.
            You should use only ``__call__`` method.
    """

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
    """Transform an image array to matrix to easy use it convolution.

        This function transform image to the useful for the convolution process.

        Args:
            img (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                Input container that is :class:`marquetry.Container` array.
            kernel_size (int, tuple(y, x)): The convolution process kernel size
            stride (int, tuple(y, x)): The convolution process stride size
            pad (int, tuple): The convolution process padding size
            to_matrix (bool): If this ``True``, the output array is 2-dim array.
                Otherwise, the output array is 6-dims array.
    """
    return Im2col(kernel_size, stride, pad, to_matrix)(img)
