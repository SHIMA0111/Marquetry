from marquetry import Function
from marquetry import functions
from marquetry import utils


class Col2im(Function):
    """Export image array from array.

        This class is commonly used by the backward computation of the convolution.

        Note:
            Generally, you don't need to execute ``forward`` and ``backward`` method manually.
            You should use only ``__call__`` method.
    """

    def __init__(self, image_shape, kernel_size, stride, pad, to_matrix):
        super().__init__()

        self.image_shape = image_shape
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.to_matrix = to_matrix

    def forward(self, x):
        y = utils.col2im_array(x, self.image_shape, self.kernel_size, self.stride, self.pad, self.to_matrix)

        self.retain_inputs(())
        return y

    def backward(self, grad_y):
        grad_x = functions.im2col(grad_y[0], self.kernel_size, self.stride, self.pad, self.to_matrix)

        return grad_x


def col2im(col, image_shape, kernel_size, stride=1, pad=0, to_matrix=True):
    """Export image array from array.

        This function is used by the backward computation of the convolution.

        Args:
            col (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                Input container that is :class:`marquetry.Container` array.
            image_shape (tuple): image shape the output image.
            kernel_size (int, tuple(y, x)): The convolution process kernel size.
            stride (int, tuple(y, x)): The convolution process stride size
            pad (int, tuple): The convolution process padding size
            to_matrix (bool): If this ``True``, the input array should be 2-dim array.
                Otherwise, the input array should be 6-dims array
                (batch_size, channels, kernel_height, kernel_width, output_height, output_width).
    """

    return Col2im(image_shape, kernel_size, stride, pad, to_matrix)(col)
