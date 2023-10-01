from marquetry import cuda_backend
from marquetry import Function
from marquetry import functions
from marquetry import utils
from marquetry.functions.connection.convolution_2d_grad_w import Conv2DGradW


class Deconvolution2D(Function):
    """Perform 2D deconvolution on an input array.

        This class defines a function that performs 2D deconvolution on an input array or container using the given
        de-convolutional kernel and optional bias.

        Note:
            Generally, you don't need to execute ``forward`` and ``backward`` method manually.
            You should use only ``__call__`` method.
    """

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
    """Perform 2D deconvolution on an input array.


        This class defines a function that performs 2D deconvolution on an input array or container using the given
        de-convolutional kernel and optional bias.

        Args:
            x (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The input array or container to be de-convolved.
            w (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The de-convolutional kernel.
            b (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray` or None):
                The bias term. If None, no bias is added.
            stride (int or tuple of ints):
                The stride for moving the de-convolutional kernel. If an int is provided,
                it is converted to a tuple of the same value for both dimensions.
                Defaults to 1.
            pad (int or tuple of ints): The padding added to the input array or container before deconvolution.
                If an int is provided, it is converted to a tuple of the same value for both dimensions.
                Defaults to 0.
            out_size (tuple of ints or None): The size of the output feature map.
                If None, the size is determined based on the input and kernel sizes.

        Returns:
            :class:`marquetry.Container`: The result of 2D deconvolution on the input
                array or container using the provided kernel and bias (if any).

        Examples:
            >>> x = np.arange(0, 576).reshape(1, 16, 6, 6)
            >>> w = np.ones((16, 3, 2, 2))
            >>> deconvolution_2d(x, w).shape
            (1, 3, 7, 7)
        """

    return Deconvolution2D(stride, pad, out_size=out_size)(x, w, b)
