import numpy as np

import marquetry.cuda_backend as cuda_backend
import marquetry.utils as utils
from marquetry import functions
from marquetry import Layer
from marquetry import Parameter


class Deconvolution2D(Layer):
    """2D deconvolutional layer (transpose convolution).

        The `Deconvolution2D` layer is commonly used for upsampling in tasks such as image segmentation
        and super-resolution.
        It performs 2D deconvolution operations, also known as transpose convolution or
        fractionally strided convolution.

        Args:
            out_channels (int): The number of output channels (i.e., the number of filters or kernels).
            kernel_size (int or tuple of int): The size of the deconvolutional kernel.
            stride (int, tuple of ints, optional): The stride of the deconvolution operation.
                Default is 1.
            pad (int, tuple of ints): The amount of zero-padding to apply around the input data.
                Default is 0.
            nobias (bool): If True, no bias term is added to the deconvolution.
                Default is False.
            dtype (numpy.dtype or string): The data type used for weights and biases.
                Default is np.float32.
            in_channels (int or None): The number of input channels.

        Note:
            kernel_size:
                If an int is provided, it's treated as a square kernel size (e.g., 3 for a 3x3 kernel).
                If a tuple of int is provided, it specifies the height and width
                of the kernel (e.g., (3, 3) for a 3x3 kernel).
            in_channels:
                This parameter is automatically determined from the input data shape
                and does not need to be specified except a special use case.

        Attributes:
            w (marquetry.Parameter): The weight parameter.
            b (marquetry.Parameter): The bias parameter (if used).

        Examples:
            >>> x = np.arange(0, 384).reshape((1, 64, 2, 3))
            >>> x.shape
            (1, 64, 2, 3)
            >>> deconv2d = Deconvolution2D(3, 3)
            >>> y = deconv2d(x)
            >>> y.shape
            (1, 3, 4, 5)

    """

    def __init__(self, out_channels, kernel_size, stride=1, pad=0, nobias=False,
                 dtype=np.float32, in_channels=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.dtype = dtype

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_channels, dtype=dtype), name="b")

        self.w = Parameter(None, name="w")

    def _init_w(self, xp=np):
        channels, out_channels = self.in_channels, self.out_channels
        kernel_height, kernel_width = utils.pair(self.kernel_size)
        scale = xp.sqrt(1 / (channels * kernel_height * kernel_width))
        w_data = xp.random.randn(channels, out_channels, kernel_height, kernel_width).astype(self.dtype) * scale
        self.w.data = w_data

        if self.b is not None and xp is not np:
            self.b.to_gpu()

    def forward(self, x):
        if self.w.data is None:
            self.in_channels = x.shape[1]
            xp = cuda_backend.get_array_module(x)
            self._init_w(xp=xp)

        y = functions.deconvolution_2d(x, self.w, self.b, self.stride, self.pad)

        return y
