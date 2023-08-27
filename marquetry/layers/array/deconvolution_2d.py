import numpy as np

import marquetry.cuda_backend as cuda_backend
import marquetry.functions as funcs
import marquetry.utils as utils
from marquetry import Layer, Parameter


class Deconvolution2D(Layer):
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

        y = funcs.deconvolution_2d(x, self.w, self.b, self.stride, self.pad)

        return y