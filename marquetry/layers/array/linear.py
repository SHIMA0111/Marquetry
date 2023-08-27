import numpy as np

import marquetry.cuda_backend as cuda_backend
import marquetry.functions as funcs
from marquetry import Layer, Parameter


class Linear(Layer):
    def __init__(self, out_size, nobias=False, dtype=np.float32, in_size=None):
        super().__init__()
        self.in_size = in_size
        self.outsize = out_size
        self.dtype = dtype

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_size, dtype=dtype), name="bias")

        self.w = Parameter(None, name="weight")

    def _init_w(self, xp=np):
        in_size, out_size = self.in_size, self.outsize
        w_data = xp.random.randn(in_size, out_size).astype(self.dtype) * xp.sqrt(1 / in_size)
        self.w.data = w_data
        if self.b is not None and xp is not np:
            self.b.to_gpu()

    def forward(self, x):
        if self.w.data is None:
            self.in_size = x.shape[-1]
            xp = cuda_backend.get_array_module(x)
            self._init_w(xp=xp)
        y = funcs.linear(x, self.w, self.b)
        return y
