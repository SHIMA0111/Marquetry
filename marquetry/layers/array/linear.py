import numpy as np

import marquetry.cuda_backend as cuda_backend
from marquetry import functions
from marquetry import Layer
from marquetry import Parameter


class Linear(Layer):
    """Linear layer (fully connected layer).

        The `Linear` layer, also known as the fully connected layer,
        computes the matrix multiplication between input data and weights.
        It is commonly used in neural networks as intermediate layers or
        output layers to transform input data into the desired dimension.

        Args:
            out_size (int): The number of output units (dimension of the output).
            nobias (bool): If True, no bias term is added to the linear transformation.
                Default is False.
            dtype (numpy.dtype): The data type used for weights and biases.
                Default is np.float32.
            in_size (int or None): The number of input units (dimension of the input).

        Note:
            in_size:
                This is automatically determined from the input data shape and
                does not need to be specified except a special use case.

        Attributes:
            w (marquetry.Parameter): The weight parameter.
            b (marquetry.Parameter): The bias parameter (if used).

        Examples:
            >>> x = np.array([[1, 2, 3, 4, 5]])
            >>> linear = Linear(100)
            >>> y = linear(x)
            >>> y.shape
            (1, 100)

    """

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
        y = functions.linear(x, self.w, self.b)
        return y
