import numpy as np

from marquetry import cuda_backend
from marquetry import functions
from marquetry import Layer
from marquetry import Parameter


class DynamicSwish(Layer):
    """Dynamic Swish layer.

        This layer defines the Swish activation function with learnable ``beta``.
        Swish with learnable parameter sometimes indicates more accuracy than normal swish.
        If you want to use learnable swish, please use this layer as activation function.

        However, the normal swish express excellent result as-is,
        so you may use :meth:`marquetry.functions.swish` at first.

        Args:
            init_beta (float): The init param value. Default is 1.0.

        Attributes:
            beta(float or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The learnable parameter in this activation function.

        Examples:
            >>> x = np.random.randn(4, 5)
            >>> print(x)
            [[ 0.68458909,  0.44318521,  0.97686183,  1.34406938,  0.02520422],
             [ 1.1901295 , -0.90362939,  0.70848885,  2.15739155, -1.32956708],
             [-1.39980426, -0.72885428, -0.88279792, -1.58149494,  0.83390716],
             [ 0.13488352,  2.42775131,  0.5688105 ,  0.82730737, -0.85926043]]
            >>> swish = DynamicSwish()
            >>> y = swish(x)
            >>> print(y)
            container([[ 0.45508893  0.26990765  0.70967556  1.06605986  0.01276092]
                       [ 0.91254873 -0.26052108  0.47473511  1.93379404 -0.27818663]
                       [-0.27694732 -0.23720285 -0.25830519 -0.26977752  0.58138376]
                       [ 0.07198327  2.23090716  0.36317905  0.5756283  -0.25562472]])

    """

    def __init__(self, init_beta=1.):
        super().__init__()

        self.beta = Parameter(np.array(init_beta))
        self.init_check = False

    def forward(self, x):
        xp = cuda_backend.get_array_module(x)

        if not self.init_check and xp is not np:
            self.beta.to_gpu()
            self.init_check = True

        elif not self.init_check:
            self.init_check = True

        return functions.dynamic_swish(x, self.beta)
