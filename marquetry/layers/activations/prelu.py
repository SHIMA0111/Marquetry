import numpy as np

from marquetry import cuda_backend
from marquetry import functions
from marquetry import Layer
from marquetry import Parameter


class PReLU(Layer):
    """Parametric Rectified Linear Units layer.

        This function is improved version of the ReLU(Rectified Linear Unit).
        ``Parametric`` means the function having learnable parameter.
        This layer object helps you to learn the parameter of the PReLU.
        In generally case, you can use this as activation function.

        PReLU's details: :class:`marquetry.functions.prelu`

        Args:
            num_parameter (int): The number of the parameter which allows `1` or the second dim of the input data.
                Default is `1`.
            init (float): The init param value. Default is 0.01.

        Attributes:
            alpha(float or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The learnable parameter in this activation function.

        Examples:
            >>> x = np.random.randn(4, 5)
            >>> print(x)
            [[ 1.1667476   1.05054986  0.79685889 -0.18952283  0.15200651]
             [ 0.2586019  -0.26523651  2.06032381 -0.74969459 -0.18439514]
             [ 1.42829195  0.21610936 -0.02963907 -0.45781017  1.55797971]
             [ 1.16250809 -0.08423502 -0.81557292 -0.00216238  0.6291491 ]]
            >>> prelu = PReLU()
            >>> y = prelu(x)
            >>> print(y)
            container([[ 1.16674760e+00  1.05054986e+00  7.96858893e-01 -1.89522835e-03 1.52006511e-01]
                       [ 2.58601904e-01 -2.65236510e-03  2.06032381e+00 -7.49694591e-03 -1.84395144e-03]
                       [ 1.42829195e+00  2.16109356e-01 -2.96390676e-04 -4.57810171e-03 1.55797971e+00]
                       [ 1.16250809e+00 -8.42350188e-04 -8.15572916e-03 -2.16238255e-05 6.29149104e-01]])
    """

    def __init__(self, num_parameter=1, init=0.01):
        super().__init__()

        alpha = np.zeros(num_parameter) + init
        self.alpha = Parameter(alpha)

    def forward(self, x):
        xp = cuda_backend.get_array_module(x)
        if xp is not np:
            self.alpha.to_gpu()

        return functions.prelu(x, self.alpha)
