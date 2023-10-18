from marquetry import cuda_backend
from marquetry import Function
from marquetry import functions
from marquetry.utils import sigmoid_array


class GLU(Function):
    """Gated Linear Units (GLU) Function.

        This class implements the Gated Linear Units activation function,
        which is used in neural networks.
    """

    def __init__(self, axis):
        self.axis = axis
        self.sigmoid = None

    def forward(self, x):
        if x.ndim - 1 < self.axis:
            raise RuntimeError("input data has no {} axis.".format(self.axis))

        axis_dim = x.shape[self.axis]

        if axis_dim % 2 != 0:
            raise RuntimeError("GLU requires the specified axis's data dimension being even, "
                               "but the size of the input {} dim is {}"
                               .format(range(x.ndim)[self.axis], axis_dim))

        xp = cuda_backend.get_array_module(x)

        a, b = xp.split(x, 2, axis=self.axis)
        sigmoid = sigmoid_array(b)

        y = a * sigmoid

        self.sigmoid = sigmoid

        return y

    def backward(self, x, grad_y):
        x = x[0]
        a, b = functions.split(x, 2, axis=self.axis)

        grad_a = self.sigmoid * grad_y[0]
        grad_b = self.sigmoid * (1 - self.sigmoid) * a * grad_y[0]

        grad_x = functions.concat((grad_a, grad_b), axis=self.axis)

        return grad_x


def glu(x, axis=-1):
    """Gated Linear Units (GLU) Function.

        This function implements the Gated Linear Units activation function,
        which is used in neural networks.

        This function is obtained by:

        :math:`GLU(a, b) = a \cdot \sigma (b)`

        (`a` and `b` are splited from x as the same size. From this specification,
        the input data's specified axis should be even.)

        Args:
            x (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                Input container or ndarray that is float array.
            axis (int): The direction for splitting the `a` and `b`

        Examples:
            >>> x = np.array([[1, 2, 3], [2, 4, 6]])
            >>> output = glu(x)
            ..
            RuntimeError: GLU requires the specified axis's data dimension being even, but the size of the input 1 dim is 3
            >>> output = glu(x, axis=0)
            >>> print(output)
            container([[0.88079708 1.96402758 2.99258213]])

    """

    return GLU(axis)(x)
