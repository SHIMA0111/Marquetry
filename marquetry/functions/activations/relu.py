from marquetry import cuda_backend
from marquetry import Function


class ReLU(Function):
    """Rectifier Linear Unit."""
    def forward(self, x):
        xp = cuda_backend.get_array_module(x)
        y = xp.maximum(x, 0.0)

        return y

    def backward(self, x, grad_y):
        x, = x
        mask = x > 0
        grad_x = grad_y[0] * mask

        return grad_x


def relu(x):
    """Rectified Linear Unit function.

    f(x) = {x if x >= 0, 0 if x < 0}

    Args:
        x (:class:`marquetry.Variable` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
            Input variable that is float array.

    Returns:
        marquetry.Variable: Output variable. A float array.

    Examples:

        >>> x = np.array([[-1, 0], [2, -3], [-2, 1]], 'f')
        >>> x
        array([[-1.,  0.],
               [ 2., -3.],
               [-2.,  1.]], dtype=float32)
        >>> relu(x)
        matrix([[0. 0.]
                [2. 0.]
                [0. 1.]])

    """
    return ReLU()(x)
