from marquetry import Function
from marquetry import utils
from marquetry import functions


class LogSoftmax(Function):
    """Log Softmax.

        Note:
            Generally, you don't need to execute ``forward`` and ``backward`` method manually.
            You should use only ``__call__`` method.
    """
    def __init__(self, axis):
        self.axis = axis

    def forward(self, x):
        log_z = utils.log_sum_exp(x, self.axis)
        y = x - log_z

        self.retain_inputs(())
        self.retain_outputs((0,))
        return y

    def backward(self, x, grad_y):
        y = self.output_data[0]
        grad_y = grad_y[0]

        grad_x = grad_y - functions.exp(y) * grad_y.sum(axis=self.axis, keepdims=True)

        return grad_x


def log_softmax(x, axis=1):
    """Log Softmax.

        :math:`f(x) = log{exp(x) / Σexp(x)}`
            - The ``log`` is a natural logarithm.

        Args:
            x (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                Input container that is float array.
            axis (int): The sum calc axis.

        Returns:
            marquetry.Container: Output container. A float array.

        Examples:

            >>> x = np.array([[-1, 0], [2, -3], [-2, 1]], 'f')
            >>> x
            array([[-1.,  0.],
                    [ 2., -3.],
                    [-2.,  1.]], dtype=float32)
            >>> log_softmax(x, axis=1)
            container([[-1.3132616  -0.31326166]
                       [-0.0067153  -5.0067153 ]
                       [-3.0485873  -0.04858732]])

    """
    return LogSoftmax(axis)(x)
