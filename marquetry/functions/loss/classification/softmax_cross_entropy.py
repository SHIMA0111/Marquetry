from marquetry import cuda_backend
from marquetry import Function
from marquetry import functions
from marquetry import utils


class SoftmaxCrossEntropy(Function):
    """Calculate the Softmax Cross-Entropy loss between predicted values and true labels.

        This function calculates the Softmax Cross-Entropy loss,
        which measures the difference between the predicted values (scores) and
        the true labels for multi-class classification tasks.
        It takes the softmax of the input values to compute class probabilities and then
        compares them to the true labels.

        Note:
            Generally, you don't need to execute ``forward`` and ``backward`` method manually.
            You should use only ``__call__`` method.
    """
    def forward(self, x, t):
        xp = cuda_backend.get_array_module(x)
        batch_size = x.shape[0]
        log_z = utils.log_sum_exp(x, axis=1)
        log_p = x - log_z
        log_p = log_p[xp.arange(batch_size), t.ravel()]
        y = -log_p.sum() / xp.float32(batch_size)

        return y

    def backward(self, inputs, grad_y):
        x, t = inputs
        grad_y = grad_y[0]

        batch_size, data_dim = x.shape

        grad_y *= 1 / batch_size
        y = functions.softmax(x)
        xp = cuda_backend.get_array_module(t)
        if y.size != t.size:
            # convert class num to one-hot
            t_onehot = xp.eye(data_dim, dtype=t.dtype)[t]
        else:
            t_onehot = t

        y = (y - t_onehot) * grad_y
        return y


def softmax_cross_entropy(x, t):
    """Calculate the Softmax Cross-Entropy loss between predicted values and true labels.


        This function calculates the Softmax Cross-Entropy loss,
        which measures the difference between the predicted values(scores) and
        the true labels for multi-class classification tasks.
        It takes the softmax of the input values to compute class probabilities and then
        compares them to the true labels.

        Args:
            x (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The predicted values or scores.
            t (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The true labels.

        Returns:
            :class:`marquetry.Container`: The Softmax Cross-Entropy loss.
    """

    return SoftmaxCrossEntropy()(x, t)
