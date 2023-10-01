from marquetry import as_container
from marquetry import cuda_backend
from marquetry import Function
from marquetry import functions


class SigmoidCrossEntropy(Function):
    """Calculate the Sigmoid Cross-Entropy loss between predicted values and true labels.

        This class defines a function that computes the Sigmoid Cross-Entropy loss,
        which is commonly used as the loss function in binary classification problems.
        It measures the dissimilarity between the predicted values and the true binary labels.

        Note:
            Generally, you don't need to execute ``forward`` and ``backward`` method manually.
            You should use only ``__call__`` method.
    """
    def __init__(self):
        self.batch_size = None

    def forward(self, x, t):
        if x.ndim != t.ndim:
            t = t.reshape(*x.shape)

        xp = cuda_backend.get_array_module(x)

        batch_size = x.shape[0] if x.ndim != 1 else len(x)
        p = xp.exp(xp.minimum(0, x)) / (1 + xp.exp(-xp.abs(x)))
        p = xp.clip(p, 1e-15, .999)
        tlog_p = t * xp.log(p) + (1 - t) * xp.log(1 - p)
        y = -1 * tlog_p.sum() / batch_size

        self.batch_size = batch_size

        return y

    def backward(self, inputs, grad_y):
        x, t = inputs
        if x.ndim != t.ndim:
            t = t.reshape(*x.shape)
        y = functions.sigmoid(x)

        batch_size = self.batch_size

        # grad_x = -(1 / batch_size) * ((t / y) - ((1 - t) / (1 - y))) * (y * (1 - y)) * grad_y
        grad_x = -(1 / batch_size) * (t * (1 - y) - y * (1 - t)) * grad_y[0]

        return grad_x


def sigmoid_cross_entropy(x, t):
    """Calculate the Sigmoid Cross-Entropy loss between predicted values and true binary labels.

        This function defines that computes the Sigmoid Cross-Entropy loss,
        which is commonly used as the loss function in binary classification problems.
        It measures the dissimilarity between the predicted values and the true binary labels.

        Args:
            x (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The predicted values.
            t (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The true binary labels.

        Returns:
            :class:`marquetry.Container`: The Sigmoid Cross-Entropy loss
                between the predicted values and true binary labels.
    """

    return SigmoidCrossEntropy()(x, t)


def simple_sigmoid_cross_entropy(x, t):
    if x.ndim != t.ndim:
        t = t.reshape(*x.shape)

    x, t = as_container(x), as_container(t)
    batch_size = len(x)
    p = functions.sigmoid(x)
    p = functions.clip(p, 1e-15, 0.99)
    tlog_p = t * functions.log(p) + (1 - t) * functions.log(1 - p)
    y = -1 * sum(tlog_p) / batch_size
    return y
