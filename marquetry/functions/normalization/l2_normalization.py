from marquetry import cuda_backend
from marquetry import Function
from marquetry import functions


class L2Normalization(Function):
    """L2Normalization performs as L2 normalization on input data.

        L2 normalization is a technique used to scale input data such
        that the L2 (Euclidean) norm of the vector becomes 1.
        It can be useful in various machine learning applications,
        such as feature scaling and neural network training.

        This implementation is part of the Marquetry framework and is used for deep learning tasks.
    """

    def __init__(self, eps, axis):
        self.eps = eps
        self.axis = axis

    def forward(self, x):
        xp = cuda_backend.get_array_module(x)

        norm = xp.sqrt(xp.square(x).sum(axis=self.axis, keepdims=True)) + self.eps
        return x / norm

    def backward(self, x, grad_y):
        x = x[0]
        grad_y = grad_y[0]

        norm_without_eps = functions.sqrt(functions.square(x).sum(axis=self.axis))
        norm = norm_without_eps + self.eps
        norm = functions.broadcast_to(functions.unsqueeze(norm, axis=self.axis), grad_y.shape)

        x_grad_y_reduce = functions.sum((x * grad_y), axis=self.axis)
        x_grad_y_reduce /= norm_without_eps
        x_grad_y_reduce = functions.broadcast_to(
            functions.unsqueeze(x_grad_y_reduce, self.axis), grad_y.shape)
        grad_x = grad_y * norm - x_grad_y_reduce * x
        grad_x = grad_x / norm ** 2

        return grad_x


def l2_normalization(x, eps=1e-15, axis=1):
    """L2Normalization performs as L2 normalization on input data.

        L2 normalization is a technique used to scale input data such
        that the L2 (Euclidean) norm of the vector becomes 1.
        It can be useful in various machine learning applications,
        such as feature scaling and neural network training.

        This implementation is part of the Marquetry framework and is used for deep learning tasks.

        Args:
            x (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The input tensor.
            eps (float): A small value to prevent division by zero.
                Default is 1e-15.
            axis (int or tuple of ints): Axis or Axes for computing the L2 norm

        Returns:
            marquetry.Container: The normalized input container.

    """

    return L2Normalization(eps, axis)(x)
