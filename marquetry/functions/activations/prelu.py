import inspect

import numpy as np

from marquetry import cuda_backend
from marquetry import Function
from marquetry import functions


class PReLU(Function):
    """Parametric Rectified Linear Units

        Note:
            Generally, you don't need to execute ``forward`` and ``backward`` method manually.
            You should use only ``__call__`` method.
    """

    def __init__(self):
        self.mask = None
        self.alpha_shape = None
        self.ex_alpha_shape = None

    def forward(self, x, alpha: np.ndarray):
        if alpha.size != 1 and x.ndim <= 1:
            raise RuntimeError("When `x` is 0 or 1 dim data, alpha should be 1 only.")
        elif not (alpha.size == 1 or alpha.size == x.shape[1]):
            raise RuntimeError("Learnable parameter `alpha` should be 1 or data 2nd dim size, "
                               "your input has {} dim but the alpha is {}".format(x.shape[1], alpha.size))

        xp = cuda_backend.get_array_module(x)

        y = x.copy()

        mask = alpha.copy()

        x_dim = x.ndim
        axes = list(range(x_dim))
        axes.remove(1)
        mask = xp.expand_dims(mask, tuple(axes))
        mask = xp.broadcast_to(mask, x.shape)
        mask = mask * (x <= 0).astype(x.dtype)
        x_map = (x > 0).astype(x.dtype)
        mask += x_map

        y *= mask

        self.mask = mask
        self.alpha_shape = alpha.shape
        self.ex_alpha_shape = xp.expand_dims(alpha, tuple(axes)).shape

        self.retain_inputs((0,))

        return y

    def backward(self, inputs, grad_y):
        x, _ = inputs
        mask = self.mask

        grad_x = grad_y[0] * mask

        mask = (x <= 0).astype(grad_y[0].dtype)
        grad_alpha = x * mask * grad_y[0]
        grad_alpha = functions.sum_to(grad_alpha, self.ex_alpha_shape)
        grad_alpha = grad_alpha.squeeze().reshape(self.alpha_shape)

        return grad_x, grad_alpha


def prelu(x, alpha):
    """Parametric Rectified Linear Unit function.

        This function is improved version of the ReLU(Rectified Linear Unit).
        ``Parametric`` means the function having learnable parameter.

        This function is obtained by:

        if x >= 0, :math:`f(x) = x` || if x < 0, :math:`f(x) = param * x`
            - The ``param`` is a small learnable value.

        Args:
            x (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                Input container or ndarray that is float array.
            alpha (float or :class:`numpy.ndarray`): The ``param`` is a small learnable value.
                alpha should be 1 size or the same size of the input's axis 1.
                If another values are inputed as alpha, raise exception.

        Returns:
            marquetry.Container: Output container. A float array.

        Caution:
            This activation has learnable parameter
            (if the ``alpha`` is constant value, it is the same as :meth:`leaky_relu`).
            In generally, we suggest to use :class:`marquetry.layers.PReLU` instead of this.

            The layers object manages the leanable parameter itself so in PReLU, the `alpha` is managed by the object.

        Examples:

            >>> x = np.array([[-1, 0], [2, -3], [-2, 1]], 'f')
            >>> x
            array([[-1.,  0.],
                   [ 2., -3.],
                   [-2.,  1.]], dtype=float32)
            >>
            >>> prelu(x, alpha=0.25)
            container([[-0.25  0.  ]
                       [ 2.   -0.75]
                       [-0.5   1.  ]])

    """

    return PReLU()(x, alpha)
