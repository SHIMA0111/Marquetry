from marquetry import Function


class Reshape(Function):
    """Reshape an input array to the specified shape."""
    def __init__(self, shape):
        self.shape = shape

        self.x_shape = None

    def forward(self, x):
        self.x_shape = x.shape
        y = x.reshape(self.shape)

        self.retain_inputs(())
        return y

    def backward(self, x, grad_y):
        return reshape(grad_y[0], self.x_shape)


def reshape(x, shape):
    """This function reshapes an input array to the specified shape.

        Args:
            x (:class:`marquetry.Variable` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The input array to be reshaped.
            shape (tuple or int): The target shape to which the input array should be reshaped.

        Returns:
            A :class:`marquetry.Variable` object which is the result of reshaping
            the input array to the specified shape.

        Examples:
            >>> x = np.arange(0, 3).reshape(1, 3)
            array([[0, 1, 2]])
            >>> reshape(x, shape=3)
            matrix([0 1 2])
    """

    return Reshape(shape)(x)
