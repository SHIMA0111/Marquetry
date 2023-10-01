from marquetry import Function


class Identity(Function):
    """Identity function.

        The Identity class simply returns its input unchanged. It is often used as a building block
        in neural networks to maintain the same input shape and value.
    """

    def forward(self, x):
        y = x.copy()

        self.retain_inputs(())
        return y

    def backward(self, input_data, grad_y):
        return grad_y


def identity(x):
    """Apply the Identity function to an input.

        The Identity function takes an input and returns it unchanged. It is often used as a
        simple way to pass data through a function without any modification.

        Args:
            x (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The input data to be passed through the Identity function.

        Returns:
            Container: A new Variable object containing the input data.

        Examples:
            >>> import marquetry as mq
            >>> input_data = mq.array([1, 2, 3])
            >>> result = mq.functions.identity(input_data)
            >>> print(result)
            container([1, 2, 3])
    """
    return Identity()(x)
