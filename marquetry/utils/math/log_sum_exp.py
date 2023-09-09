from marquetry import cuda_backend


def log_sum_exp(x, axis=1):
    """Compute the log-sum-exp of an array along a specified axis.

        This function computes the log-sum-exp of an input array along the specified axis.
        It first subtracts the maximum value along the axis from the input array to prevent overflow,
        exponentiates the result, computes the sum along the axis, and finally takes the logarithm of the sum.

        logsumexp is used for preventing over/underflow the exponential generally.

        Args:
            x (:class:`numpy.ndarray`, or :class:`cupy.ndarray`):
                The input array.
            axis (int): The axis along which the log-sum-exp should be computed.
                Default is axis 1.

        Returns:
            :class:`numpy.ndarray` or :class:`cupy.ndarray`:
                The log-sum-exp of the input array along the specified axis.
    """

    xp = cuda_backend.get_array_module(x)
    x_max = x.max(axis=axis, keepdims=True)
    x_norm = x - x_max
    xp.exp(x_norm, out=x_norm)
    sum_exp = x_norm.sum(axis=axis, keepdims=True)
    xp.log(sum_exp, out=sum_exp)

    y = x_max + sum_exp

    return y
