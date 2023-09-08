def reshape_sum_backward(grad_y, x_shape, axis, keepdims):
    """Reshape gradient appropriately for sum's backward."""

    ndim = len(x_shape)
    tupled_axis = axis
    if axis is None:
        tupled_axis = None
    elif not isinstance(axis, tuple):
        tupled_axis = (axis,)

    if not (ndim == 0 or tupled_axis is None or keepdims):
        actual_axis = [a if a >= 0 else a + ndim for a in tupled_axis]
        shape = list(grad_y.shape)
        for a in sorted(actual_axis):
            shape.insert(a, 1)
    else:
        shape = grad_y.shape

    grad_y = grad_y.reshape(shape)
    return grad_y


def max_backward_shape(x, axis):
    """Calc the origin input data's shape for max(min) backward."""

    if axis is None:
        axis = range(x.ndim)
    elif isinstance(axis, int):
        axis = (axis,)
    else:
        axis = axis

    shape = [s if ax not in axis else 1 for ax, s in enumerate(x.shape)]

    return shape
